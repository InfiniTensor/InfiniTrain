#include "infini_train/include/nn/parallel/context_parallel.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/nn/parallel/work.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

thread_local int cp_rank = 0;

namespace {

const ProcessGroup *GetCPGroup(const std::shared_ptr<Tensor> &tensor) {
    auto cp_size = global::GetContextParallelSize();
    CHECK_GT(cp_size, 0);
    if (cp_size == 1) {
        return nullptr;
    }
    return ProcessGroupFactory::Instance(tensor->GetDevice().type())
        ->Get(GetContextParallelProcessGroupName(tensor->GetDevice().Rank().GlobalRank()));
}

// Comm Kernel Call Functions
std::shared_ptr<Tensor> GatherAlongFirstDim(const std::shared_ptr<Tensor> &tensor, const ProcessGroup *cp_group) {
    const int cp_size = global::GetContextParallelSize();
    auto output_shape = tensor->Dims();
    output_shape[0] *= cp_size;
    auto output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), tensor->GetDevice());
    cp_group->AllGather(output, tensor, false);
    return output;
}

std::shared_ptr<Tensor> ReduceScatterAlongFirstDim(const std::shared_ptr<Tensor> &tensor,
                                                   const ProcessGroup *cp_group) {
    const int cp_size = global::GetContextParallelSize();
    auto output_shape = tensor->Dims();
    CHECK_EQ(output_shape[0] % cp_size, 0) << "First dimension must be divisible by CP size";
    output_shape[0] /= cp_size;
    auto output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), tensor->GetDevice());
    cp_group->ReduceScatter(output, tensor, function::ReduceOpType::kSum, false);
    return output;
}

// Attention Helper Functions
std::shared_ptr<Tensor> NewZeroTensorLike(const std::shared_ptr<Tensor> &tensor) {
    auto output = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), tensor->GetDevice());
    output->Fill(0.0f);
    return output;
}

std::shared_ptr<Tensor> RepeatKVHeads(const std::shared_ptr<Tensor> &x, int64_t n_rep) {
    if (n_rep == 1) {
        return x;
    }

    const auto &shape = x->Dims();
    const int64_t B = shape[0], H = shape[1], T = shape[2], D = shape[3];
    return x->View({B, H, 1, T, D})->RepeatInterleave(n_rep, 2)->Contiguous()->View({B, H * n_rep, T, D});
}

std::shared_ptr<Tensor> SumRepeatedKVHeads(const std::shared_ptr<Tensor> &x, int64_t n_rep) {
    if (n_rep == 1) {
        return x;
    }

    const auto &shape = x->Dims();
    const int64_t B = shape[0], H = shape[1], T = shape[2], D = shape[3];
    CHECK_EQ(H % n_rep, 0);
    return x->View({B, H / n_rep, n_rep, T, D})->Sum(2);
}

struct RingAttentionForwardResult {
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> softmax_max;
    std::shared_ptr<Tensor> softmax_sum;
};

RingAttentionForwardResult RingOnlineAttentionForward(const std::shared_ptr<Tensor> &q,
                                                      const std::shared_ptr<Tensor> &k_local,
                                                      const std::shared_ptr<Tensor> &v_local,
                                                      const std::shared_ptr<Tensor> &mask, bool mask_true_means_invalid,
                                                      float scale, int64_t n_rep) {
    const int cp_size = global::GetContextParallelSize();
    CHECK_GT(cp_size, 1);
    CHECK(mask) << "CP ring attention expects a causal mask.";

    auto cp_group = GetCPGroup(q);
    CHECK_NOTNULL(cp_group);
    const int rank = cp_group->GetGroupRank(q->GetDevice().Rank().GlobalRank());
    const int send_to = (rank + 1) % cp_size;
    const int recv_from = (rank - 1 + cp_size) % cp_size;

    const int64_t local_t = q->Dims()[2];
    CHECK_EQ(k_local->Dims()[2], local_t);
    CHECK_EQ(v_local->Dims()[2], local_t);
    CHECK_EQ(q->Dims()[1], k_local->Dims()[1] * n_rep);
    CHECK_EQ(q->Dims()[1], v_local->Dims()[1] * n_rep);

    auto current_k = k_local;
    auto current_v = v_local;
    std::shared_ptr<Tensor> running_max;
    std::shared_ptr<Tensor> running_sum;
    std::shared_ptr<Tensor> running_out;

    for (int step = 0; step < cp_size; ++step) {
        std::shared_ptr<Tensor> next_k;
        std::shared_ptr<Tensor> next_v;
        std::shared_ptr<Work> send_work;
        std::shared_ptr<Work> recv_work;
        if (step + 1 < cp_size) {
            next_k = std::make_shared<Tensor>(k_local->Dims(), k_local->Dtype(), k_local->GetDevice());
            next_v = std::make_shared<Tensor>(v_local->Dims(), v_local->Dtype(), v_local->GetDevice());
            if (rank % 2 == 0) {
                send_work = cp_group->Send({current_k, current_v}, send_to, true);
                recv_work = cp_group->Recv({next_k, next_v}, recv_from, true);
            } else {
                recv_work = cp_group->Recv({next_k, next_v}, recv_from, true);
                send_work = cp_group->Send({current_k, current_v}, send_to, true);
            }
        }

        const int owner = (rank - step + cp_size) % cp_size;
        const int64_t kv_start = static_cast<int64_t>(owner) * local_t;
        const int64_t kv_end = kv_start + local_t;

        auto k_for_attn = RepeatKVHeads(current_k, n_rep);
        auto v_for_attn = RepeatKVHeads(current_v, n_rep);
        auto scores = q->Matmul(k_for_attn->Transpose(-2, -1)) * scale;
        auto mask_chunk = mask->Slice(3, kv_start, kv_end);
        auto invalid_mask = mask_true_means_invalid ? mask_chunk : (mask_chunk == 0);
        scores = scores->MaskedFill(invalid_mask, std::numeric_limits<float>::lowest());

        auto chunk_max = scores->Max(-1, true);
        auto probs = (scores - chunk_max)->Exp()->MaskedFill(invalid_mask, 0.0f);
        auto chunk_sum = probs->Sum(-1, true);
        auto chunk_out = probs->Matmul(v_for_attn);

        if (!running_out) {
            running_max = chunk_max;
            running_sum = chunk_sum;
            running_out = chunk_out;
        } else {
            auto new_max
                = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{running_max, chunk_max}, -1)->Max(-1);
            auto old_scale = (running_max - new_max)->Exp();
            auto new_scale = (chunk_max - new_max)->Exp();
            running_sum = running_sum * old_scale + chunk_sum * new_scale;
            running_out = running_out * old_scale + chunk_out * new_scale;
            running_max = new_max;
        }

        if (recv_work) {
            recv_work->WaitNonBlocking();
            send_work->WaitNonBlocking();
            current_k = next_k;
            current_v = next_v;
        }
    }

    return {.output = running_out / running_sum, .softmax_max = running_max, .softmax_sum = running_sum};
}

std::shared_ptr<Tensor> ApplyAttention(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                       const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask,
                                       bool mask_true_means_invalid, float scale) {
    auto scores = q->Matmul(k->Transpose(-2, -1)) * scale;
    if (mask) {
        auto invalid_mask = mask_true_means_invalid ? mask : (mask == 0);
        scores = scores->MaskedFill(invalid_mask, std::numeric_limits<float>::lowest());
    }
    auto probs = nn::function::Softmax(scores, -1);
    return probs->Matmul(v);
}

// Autograd Function Definitions
class GatherFromCPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFromCPRegionFunction";

    explicit GatherFromCPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        auto input = input_tensors[0];
        if (global::GetContextParallelSize() == 1) {
            return {std::make_shared<Tensor>(*input)};
        }
        auto cp_group = GetCPGroup(input);
        // FIXME(zbl): Megatron keeps sequence as dim 0. InfiniTrain uses [B, H, S, D], so move only
        //              the sequence dimension to dim 0 before the CP gather.
        return {GatherAlongFirstDim(input->Transpose(0, 2), cp_group)->Transpose(0, 2)->Contiguous()};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        if (global::GetContextParallelSize() == 1) {
            return {std::make_shared<Tensor>(*grad_outputs[0])};
        }
        auto cp_group = GetCPGroup(grad_outputs[0]);
        // FIXME(zbl): See Forward() for the [B, H, S, D] sequence-first bridge.
        return {ReduceScatterAlongFirstDim(grad_outputs[0]->Transpose(0, 2), cp_group)->Transpose(0, 2)->Contiguous()};
    }
};

class GatherFromCPHeadRegion : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFromCPHeadRegionFunction";

    explicit GatherFromCPHeadRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        auto input = input_tensors[0];
        if (global::GetContextParallelSize() == 1) {
            return {std::make_shared<Tensor>(*input)};
        }
        auto cp_group = GetCPGroup(input);
        // A2A CP shards heads across CP ranks. ProcessGroup all-gather works on dim 0, so move heads there.
        return {GatherAlongFirstDim(input->Transpose(0, 1), cp_group)->Transpose(0, 1)->Contiguous()};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        if (global::GetContextParallelSize() == 1) {
            return {std::make_shared<Tensor>(*grad_outputs[0])};
        }
        auto cp_group = GetCPGroup(grad_outputs[0]);
        return {ReduceScatterAlongFirstDim(grad_outputs[0]->Transpose(0, 1), cp_group)->Transpose(0, 1)->Contiguous()};
    }
};

class AttnWithCPAndKVP2P : public autograd::Function {
public:
    static constexpr char kType[] = "AttnWithCPAndKVP2PFunction";

    AttnWithCPAndKVP2P(bool mask_true_means_invalid, float scale, int64_t n_rep)
        : autograd::Function(kType), mask_true_means_invalid_(mask_true_means_invalid), scale_(scale), n_rep_(n_rep) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        CHECK_EQ(input_tensors.size(), 4);
        auto result = RingOnlineAttentionForward(input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3],
                                                 mask_true_means_invalid_, scale_, n_rep_);
        softmax_max_ = result.softmax_max;
        softmax_sum_ = result.softmax_sum;
        return {result.output};
    }

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override {
        ctx_.SaveForBackward({input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], output_tensors[0],
                              softmax_max_, softmax_sum_});
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        CHECK_EQ(grad_outputs.size(), 1);
        auto saved_tensors = ctx_.GetSavedTensors();
        CHECK_EQ(saved_tensors.size(), 7);

        const auto &q = saved_tensors[0];
        const auto &k_local = saved_tensors[1];
        const auto &v_local = saved_tensors[2];
        const auto &mask = saved_tensors[3];
        const auto &output = saved_tensors[4];
        const auto &softmax_max = saved_tensors[5];
        const auto &softmax_sum = saved_tensors[6];
        const auto &grad_output = grad_outputs[0];

        const int cp_size = global::GetContextParallelSize();
        CHECK_GT(cp_size, 1);

        auto cp_group = GetCPGroup(q);
        CHECK_NOTNULL(cp_group);
        const int rank = cp_group->GetGroupRank(q->GetDevice().Rank().GlobalRank());
        const int send_to = (rank + 1) % cp_size;
        const int recv_from = (rank - 1 + cp_size) % cp_size;

        const int64_t local_t = q->Dims()[2];
        CHECK_EQ(k_local->Dims()[2], local_t);
        CHECK_EQ(v_local->Dims()[2], local_t);
        CHECK_EQ(q->Dims()[1], k_local->Dims()[1] * n_rep_);
        CHECK_EQ(q->Dims()[1], v_local->Dims()[1] * n_rep_);

        auto current_k = k_local;
        auto current_v = v_local;
        auto current_grad_k = NewZeroTensorLike(k_local);
        auto current_grad_v = NewZeroTensorLike(v_local);
        std::shared_ptr<Tensor> grad_q;
        const auto softmax_delta = (grad_output * output)->Sum(-1, true);

        for (int step = 0; step < cp_size; ++step) {
            const int owner = (rank - step + cp_size) % cp_size;
            const int64_t kv_start = static_cast<int64_t>(owner) * local_t;
            const int64_t kv_end = kv_start + local_t;

            auto k_for_attn = RepeatKVHeads(current_k, n_rep_);
            auto v_for_attn = RepeatKVHeads(current_v, n_rep_);
            auto scores = q->Matmul(k_for_attn->Transpose(-2, -1)) * scale_;
            auto mask_chunk = mask->Slice(3, kv_start, kv_end);
            auto invalid_mask = mask_true_means_invalid_ ? mask_chunk : (mask_chunk == 0);
            scores = scores->MaskedFill(invalid_mask, std::numeric_limits<float>::lowest());

            auto probs = (scores - softmax_max)->Exp()->MaskedFill(invalid_mask, 0.0f) / softmax_sum;
            auto grad_v_repeated = probs->Transpose(-2, -1)->Matmul(grad_output);
            auto grad_probs = grad_output->Matmul(v_for_attn->Transpose(-2, -1));
            auto grad_scores = probs * (grad_probs - softmax_delta);
            auto grad_k_repeated = grad_scores->Transpose(-2, -1)->Matmul(q) * scale_;

            current_grad_k = current_grad_k + SumRepeatedKVHeads(grad_k_repeated, n_rep_);
            current_grad_v = current_grad_v + SumRepeatedKVHeads(grad_v_repeated, n_rep_);

            std::shared_ptr<Work> send_work;
            std::shared_ptr<Work> recv_work;
            std::shared_ptr<Tensor> next_k;
            std::shared_ptr<Tensor> next_v;
            std::shared_ptr<Tensor> next_grad_k;
            std::shared_ptr<Tensor> next_grad_v;

            if (step + 1 < cp_size) {
                next_k = std::make_shared<Tensor>(k_local->Dims(), k_local->Dtype(), k_local->GetDevice());
                next_v = std::make_shared<Tensor>(v_local->Dims(), v_local->Dtype(), v_local->GetDevice());
                next_grad_k = NewZeroTensorLike(k_local);
                next_grad_v = NewZeroTensorLike(v_local);
                if (rank % 2 == 0) {
                    send_work = cp_group->Send({current_k, current_v, current_grad_k, current_grad_v}, send_to, true);
                    recv_work = cp_group->Recv({next_k, next_v, next_grad_k, next_grad_v}, recv_from, true);
                } else {
                    recv_work = cp_group->Recv({next_k, next_v, next_grad_k, next_grad_v}, recv_from, true);
                    send_work = cp_group->Send({current_k, current_v, current_grad_k, current_grad_v}, send_to, true);
                }
            } else {
                next_grad_k = NewZeroTensorLike(k_local);
                next_grad_v = NewZeroTensorLike(v_local);
                if (rank % 2 == 0) {
                    send_work = cp_group->Send({current_grad_k, current_grad_v}, send_to, true);
                    recv_work = cp_group->Recv({next_grad_k, next_grad_v}, recv_from, true);
                } else {
                    recv_work = cp_group->Recv({next_grad_k, next_grad_v}, recv_from, true);
                    send_work = cp_group->Send({current_grad_k, current_grad_v}, send_to, true);
                }
            }

            auto grad_q_chunk = grad_scores->Matmul(k_for_attn) * scale_;
            grad_q = grad_q ? grad_q + grad_q_chunk : grad_q_chunk;

            recv_work->WaitNonBlocking();
            send_work->WaitNonBlocking();

            if (step + 1 < cp_size) {
                current_k = next_k;
                current_v = next_v;
                current_grad_k = next_grad_k;
                current_grad_v = next_grad_v;
            } else {
                current_grad_k = next_grad_k;
                current_grad_v = next_grad_v;
            }
        }

        return {grad_q, current_grad_k, current_grad_v, nullptr};
    }

private:
    bool mask_true_means_invalid_ = false;
    float scale_ = 1.0f;
    int64_t n_rep_ = 1;
    std::shared_ptr<Tensor> softmax_max_;
    std::shared_ptr<Tensor> softmax_sum_;
};

std::shared_ptr<Tensor> GatherFromCPHeadRegionFunc(const std::shared_ptr<Tensor> &input) {
    if (global::GetContextParallelSize() == 1) {
        return input;
    }
    return std::make_shared<GatherFromCPHeadRegion>()->Apply({input})[0];
}

} // namespace

// CP State Helper Functions
int GetContextParallelRank() { return cp_rank; }

int64_t GetContextParallelSequenceStart(int64_t local_sequence_length) {
    return static_cast<int64_t>(GetContextParallelRank()) * local_sequence_length;
}

// CP Communication Helper Functions
std::shared_ptr<Tensor> SliceAlongCPRegionFunc(const std::shared_ptr<Tensor> &input, int64_t dim) {
    const int cp_size = global::GetContextParallelSize();
    if (cp_size == 1) {
        return input;
    }

    int64_t normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim += static_cast<int64_t>(input->Dims().size());
    }
    CHECK_GE(normalized_dim, 0);
    CHECK_LT(normalized_dim, static_cast<int64_t>(input->Dims().size()));
    const auto dim_size = input->Dims()[normalized_dim];
    CHECK_EQ(dim_size % cp_size, 0) << "Sequence dimension must be divisible by CP size";
    const int64_t local_size = dim_size / cp_size;
    const int64_t start = GetContextParallelSequenceStart(local_size);
    return input->Slice(normalized_dim, start, start + local_size, 1)->Contiguous();
}

std::shared_ptr<Tensor> GatherFromCPRegionFunc(const std::shared_ptr<Tensor> &input) {
    if (global::GetContextParallelSize() == 1) {
        return input;
    }
    return std::make_shared<GatherFromCPRegion>()->Apply({input})[0];
}

// CP Attention Backend Functions
std::shared_ptr<Tensor> AttnFuncWithCPAndKVP2P(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                               const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask,
                                               bool mask_true_means_invalid, float scale, int64_t n_rep) {
    return std::make_shared<AttnWithCPAndKVP2P>(mask_true_means_invalid, scale, n_rep)->Apply({q, k, v, mask})[0];
}

std::shared_ptr<Tensor> AttnFuncWithCPAndKVAllGather(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                                     const std::shared_ptr<Tensor> &v,
                                                     const std::shared_ptr<Tensor> &mask, bool mask_true_means_invalid,
                                                     float scale, int64_t n_rep) {
    auto gathered_k = GatherFromCPRegionFunc(k);
    auto gathered_v = GatherFromCPRegionFunc(v);
    auto k_for_attn = RepeatKVHeads(gathered_k, n_rep);
    auto v_for_attn = RepeatKVHeads(gathered_v, n_rep);
    return ApplyAttention(q, k_for_attn, v_for_attn, mask, mask_true_means_invalid, scale);
}

std::shared_ptr<Tensor> AttnFuncWithCPAndQKVOA2A(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                                 const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask,
                                                 bool mask_true_means_invalid, float scale, int64_t n_rep) {
    const int cp_size = global::GetContextParallelSize();
    const int rank = GetCPGroup(q)->GetGroupRank(q->GetDevice().Rank().GlobalRank());
    const int64_t q_heads = q->Dims()[1];
    const int64_t kv_heads = k->Dims()[1];
    CHECK_EQ(q_heads % cp_size, 0) << "A2A CP requires local query heads divisible by CP size";
    CHECK_EQ(kv_heads % cp_size, 0) << "A2A CP requires local KV heads divisible by CP size";

    // TODO(zbl): Replace this semantic all-gather/slice implementation with true QKV/O all-to-all
    //            once ProcessGroup exposes an async all-to-all primitive.
    auto full_q = GatherFromCPRegionFunc(q);
    auto full_k = GatherFromCPRegionFunc(k);
    auto full_v = GatherFromCPRegionFunc(v);
    auto full_mask = mask ? GatherFromCPRegionFunc(mask) : nullptr;

    const int64_t q_heads_per_cp = q_heads / cp_size;
    const int64_t kv_heads_per_cp = kv_heads / cp_size;
    auto q_shard = full_q->Slice(1, rank * q_heads_per_cp, (rank + 1) * q_heads_per_cp);
    auto k_shard = full_k->Slice(1, rank * kv_heads_per_cp, (rank + 1) * kv_heads_per_cp);
    auto v_shard = full_v->Slice(1, rank * kv_heads_per_cp, (rank + 1) * kv_heads_per_cp);
    auto k_for_attn = RepeatKVHeads(k_shard, n_rep);
    auto v_for_attn = RepeatKVHeads(v_shard, n_rep);
    auto output_shard = ApplyAttention(q_shard, k_for_attn, v_for_attn, full_mask, mask_true_means_invalid, scale);

    auto gathered_heads = GatherFromCPHeadRegionFunc(output_shard);
    const int64_t local_t = q->Dims()[2];
    const int64_t seq_start = static_cast<int64_t>(rank) * local_t;
    return gathered_heads->Slice(2, seq_start, seq_start + local_t)->Contiguous();
}

std::shared_ptr<Tensor> ContextParallelAttentionFunc(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                                     const std::shared_ptr<Tensor> &v,
                                                     const std::shared_ptr<Tensor> &mask, bool mask_true_means_invalid,
                                                     float scale, int64_t n_rep) {
    CHECK_GT(global::GetContextParallelSize(), 1);
    const auto comm_type = global::GetContextParallelCommType();
    if (comm_type == "p2p") {
        return AttnFuncWithCPAndKVP2P(q, k, v, mask, mask_true_means_invalid, scale, n_rep);
    }
    if (comm_type == "a2a") {
        return AttnFuncWithCPAndQKVOA2A(q, k, v, mask, mask_true_means_invalid, scale, n_rep);
    }
    return AttnFuncWithCPAndKVAllGather(q, k, v, mask, mask_true_means_invalid, scale, n_rep);
}

} // namespace infini_train::nn::parallel

#include "infini_train/include/nn/parallel/context_parallel.h"

#include <cmath>
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
    return ProcessGroupFactory::Instance(tensor->GetDevice().type())
        ->Get(GetContextParallelProcessGroupName(tensor->GetDevice().Rank().GlobalRank()));
}

// Comm Kernel Call Functions
std::shared_ptr<Tensor> GatherAlongFirstDim(const std::shared_ptr<Tensor> &tensor) {
    const int cp_size = global::GetContextParallelSize();
    CHECK_GT(cp_size, 0) << "Context Parallel group not initialized";
    if (cp_size == 1) {
        return tensor;
    }

    auto cp_group = GetCPGroup(tensor);
    auto output_shape = tensor->Dims();
    output_shape[0] *= cp_size;
    auto output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), tensor->GetDevice());
    cp_group->AllGather(output, tensor, false);
    return output;
}

std::shared_ptr<Tensor> ReduceScatterAlongFirstDim(const std::shared_ptr<Tensor> &tensor) {
    const int cp_size = global::GetContextParallelSize();
    CHECK_GT(cp_size, 0) << "Context Parallel group not initialized";
    if (cp_size == 1) {
        return tensor;
    }

    auto cp_group = GetCPGroup(tensor);
    auto output_shape = tensor->Dims();
    CHECK_EQ(output_shape[0] % cp_size, 0) << "First dimension must be divisible by CP size";
    output_shape[0] /= cp_size;
    auto output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), tensor->GetDevice());
    cp_group->ReduceScatter(output, tensor, function::ReduceOpType::kSum, false);
    return output;
}

std::shared_ptr<Tensor> AllToAllAlongFirstDim(const std::shared_ptr<Tensor> &tensor) {
    // Tensor P is split along first dim in [P0 | P1 | ... | Pn]
    // Each rank j sends Pj to every other rank and receive the rest of P from every other rank
    const int cp_size = global::GetContextParallelSize();
    CHECK_GT(cp_size, 0) << "Context Parallel group not initialized";
    if (cp_size == 1) {
        return tensor;
    }

    auto cp_group = GetCPGroup(tensor);
    auto output_shape = tensor->Dims();
    CHECK_EQ(output_shape[0] % cp_size, 0) << "First dimension must be divisible by CP size";
    auto output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), tensor->GetDevice());
    cp_group->AllToAll(output, tensor, false);
    return output;
}

std::shared_ptr<Tensor> AllToAllSeqToHead(const std::shared_ptr<Tensor> &input) {
    if (global::GetContextParallelSize() == 1) {
        return input;
    }

    const int cp_size = global::GetContextParallelSize();
    const auto &shape = input->Dims();
    CHECK_EQ(shape.size(), 4);
    const int64_t B = shape[0], H = shape[1], T_l = shape[2], D = shape[3];
    CHECK_EQ(H % cp_size, 0) << "A2A CP requires head dimension divisible by CP size";
    const int64_t H_per_cp = H / cp_size;

    // input: (B, H, T_l, D)
    //
    // send_input: (H, B, T_l, D), split dim 0 into CP chunks of H_per_cp heads.
    auto send_input = input->Transpose(0, 1)->Contiguous();
    // exchanged: (H, B, T_l, D), dim 0 chunks are ordered by source sequence-owner rank.
    auto exchanged = AllToAllAlongFirstDim(send_input);
    // output: (B, H_per_cp, T_g, D)
    return exchanged->View({cp_size, H_per_cp, B, T_l, D})
        ->Transpose(0, 2)
        ->Contiguous()
        ->View({B, H_per_cp, static_cast<int64_t>(cp_size) * T_l, D});
}

std::shared_ptr<Tensor> AllToAllHeadToSeq(const std::shared_ptr<Tensor> &input) {
    if (global::GetContextParallelSize() == 1) {
        return input;
    }

    const int cp_size = global::GetContextParallelSize();
    const auto &shape = input->Dims();
    CHECK_EQ(shape.size(), 4);
    const int64_t B = shape[0], H_per_cp = shape[1], T_g = shape[2], D = shape[3];
    CHECK_EQ(T_g % cp_size, 0) << "A2A CP requires sequence dimension divisible by CP size";
    const int64_t T_l = T_g / cp_size;

    // input: (B, H_per_cp, T_g, D)
    //
    // send_input: (CP * H_per_cp, B, T_l, D), split dim 0 into CP sequence-owner chunks.
    auto send_input = input->View({B, H_per_cp, cp_size, T_l, D})
                          ->Transpose(0, 2)
                          ->Contiguous()
                          ->View({static_cast<int64_t>(cp_size) * H_per_cp, B, T_l, D});
    // exchanged: (CP * H_per_cp, B, T_l, D), dim 0 chunks are ordered by source head-owner rank.
    auto exchanged = AllToAllAlongFirstDim(send_input);
    // output: (B, CP * H_per_cp, T_l, D)
    return exchanged->View({cp_size, H_per_cp, B, T_l, D})
        ->Transpose(1, 2)
        ->Transpose(0, 1)
        ->Contiguous()
        ->View({B, static_cast<int64_t>(cp_size) * H_per_cp, T_l, D});
}

// Attention Helper Functions
std::shared_ptr<Tensor> NewZeroTensorLike(const std::shared_ptr<Tensor> &tensor) {
    auto output = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), tensor->GetDevice());
    output->Fill(0.0f);
    return output;
}

std::vector<std::shared_ptr<Work>> P2PCommunicate(int rank, const std::vector<std::shared_ptr<Tensor>> &send_tensors,
                                                  int send_dst,
                                                  const std::vector<std::shared_ptr<Tensor>> &recv_tensors,
                                                  int recv_src, const ProcessGroup *cp_group, bool batch_p2p_comm) {
    std::vector<P2POp> ops;
    ops.reserve(send_tensors.size() + recv_tensors.size());
    std::vector<std::shared_ptr<Work>> works;

    if (rank % 2 == 0) {
        if (batch_p2p_comm) {
            for (const auto &tensor : send_tensors) { ops.push_back({P2POpType::kSend, tensor, send_dst}); }
            for (const auto &tensor : recv_tensors) { ops.push_back({P2POpType::kRecv, tensor, recv_src}); }
        } else {
            works.push_back(cp_group->Send(send_tensors, send_dst, true));
            works.push_back(cp_group->Recv(recv_tensors, recv_src, true));
        }
    } else {
        if (batch_p2p_comm) {
            for (const auto &tensor : recv_tensors) { ops.push_back({P2POpType::kRecv, tensor, recv_src}); }
            for (const auto &tensor : send_tensors) { ops.push_back({P2POpType::kSend, tensor, send_dst}); }
        } else {
            works.push_back(cp_group->Recv(recv_tensors, recv_src, true));
            works.push_back(cp_group->Send(send_tensors, send_dst, true));
        }
    }
    if (batch_p2p_comm) {
        works.push_back(cp_group->BatchSendRecv(ops, true));
    }
    return works;
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

std::shared_ptr<Tensor> ApplyCoreAttention(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                           const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask) {
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(q->Dims().back())));
    // scores: (B, H, T_q, T_k)
    auto scores = q->Matmul(k->Transpose(-2, -1)) * scale;
    if (mask) {
        scores = scores->MaskedFill(mask, std::numeric_limits<float>::lowest());
    }
    // probs: (B, H, T_q, T_k)
    auto probs = nn::function::Softmax(scores, -1);
    // output: (B, H, T_q, D)
    return probs->Matmul(v);
}

// Autograd Function Definitions
class GatherFromCPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFromCPRegionFunction";

    explicit GatherFromCPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        auto input = input_tensors[0];
        // FIXME(zbl): Megatron keeps sequence as dim 0. We uses [B, H, S, D], so move only
        //              the sequence dimension to dim 0 before the CP gather.
        return {GatherAlongFirstDim(input->Transpose(0, 2))->Transpose(0, 2)->Contiguous()};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        // FIXME(zbl): Megatron keeps sequence as dim 0. We uses [B, H, S, D], so move only
        //              the sequence dimension to dim 0 before the CP gather.
        return {ReduceScatterAlongFirstDim(grad_outputs[0]->Transpose(0, 2))->Transpose(0, 2)->Contiguous()};
    }
};

class AllToAllSeqToHeadCPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "AllToAllSeqToHeadCPRegionFunction";

    explicit AllToAllSeqToHeadCPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        return {AllToAllSeqToHead(input_tensors[0])};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {AllToAllHeadToSeq(grad_outputs[0])};
    }
};

class AllToAllHeadToSeqCPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "AllToAllHeadToSeqCPRegionFunction";

    explicit AllToAllHeadToSeqCPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        return {AllToAllHeadToSeq(input_tensors[0])};
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {AllToAllSeqToHead(grad_outputs[0])};
    }
};

class AttnWithCPAndKVP2P : public autograd::Function {
public:
    static constexpr char kType[] = "AttnWithCPAndKVP2PFunction";

    AttnWithCPAndKVP2P() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        CHECK_EQ(input_tensors.size(), 4);
        // Shape notation:
        // B: batch size, H_q: local query heads after TP, H_kv: local KV heads before GQA repeat,
        // T_l: CP-local sequence length, T_g: global sequence length, D: head dimension.
        // q: (B, H_q, T_l, D)
        const auto &q = input_tensors[0];
        // k_local: (B, H_kv, T_l, D)
        const auto &k_local = input_tensors[1];
        // v_local: (B, H_kv, T_l, D)
        const auto &v_local = input_tensors[2];
        // mask: (1, 1, T_l, T_g), true values are invalid attention locations.
        const auto &mask = input_tensors[3];
        const int cp_size = global::GetContextParallelSize();
        CHECK_GT(cp_size, 1);
        CHECK(mask) << "CP ring attention expects a causal mask.";

        auto cp_group = GetCPGroup(q);
        CHECK_NOTNULL(cp_group);
        const int rank = cp_group->GetGroupRank(q->GetDevice().Rank().GlobalRank());
        const int send_to = (rank + 1) % cp_size;
        const int recv_from = (rank - 1 + cp_size) % cp_size;
        // NOTE(zbl): Megatron-LM enables batched P2P by default for CP=2 on pre-Blackwell GPUs.
        const bool batch_p2p_comm = (cp_size == 2);

        const int64_t local_t = q->Dims()[2];
        CHECK_EQ(k_local->Dims()[2], local_t);
        CHECK_EQ(v_local->Dims()[2], local_t);
        CHECK_EQ(k_local->Dims()[1], v_local->Dims()[1]);
        CHECK_EQ(q->Dims()[1] % k_local->Dims()[1], 0);
        const int64_t n_rep = q->Dims()[1] / k_local->Dims()[1];
        const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(q->Dims().back())));

        // current_k: (B, H_kv, T_l, D), owned by rank `(rank - step + cp_size) % cp_size`.
        auto current_k = k_local;
        // current_v: (B, H_kv, T_l, D), owned by rank `(rank - step + cp_size) % cp_size`.
        auto current_v = v_local;
        // running_max: (B, H_q, T_l, 1)
        std::shared_ptr<Tensor> running_max;
        // running_sum: (B, H_q, T_l, 1)
        std::shared_ptr<Tensor> running_sum;
        // running_out: (B, H_q, T_l, D)
        std::shared_ptr<Tensor> running_out;

        for (int step = 0; step < cp_size; ++step) {
            std::shared_ptr<Tensor> next_k;
            std::shared_ptr<Tensor> next_v;
            std::vector<std::shared_ptr<Work>> p2p_works;
            if (step + 1 < cp_size) {
                // next_k: (B, H_kv, T_l, D)
                next_k = std::make_shared<Tensor>(k_local->Dims(), k_local->Dtype(), k_local->GetDevice());
                // next_v: (B, H_kv, T_l, D)
                next_v = std::make_shared<Tensor>(v_local->Dims(), v_local->Dtype(), v_local->GetDevice());
                p2p_works = P2PCommunicate(rank, {current_k, current_v}, send_to, {next_k, next_v}, recv_from, cp_group,
                                           batch_p2p_comm);
            }

            const int owner = (rank - step + cp_size) % cp_size;
            const int64_t kv_start = static_cast<int64_t>(owner) * local_t;
            const int64_t kv_end = kv_start + local_t;

            // k_for_attn: (B, H_q, T_l, D)
            auto k_for_attn = RepeatKVHeads(current_k, n_rep);
            // v_for_attn: (B, H_q, T_l, D)
            auto v_for_attn = RepeatKVHeads(current_v, n_rep);
            // scores: (B, H_q, T_l, T_l)
            auto scores = q->Matmul(k_for_attn->Transpose(-2, -1)) * scale;
            // invalid_mask: (1, 1, T_l, T_l)
            auto invalid_mask = mask->Slice(3, kv_start, kv_end);
            scores = scores->MaskedFill(invalid_mask, std::numeric_limits<float>::lowest());

            // chunk_max: (B, H_q, T_l, 1)
            auto chunk_max = scores->Max(-1, true);
            // probs: (B, H_q, T_l, T_l)
            auto probs = (scores - chunk_max)->Exp()->MaskedFill(invalid_mask, 0.0f);
            // chunk_sum: (B, H_q, T_l, 1)
            auto chunk_sum = probs->Sum(-1, true);
            // chunk_out: (B, H_q, T_l, D)
            auto chunk_out = probs->Matmul(v_for_attn);

            if (!running_out) {
                running_max = chunk_max;
                running_sum = chunk_sum;
                running_out = chunk_out;
            } else {
                // new_max: (B, H_q, T_l, 1)
                auto new_max
                    = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{running_max, chunk_max}, -1)->Max(-1);
                // old_scale: (B, H_q, T_l, 1)
                auto old_scale = (running_max - new_max)->Exp();
                // new_scale: (B, H_q, T_l, 1)
                auto new_scale = (chunk_max - new_max)->Exp();
                running_sum = running_sum * old_scale + chunk_sum * new_scale;
                running_out = running_out * old_scale + chunk_out * new_scale;
                running_max = new_max;
            }

            if (!p2p_works.empty()) {
                for (const auto &work : p2p_works) { work->WaitNonBlocking(); }
                current_k = next_k;
                current_v = next_v;
            }
        }

        // output: (B, H_q, T_l, D)
        // running_max: (B, H_q, T_l, 1)
        // running_sum: (B, H_q, T_l, 1)
        return {running_out / running_sum, running_max, running_sum};
    }

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override {
        CHECK_EQ(output_tensors.size(), 3);
        const auto &output = output_tensors[0];
        const auto &softmax_max = output_tensors[1];
        const auto &softmax_sum = output_tensors[2];
        ctx_.MarkNonDifferentiable({softmax_max, softmax_sum});
        ctx_.SaveForBackward(
            {input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], output, softmax_max, softmax_sum});
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        // Shape notation:
        // - B: batch size
        // - H_q: local query heads after TP
        // - H_kv: local KV heads before GQA repeat
        // - T_l: CP-local sequence length
        // - T_g: global sequence length
        // - D: head dimension.

        CHECK_GE(grad_outputs.size(), 1);
        auto saved_tensors = ctx_.GetSavedTensors();
        CHECK_EQ(saved_tensors.size(), 7);

        // q: (B, H_q, T_l, D)
        const auto &q = saved_tensors[0];
        // k_local: (B, H_kv, T_l, D)
        const auto &k_local = saved_tensors[1];
        // v_local: (B, H_kv, T_l, D)
        const auto &v_local = saved_tensors[2];
        // mask: (1, 1, T_l, T_g), true values are invalid attention locations.
        const auto &mask = saved_tensors[3];
        // output: (B, H_q, T_l, D)
        const auto &output = saved_tensors[4];
        // softmax_max: (B, H_q, T_l, 1)
        const auto &softmax_max = saved_tensors[5];
        // softmax_sum: (B, H_q, T_l, 1)
        const auto &softmax_sum = saved_tensors[6];
        // grad_output: (B, H_q, T_l, D)
        const auto &grad_output = grad_outputs[0];

        const int cp_size = global::GetContextParallelSize();
        CHECK_GT(cp_size, 1);

        auto cp_group = GetCPGroup(q);
        CHECK_NOTNULL(cp_group);
        const int rank = cp_group->GetGroupRank(q->GetDevice().Rank().GlobalRank());
        const int send_to = (rank + 1) % cp_size;
        const int recv_from = (rank - 1 + cp_size) % cp_size;
        // NOTE(zbl): Megatron-LM enables batched P2P by default for CP=2 on pre-Blackwell GPUs.
        const bool batch_p2p_comm = (cp_size == 2);

        const int64_t local_t = q->Dims()[2];
        CHECK_EQ(k_local->Dims()[2], local_t);
        CHECK_EQ(v_local->Dims()[2], local_t);
        CHECK_EQ(k_local->Dims()[1], v_local->Dims()[1]);
        CHECK_EQ(q->Dims()[1] % k_local->Dims()[1], 0);
        const int64_t n_rep = q->Dims()[1] / k_local->Dims()[1];
        const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(q->Dims().back())));

        // current_k: (B, H_kv, T_l, D)
        auto current_k = k_local;
        // current_v: (B, H_kv, T_l, D)
        auto current_v = v_local;
        // current_grad_k: (B, H_kv, T_l, D)
        auto current_grad_k = NewZeroTensorLike(k_local);
        // current_grad_v: (B, H_kv, T_l, D)
        auto current_grad_v = NewZeroTensorLike(v_local);
        // grad_q: (B, H_q, T_l, D)
        std::shared_ptr<Tensor> grad_q;
        // softmax_delta: (B, H_q, T_l, 1)
        const auto softmax_delta = (grad_output * output)->Sum(-1, true);

        for (int step = 0; step < cp_size; ++step) {
            const int owner = (rank - step + cp_size) % cp_size;
            const int64_t kv_start = static_cast<int64_t>(owner) * local_t;
            const int64_t kv_end = kv_start + local_t;

            // k_for_attn: (B, H_q, T_l, D)
            auto k_for_attn = RepeatKVHeads(current_k, n_rep);
            // v_for_attn: (B, H_q, T_l, D)
            auto v_for_attn = RepeatKVHeads(current_v, n_rep);
            // scores: (B, H_q, T_l, T_l)
            auto scores = q->Matmul(k_for_attn->Transpose(-2, -1)) * scale;
            // invalid_mask: (1, 1, T_l, T_l)
            auto invalid_mask = mask->Slice(3, kv_start, kv_end);
            scores = scores->MaskedFill(invalid_mask, std::numeric_limits<float>::lowest());

            // probs: (B, H_q, T_l, T_l)
            auto probs = (scores - softmax_max)->Exp()->MaskedFill(invalid_mask, 0.0f) / softmax_sum;
            // grad_v_repeated: (B, H_q, T_l, D)
            auto grad_v_repeated = probs->Transpose(-2, -1)->Matmul(grad_output);
            // grad_probs: (B, H_q, T_l, T_l)
            auto grad_probs = grad_output->Matmul(v_for_attn->Transpose(-2, -1));
            // grad_scores: (B, H_q, T_l, T_l)
            auto grad_scores = probs * (grad_probs - softmax_delta);
            // grad_k_repeated: (B, H_q, T_l, D)
            auto grad_k_repeated = grad_scores->Transpose(-2, -1)->Matmul(q) * scale;

            // SumRepeatedKVHeads maps repeated GQA gradients from (B, H_q, T_l, D) to (B, H_kv, T_l, D).
            current_grad_k = current_grad_k + SumRepeatedKVHeads(grad_k_repeated, n_rep);
            current_grad_v = current_grad_v + SumRepeatedKVHeads(grad_v_repeated, n_rep);

            std::vector<std::shared_ptr<Work>> p2p_works;
            std::shared_ptr<Tensor> next_k;
            std::shared_ptr<Tensor> next_v;
            std::shared_ptr<Tensor> next_grad_k;
            std::shared_ptr<Tensor> next_grad_v;

            if (step + 1 < cp_size) {
                // Send current K/V plus accumulated K/V grads to the next rank; receive the previous owner chunk.
                // next_k: (B, H_kv, T_l, D)
                next_k = std::make_shared<Tensor>(k_local->Dims(), k_local->Dtype(), k_local->GetDevice());
                // next_v: (B, H_kv, T_l, D)
                next_v = std::make_shared<Tensor>(v_local->Dims(), v_local->Dtype(), v_local->GetDevice());
                // next_grad_k: (B, H_kv, T_l, D)
                next_grad_k = NewZeroTensorLike(k_local);
                // next_grad_v: (B, H_kv, T_l, D)
                next_grad_v = NewZeroTensorLike(v_local);
                p2p_works
                    = P2PCommunicate(rank, {current_k, current_v, current_grad_k, current_grad_v}, send_to,
                                     {next_k, next_v, next_grad_k, next_grad_v}, recv_from, cp_group, batch_p2p_comm);
            } else {
                // Last step only needs to rotate accumulated K/V grads back to the local owner rank.
                // next_grad_k: (B, H_kv, T_l, D)
                next_grad_k = NewZeroTensorLike(k_local);
                // next_grad_v: (B, H_kv, T_l, D)
                next_grad_v = NewZeroTensorLike(v_local);
                p2p_works = P2PCommunicate(rank, {current_grad_k, current_grad_v}, send_to, {next_grad_k, next_grad_v},
                                           recv_from, cp_group, batch_p2p_comm);
            }

            // grad_q_chunk: (B, H_q, T_l, D)
            auto grad_q_chunk = grad_scores->Matmul(k_for_attn) * scale;
            grad_q = grad_q ? grad_q + grad_q_chunk : grad_q_chunk;

            for (const auto &work : p2p_works) { work->WaitNonBlocking(); }

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
};

std::shared_ptr<Tensor> AllToAllSeqToHeadCPRegionFunc(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<AllToAllSeqToHeadCPRegion>()->Apply({input})[0];
}

std::shared_ptr<Tensor> AllToAllHeadToSeqCPRegionFunc(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<AllToAllHeadToSeqCPRegion>()->Apply({input})[0];
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
    return std::make_shared<GatherFromCPRegion>()->Apply({input})[0];
}

// CP Attention Backend Functions
std::shared_ptr<Tensor> AttnFuncWithCPAndKVP2P(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                               const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask) {
    return std::make_shared<AttnWithCPAndKVP2P>()->Apply({q, k, v, mask})[0];
}

std::shared_ptr<Tensor> AttnFuncWithCPAndKVAllGather(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                                     const std::shared_ptr<Tensor> &v,
                                                     const std::shared_ptr<Tensor> &mask) {
    CHECK_EQ(k->Dims()[1], v->Dims()[1]);
    CHECK_EQ(q->Dims()[1] % k->Dims()[1], 0);
    const int64_t n_rep = q->Dims()[1] / k->Dims()[1];
    // gathered_k: (B, H_kv, T_g, D)
    auto gathered_k = GatherFromCPRegionFunc(k);
    // gathered_v: (B, H_kv, T_g, D)
    auto gathered_v = GatherFromCPRegionFunc(v);
    // k_for_attn: (B, H_q, T_g, D)
    auto k_for_attn = RepeatKVHeads(gathered_k, n_rep);
    // v_for_attn: (B, H_q, T_g, D)
    auto v_for_attn = RepeatKVHeads(gathered_v, n_rep);
    return ApplyCoreAttention(q, k_for_attn, v_for_attn, mask);
}

std::shared_ptr<Tensor> AttnFuncWithCPAndQKVOA2A(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                                 const std::shared_ptr<Tensor> &v,
                                                 const std::shared_ptr<Tensor> &mask) {
    const int cp_size = global::GetContextParallelSize();
    const int64_t q_heads = q->Dims()[1];
    const int64_t kv_heads = k->Dims()[1];
    CHECK_EQ(kv_heads, v->Dims()[1]);
    CHECK_EQ(q_heads % cp_size, 0) << "A2A CP requires local query heads divisible by CP size";
    CHECK_EQ(kv_heads % cp_size, 0) << "A2A CP requires local KV heads divisible by CP size";
    CHECK_EQ(q_heads % kv_heads, 0);

    // q_shard: (B, H_q/CP, T_g, D)
    auto q_shard = AllToAllSeqToHeadCPRegionFunc(q);
    // k_shard: (B, H_kv/CP, T_g, D)
    auto k_shard = AllToAllSeqToHeadCPRegionFunc(k);
    // v_shard: (B, H_kv/CP, T_g, D)
    auto v_shard = AllToAllSeqToHeadCPRegionFunc(v);
    // full_mask: (1, 1, T_g, T_g)
    auto full_mask = mask ? GatherFromCPRegionFunc(mask) : nullptr;

    const int64_t n_rep = q_shard->Dims()[1] / k_shard->Dims()[1];
    // k_for_attn: (B, H_q/CP, T_g, D)
    auto k_for_attn = RepeatKVHeads(k_shard, n_rep);
    // v_for_attn: (B, H_q/CP, T_g, D)
    auto v_for_attn = RepeatKVHeads(v_shard, n_rep);
    // output_shard: (B, H_q/CP, T_g, D)
    auto output_shard = ApplyCoreAttention(q_shard, k_for_attn, v_for_attn, full_mask);

    // output: (B, H_q, T_l, D)
    return AllToAllHeadToSeqCPRegionFunc(output_shard);
}

std::shared_ptr<Tensor> AttnForwardFuncWithCP(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                              const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask) {
    CHECK_GT(global::GetContextParallelSize(), 1);
    const auto comm_type = global::GetContextParallelCommType();
    if (comm_type == "p2p") {
        return AttnFuncWithCPAndKVP2P(q, k, v, mask);
    } else if (comm_type == "a2a") {
        return AttnFuncWithCPAndQKVOA2A(q, k, v, mask);
    } else if (comm_type == "all_gather") {
        return AttnFuncWithCPAndKVAllGather(q, k, v, mask);
    } else {
        LOG(FATAL) << "AttnForwardFuncWithCP: Unsupported communication type " << comm_type << ".";
    }
}

} // namespace infini_train::nn::parallel

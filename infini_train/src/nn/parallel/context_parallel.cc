#include "infini_train/include/nn/parallel/context_parallel.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/function.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
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

core::Stream *GetCPComputeStream(Device device) {
    thread_local std::unordered_map<int, std::unique_ptr<core::Stream>> cp_compute_streams;
    auto [it, inserted] = cp_compute_streams.emplace(device.index(), nullptr);
    if (inserted) {
        it->second.reset(core::GetDeviceGuardImpl(device.type())->CreateStream(device));
    }
    return it->second.get();
}

class RuntimeStreamGuard {
public:
    RuntimeStreamGuard(Device device, core::Stream *stream)
        : device_(device), impl_(core::GetDeviceGuardImpl(device.type())),
          previous_stream_(impl_->ExchangeStream(device, stream)) {}

    ~RuntimeStreamGuard() { impl_->ExchangeStream(device_, previous_stream_); }

private:
    Device device_;
    core::DeviceGuardImpl *impl_;
    core::Stream *previous_stream_;
};

std::vector<std::shared_ptr<Work>> P2PCommunicate(int rank, const std::vector<std::shared_ptr<Tensor>> &send_tensors,
                                                  int send_dst,
                                                  const std::vector<std::shared_ptr<Tensor>> &recv_tensors,
                                                  int recv_src, const ProcessGroup *cp_group, bool batch_p2p_comm) {
    // NOTE(zbl): Sanity checks for Send/Recv calls, in case of communication hanging.
    CHECK_EQ(send_tensors.size(), recv_tensors.size());
    for (size_t i = 0; i < send_tensors.size(); ++i) {
        CHECK_NOTNULL(send_tensors[i]);
        CHECK_NOTNULL(recv_tensors[i]);
        CHECK_EQ(send_tensors[i]->NumElements(), recv_tensors[i]->NumElements())
            << "P2P send/recv tensor numel mismatch at slot " << i;
        const auto send_dtype = send_tensors[i]->Dtype();
        const auto recv_dtype = recv_tensors[i]->Dtype();
        CHECK(send_dtype == recv_dtype) << "P2P send/recv tensor dtype mismatch at slot " << i
                                        << ", send=" << kDataTypeToDesc.at(send_dtype)
                                        << ", recv=" << kDataTypeToDesc.at(recv_dtype);
    }

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
        // - B: batch size
        // - H_q: local query heads after TP
        // - H_kv: local KV heads before GQA repeat
        // - T_l: CP-local sequence length
        // - T_g: global sequence length
        // - D: head dimension.

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

        // current_k: (B, H_kv, T_l, D), owned by rank `(rank - step + cp_size) % cp_size`, init with step=0.
        // current_v: (B, H_kv, T_l, D), owned by rank `(rank - step + cp_size) % cp_size`, init with step=0.
        // K/V can be transposed views after attention input layout conversion. Make sure K/V are contiguous for P2P.
        auto current_k = k_local->Contiguous();
        auto current_v = v_local->Contiguous();
        auto *runtime_impl = core::GetDeviceGuardImpl(q->GetDevice().type());
        auto *main_compute_stream = runtime_impl->GetStream(q->GetDevice());
        auto *cp_compute_stream = GetCPComputeStream(q->GetDevice());
        core::Event *inputs_ready_event = nullptr;
        runtime_impl->EventCreateWithFlags(&inputs_ready_event, core::EventFlag::kDisableTiming);
        runtime_impl->EventRecord(inputs_ready_event, main_compute_stream);
        runtime_impl->StreamWaitEvent(cp_compute_stream, inputs_ready_event, 0);

        std::vector<std::shared_ptr<Tensor>> chunk_maxs;
        std::vector<std::shared_ptr<Tensor>> chunk_sums;
        std::vector<std::shared_ptr<Tensor>> chunk_outs;
        chunk_maxs.reserve(cp_size);
        chunk_sums.reserve(cp_size);
        chunk_outs.reserve(cp_size);

        std::shared_ptr<Tensor> next_k;
        std::shared_ptr<Tensor> next_v;
        std::vector<std::shared_ptr<Work>> p2p_works;

        // Perform `cp_size` rounds to circulate K/V chunks across all CP ranks.
        for (int step = 0; step < cp_size; ++step) {
            auto *compute_stream = step % 2 == 0 ? main_compute_stream : cp_compute_stream;
            RuntimeStreamGuard stream_guard(q->GetDevice(), compute_stream);

            if (!p2p_works.empty()) {
                for (const auto &work : p2p_works) { work->WaitNonBlocking(); }
                p2p_works.clear();
                current_k = next_k;
                current_v = next_v;
            }

            // Only performs `cp_size - 1` times of ring P2P, no comm is needed for the final round
            if (step + 1 < cp_size) {
                // next_k: (B, H_kv, T_l, D)
                next_k = std::make_shared<Tensor>(k_local->Dims(), k_local->Dtype(), k_local->GetDevice());
                // next_v: (B, H_kv, T_l, D)
                next_v = std::make_shared<Tensor>(v_local->Dims(), v_local->Dtype(), v_local->GetDevice());
                p2p_works = P2PCommunicate(rank, {current_k, current_v}, send_to, {next_k, next_v}, recv_from, cp_group,
                                           batch_p2p_comm);
            }

            // Rank of owner of current K/V chunk in this step
            const int owner = (rank - step + cp_size) % cp_size;
            // Token range of current K/V chunk
            const int64_t kv_start = static_cast<int64_t>(owner) * local_t;
            const int64_t kv_end = kv_start + local_t;
            // NOTE(zbl): CP shards the sequence dimension in rank order: higher ranks own later token chunks.
            //            Under causal attention, queries in this rank cannot attend to K/V from later chunks,
            //            so a chunk is fully masked when its owner rank is greater than the query rank.
            const bool chunk_fully_masked = owner > rank;

            std::shared_ptr<Tensor> chunk_max;
            std::shared_ptr<Tensor> chunk_sum;
            std::shared_ptr<Tensor> chunk_out;

            if (chunk_fully_masked) {
                auto stats_shape = q->Dims();
                stats_shape.back() = 1;
                // chunk_max: (B, H_q, T_l, 1)
                chunk_max = std::make_shared<Tensor>(stats_shape, q->Dtype(), q->GetDevice());
                chunk_max->Fill(std::numeric_limits<float>::lowest());
                // chunk_sum: (B, H_q, T_l, 1)
                chunk_sum = std::make_shared<Tensor>(stats_shape, q->Dtype(), q->GetDevice());
                chunk_sum->Fill(0.0f);
                // chunk_out: (B, H_q, T_l, D)
                chunk_out = NewZeroTensorLike(q);
            } else {
                // k_for_attn: (B, H_q, T_l, D)
                auto k_for_attn = RepeatKVHeads(current_k, n_rep);
                // v_for_attn: (B, H_q, T_l, D)
                auto v_for_attn = RepeatKVHeads(current_v, n_rep);
                // scores: (B, H_q, T_l, T_l)
                auto scores = q->Matmul(k_for_attn->Transpose(-2, -1)) * scale;
                // invalid_mask: (1, 1, T_l, T_l), assume true values are invalid attention locations
                auto invalid_mask = mask->Slice(-1, kv_start, kv_end);
                scores = scores->MaskedFill(invalid_mask, std::numeric_limits<float>::lowest());

                // chunk_max: (B, H_q, T_l, 1)
                chunk_max = scores->Max(-1, true);
                // probs: (B, H_q, T_l, T_l)
                auto probs = (scores - chunk_max)->Exp();
                // chunk_sum: (B, H_q, T_l, 1)
                chunk_sum = probs->Sum(-1, true);
                // chunk_out: (B, H_q, T_l, D)
                chunk_out = probs->Matmul(v_for_attn);
            }

            chunk_maxs.push_back(chunk_max);
            chunk_sums.push_back(chunk_sum);
            chunk_outs.push_back(chunk_out);
        }

        core::Event *cp_compute_done_event = nullptr;
        runtime_impl->EventCreateWithFlags(&cp_compute_done_event, core::EventFlag::kDisableTiming);
        runtime_impl->EventRecord(cp_compute_done_event, cp_compute_stream);
        runtime_impl->StreamWaitEvent(main_compute_stream, cp_compute_done_event, 0);
        runtime_impl->EventDestroy(inputs_ready_event);
        runtime_impl->EventDestroy(cp_compute_done_event);

        // Online Softmax variables
        // running_max: (B, H_q, T_l, 1)
        auto running_max = chunk_maxs[0];
        // running_sum: (B, H_q, T_l, 1)
        auto running_sum = chunk_sums[0];
        // running_out: (B, H_q, T_l, D)
        auto running_out = chunk_outs[0];
        for (int step = 1; step < cp_size; ++step) {
            // Update Online Softmax variables
            // new_max: (B, H_q, T_l, 1)
            auto new_max
                = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{running_max, chunk_maxs[step]}, -1)->Max(-1);
            // old_scale: (B, H_q, T_l, 1)
            auto old_scale = (running_max - new_max)->Exp();
            // new_scale: (B, H_q, T_l, 1)
            auto new_scale = (chunk_maxs[step] - new_max)->Exp();
            running_sum = running_sum * old_scale + chunk_sums[step] * new_scale;
            running_out = running_out * old_scale + chunk_outs[step] * new_scale;
            running_max = new_max;
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
        // mask: (1, 1, T_l, T_g), true values are invalid attention locations
        const auto &mask = saved_tensors[3];
        // output: (B, H_q, T_l, D)
        const auto &output = saved_tensors[4];
        // softmax_max: (B, H_q, T_l, 1)
        const auto &softmax_max = saved_tensors[5];
        // softmax_sum: (B, H_q, T_l, 1)
        const auto &softmax_sum = saved_tensors[6];
        // grad_output: (B, H_q, T_l, D)
        const auto &grad_output = grad_outputs[0];

        // NOTE(zbl): Backward recomputes per-chunk scores/probs instead of saving the full attn matrix.
        //            Saving probs would require O(B * H_q * T_l * T_g) activation memory and is against
        //            the purpose of context parallel ring attention. Therefore, we need to restore the
        //            autocast context used in forward.
        auto forward_autocast_context = ctx_.GetForwardAutocastContext();
        std::unique_ptr<AutocastGuard> autocast_guard;
        if (forward_autocast_context.enabled) {
            autocast_guard = std::make_unique<AutocastGuard>(forward_autocast_context);
        }

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

        // Keep the same contiguous P2P contract as forward; saved K/V may be transposed views.
        // current_k: (B, H_kv, T_l, D)
        // current_v: (B, H_kv, T_l, D)
        auto current_k = k_local->Contiguous();
        auto current_v = v_local->Contiguous();
        // current_grad_k: (B, H_kv, T_l, D)
        // current_grad_v: (B, H_kv, T_l, D)
        auto current_grad_k = NewZeroTensorLike(k_local);
        auto current_grad_v = NewZeroTensorLike(v_local);

        // grad_q: (B, H_q, T_l, D)
        auto grad_q = NewZeroTensorLike(q);
        // softmax_delta: (B, H_q, T_l, 1)
        const auto softmax_delta = (grad_output * output)->Sum(-1, true);
        const auto device = grad_output->GetDevice().type();
        auto *runtime_impl = core::GetDeviceGuardImpl(device);
        auto *main_compute_stream = runtime_impl->GetStream(q->GetDevice());
        auto *cp_compute_stream = GetCPComputeStream(q->GetDevice());
        core::Event *inputs_ready_event = nullptr;
        runtime_impl->EventCreateWithFlags(&inputs_ready_event, core::EventFlag::kDisableTiming);
        runtime_impl->EventRecord(inputs_ready_event, main_compute_stream);
        runtime_impl->StreamWaitEvent(cp_compute_stream, inputs_ready_event, 0);

        std::vector<std::shared_ptr<Work>> pending_grad_p2p_works;
        std::vector<std::shared_ptr<Tensor>> pending_grad_send_tensors;
        std::vector<std::shared_ptr<Work>> kv_p2p_works;
        std::shared_ptr<Tensor> next_k;
        std::shared_ptr<Tensor> next_v;
        std::vector<std::shared_ptr<Tensor>> grad_q_chunks;
        grad_q_chunks.reserve(cp_size);

        for (int step = 0; step < cp_size; ++step) {
            auto *compute_stream = step % 2 == 0 ? main_compute_stream : cp_compute_stream;
            RuntimeStreamGuard stream_guard(q->GetDevice(), compute_stream);

            if (!kv_p2p_works.empty()) {
                for (const auto &work : kv_p2p_works) { work->WaitNonBlocking(); }
                kv_p2p_works.clear();
                current_k = next_k;
                current_v = next_v;
            }

            const int owner = (rank - step + cp_size) % cp_size;
            const int64_t kv_start = static_cast<int64_t>(owner) * local_t;
            const int64_t kv_end = kv_start + local_t;
            const bool chunk_fully_masked = owner > rank;

            if (step + 1 < cp_size) {
                // NOTE(zbl): For compute-comm overlap purposes, prefetch the next K/V chunk before computing the
                //            current one.
                // next_k: (B, H_kv, T_l, D)
                next_k = std::make_shared<Tensor>(k_local->Dims(), k_local->Dtype(), k_local->GetDevice());
                // next_v: (B, H_kv, T_l, D)
                next_v = std::make_shared<Tensor>(v_local->Dims(), v_local->Dtype(), v_local->GetDevice());
                kv_p2p_works = P2PCommunicate(rank, {current_k, current_v}, send_to, {next_k, next_v}, recv_from,
                                              cp_group, batch_p2p_comm);
            }

            // local_grad_k: (B, H_kv, T_l, D)
            // local_grad_v: (B, H_kv, T_l, D)
            std::shared_ptr<Tensor> local_grad_k;
            std::shared_ptr<Tensor> local_grad_v;

            // If chunk is fully masked, all grads are zero.
            if (!chunk_fully_masked) {
                // Recompute attention scores
                // k_for_attn: (B, H_q, T_l, D)
                auto k_for_attn = RepeatKVHeads(current_k, n_rep);
                // v_for_attn: (B, H_q, T_l, D)
                auto v_for_attn = RepeatKVHeads(current_v, n_rep);
                // scores: (B, H_q, T_l, T_l)
                auto scores = q->Matmul(k_for_attn->Transpose(-2, -1)) * scale;
                // invalid_mask: (1, 1, T_l, T_l), assume true values are invalid attention locations
                auto invalid_mask = mask->Slice(-1, kv_start, kv_end);
                scores = scores->MaskedFill(invalid_mask, std::numeric_limits<float>::lowest());

                // probs: (B, H_q, T_l, T_l)
                auto probs = (scores - softmax_max)->Exp() / softmax_sum;
                if (probs->Dtype() != scores->Dtype()) {
                    // NOTE(zbl): Online softmax is decomposed into FP32-policy ops such as Exp, while standard
                    //            SoftmaxForward returns the same dtype as its input scores. Cast back before
                    //            using Matmul backward kernels.
                    probs = std::make_shared<Tensor>(probs->To(scores->Dtype()));
                }
                // NOTE(zbl): MatmulBackwardXXX will perform a upcast to FP32. So use the same MatmulBackward kernels
                //            as the standard autograd path instead of calling Tensor::Matmul() to ensure this custom
                //            backward follows same dtype and accumulation policy as default MHA method.
                // grad_v_repeated: (B, H_q, T_l, D)
                auto grad_v_repeated = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
                    {device, "MatmulBackwardOther"}, probs, grad_output, v_for_attn->Dims());
                // grad_probs: (B, H_q, T_l, T_l)
                auto grad_probs = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
                    {device, "MatmulBackwardInput"}, v_for_attn, grad_output, probs->Dims());
                // grad_scores: (B, H_q, T_l, T_l)
                auto grad_scores = probs * (grad_probs - softmax_delta);
                // grad_scores_scaled: (B, H_q, T_l, T_l)
                auto grad_scores_scaled = grad_scores * scale;
                // grad_q_chunk: (B, H_q, T_l, D)
                auto grad_q_chunk = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
                    {device, "MatmulBackwardInput"}, k_for_attn->Transpose(-2, -1), grad_scores_scaled, q->Dims());
                grad_q_chunks.push_back(grad_q_chunk);
                // grad_k_transposed: (B, H_q, D, T_l)
                auto grad_k_transposed = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
                    {device, "MatmulBackwardOther"}, q, grad_scores_scaled, k_for_attn->Transpose(-2, -1)->Dims());
                // grad_k_repeated: (B, H_q, T_l, D)
                auto grad_k_repeated = grad_k_transposed->Transpose(-2, -1);

                // SumRepeatedKVHeads maps repeated GQA gradients from (B, H_q, T_l, D) to (B, H_kv, T_l, D).
                local_grad_k = SumRepeatedKVHeads(grad_k_repeated, n_rep);
                local_grad_v = SumRepeatedKVHeads(grad_v_repeated, n_rep);
            }

            std::shared_ptr<Tensor> next_grad_k;
            std::shared_ptr<Tensor> next_grad_v;

            // NOTE(zbl): The pass of accumulated dK/dV for this chunk was launched at the end of the previous step.
            //            For compute-comm overlap purposes, wait here before we are about to add into it.
            for (const auto &work : pending_grad_p2p_works) { work->WaitNonBlocking(); }
            pending_grad_p2p_works.clear();
            pending_grad_send_tensors.clear();

            if (local_grad_k) {
                current_grad_k = current_grad_k + local_grad_k;
                current_grad_v = current_grad_v + local_grad_v;
            }

            // NOTE(zbl): For compute-comm overlap purposes, after dK/dV is accumulated, launch the pass of dK/dV.
            // next_grad_k: (B, H_kv, T_l, D), same dtype as current_grad_k.
            next_grad_k = NewZeroTensorLike(current_grad_k);
            // next_grad_v: (B, H_kv, T_l, D), same dtype as current_grad_v.
            next_grad_v = NewZeroTensorLike(current_grad_v);
            pending_grad_send_tensors = {current_grad_k, current_grad_v};
            pending_grad_p2p_works = P2PCommunicate(rank, pending_grad_send_tensors, send_to,
                                                    {next_grad_k, next_grad_v}, recv_from, cp_group, batch_p2p_comm);
            current_grad_k = next_grad_k;
            current_grad_v = next_grad_v;
        }

        core::Event *cp_compute_done_event = nullptr;
        runtime_impl->EventCreateWithFlags(&cp_compute_done_event, core::EventFlag::kDisableTiming);
        runtime_impl->EventRecord(cp_compute_done_event, cp_compute_stream);
        runtime_impl->StreamWaitEvent(main_compute_stream, cp_compute_done_event, 0);
        runtime_impl->EventDestroy(inputs_ready_event);
        runtime_impl->EventDestroy(cp_compute_done_event);

        // NOTE(zbl): For compute-comm overlap purposes, wait for the last round of dK/dV pass before return.
        for (const auto &work : pending_grad_p2p_works) { work->WaitNonBlocking(); }
        pending_grad_p2p_works.clear();
        pending_grad_send_tensors.clear();

        for (const auto &grad_q_chunk : grad_q_chunks) { grad_q = grad_q + grad_q_chunk; }

        // Input is {q, k, v, mask}
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

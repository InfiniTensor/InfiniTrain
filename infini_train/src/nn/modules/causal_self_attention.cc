#include "infini_train/include/nn/modules/causal_self_attention.h"

#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_model.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

CausalSelfAttention::CausalSelfAttention(const TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType), config_(config) {
    SetupAttention(config);

    CHECK(spec.submodules_.contains(kCAttnLayerName))
        << "CausalSelfAttention spec missing submodule: " << kCAttnLayerName;
    CHECK(spec.submodules_.contains(kCProjLayerName))
        << "CausalSelfAttention spec missing submodule: " << kCProjLayerName;
    // Build submodules from spec
    modules_[kCAttnLayerName] = BuildModule(config, spec.submodules_.at(kCAttnLayerName));
    modules_[kCProjLayerName] = BuildModule(config, spec.submodules_.at(kCProjLayerName));

    // For standard attention (GPT2 style), precompute causal mask
    if (config_.attention_type == AttentionType::kStandard) {
        // causal mask: (1, 1, block_size, block_size)
        buffers_[kParamBiasName] = function::Tril(nn::function::Ones({config_.block_size, config_.block_size}))
                                       ->View({1, 1, config_.block_size, config_.block_size});
    }
}

void CausalSelfAttention::SetupAttention(const TransformerConfig &config) {
    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();

    CHECK_EQ(config.n_embd % config.n_head, 0) << "n_embd must be divisible by n_head";
    CHECK_EQ(config.n_head % tp_world_size, 0) << "n_head must be divisible by TP world size";

    n_head_ = config.n_head;
    n_embd_ = config.n_embd;
    head_dim_ = config.n_embd / config.n_head;
    local_n_head_ = n_head_ / tp_world_size;

    // For GQA, set n_kv_head and n_rep
    if (config.UseGQA()) {
        CHECK_EQ(config.n_head % config.n_kv_head, 0) << "n_head must be divisible by n_kv_head for GQA";
        CHECK_EQ(config.n_kv_head % tp_world_size, 0) << "n_kv_head must be divisible by TP world size for GQA";

        n_kv_head_ = config.n_kv_head;
        n_rep_ = n_head_ / n_kv_head_;
    } else {
        n_kv_head_ = n_head_;
        n_rep_ = 1;
    }
}

std::vector<std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    if (config_.attention_type == AttentionType::kRoPE) {
        return ForwardWithRoPE(x);
    } else {
        return ForwardStandard(x);
    }
}

std::vector<std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::ForwardStandard(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto tp_world_size = parallel::global::GetTensorParallelSize();

    const auto B = x[0]->Dims()[0];                  // bs
    const auto C = x[0]->Dims()[2];                  // n_embd
    const int64_t head_dim = n_embd_ / n_head_;      // per-head dim (global)
    const int64_t local_C = n_embd_ / tp_world_size; // per-rank hidden

    // (B, T, C) -> ColumnParallelLinear(C, 3*C) -> (B, T, 3 * local_C)
    // -> Split -> (3, B, T, local_C)
    auto qkv = (*modules_[kCAttnLayerName])(x)[0]->Split(local_C, 2);

    // (B, T, local_C)
    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    // NOTE(zbl): Acquire full T after AllGather is performed in ColumnParallelLinear
    const auto T = q->Dims()[1];

    // View to multi-head: local_n_head * head_dim == local_C
    // (B, T, local_C) -> (B, T, h_l, Dh) -> (B, h_l, T, Dh)
    k = k->View({B, T, local_n_head_, head_dim})->Transpose(1, 2);
    q = q->View({B, T, local_n_head_, head_dim})->Transpose(1, 2);
    v = v->View({B, T, local_n_head_, head_dim})->Transpose(1, 2);

    // (B, h_l, T, T)
    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(head_dim));
    // (1, 1, T, T)
    auto mask = buffers_[kParamBiasName]->Slice({0, 0, 0, 0}, {1, 1, T, T}, {1, 1, 1, 1});
    // (1, 1, T, T) -> eq 0 -> (1, 1, T, T) -> masked_fill -> (B, h_l, T, T)
    att = att->MaskedFill(mask == 0, -std::numeric_limits<float>::infinity());
    // (B, h_l, T, T)
    att = nn::function::Softmax(att, -1);
    // (B, h_l, T, Dh)
    auto y = att->Matmul(v);
    // (B, h_l, T, Dh) -> (B, T, h_l, Dh) -> (B, T, local_C)
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, local_C});

    // Get full tensor
    // (B, T, local_C) -> RowParallelLinear(n_embd, n_embd) -> (B, T, C)
    y = (*modules_[kCProjLayerName])({y})[0];
    // (B, T, C) == (bs, seq_len, n_embd)
    return {y};
}

// RoPE helper methods
std::tuple<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::ApplyRotaryEmbedding(const std::shared_ptr<infini_train::Tensor> &xq,
                                          const std::shared_ptr<infini_train::Tensor> &xk,
                                          const std::shared_ptr<infini_train::Tensor> &freqs_cis) {
    // Reshape freqs_cis for broadcasting
    const auto &x_shape = xq->Dims(); // (B, T, H, D)
    const int64_t T = x_shape[1];
    const int64_t D = x_shape[3];

    std::vector<int64_t> target_shape = {1, T, 1, D / 2, 2};
    auto cos_sin = freqs_cis->View(target_shape); // -> (1, T, 1, D/2, 2)

    auto cos = cos_sin->Slice(-1, 0, 1, 1)->Squeeze(-1); // (1, T, 1, D/2)
    auto sin = cos_sin->Slice(-1, 1, 2, 1)->Squeeze(-1); // (1, T, 1, D/2)

    auto slice_pair = [](const std::shared_ptr<Tensor> &x) {
        auto even = x->Slice(-1, 0, x->Dims().back(), 2);
        auto odd = x->Slice(-1, 1, x->Dims().back(), 2);
        return std::make_pair(even, odd);
    };

    auto [q_even, q_odd] = slice_pair(xq);
    auto q_rotated_left = q_even * cos - q_odd * sin;
    auto q_rotated_right = q_even * sin + q_odd * cos;
    auto q_rotated
        = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{q_rotated_left, q_rotated_right}, -1)->Flatten(-2);

    auto [k_even, k_odd] = slice_pair(xk);
    auto k_rotated_left = k_even * cos - k_odd * sin;
    auto k_rotated_right = k_even * sin + k_odd * cos;
    auto k_rotated
        = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{k_rotated_left, k_rotated_right}, -1)->Flatten(-2);

    return {q_rotated, k_rotated};
}

std::shared_ptr<infini_train::Tensor> CausalSelfAttention::RepeatKV(const std::shared_ptr<infini_train::Tensor> &x,
                                                                    int64_t n_rep) {
    const auto &shape = x->Dims();
    const int64_t B = shape[0], T = shape[1], H = shape[2], D = shape[3];

    if (n_rep == 1) {
        return x;
    }

    return x->View({B, T, H, 1, D})->RepeatInterleave(n_rep, 3)->Contiguous()->View({B, T, H * n_rep, D});
}

std::vector<std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::ForwardWithRoPE(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    const auto B = x[0]->Dims()[0]; // bs
    const auto C = x[0]->Dims()[2]; // n_embd

    const auto tp_size = nn::parallel::global::GetTensorParallelSize();

    const auto C_local = C / tp_size;
    const auto H_local = n_head_ / tp_size;
    const auto KV_local = n_kv_head_ / tp_size;
    const auto D = head_dim_; // n_embd / n_head

    const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
    const auto start_pos = x.size() > 2 ? x[2] : nullptr;
    const auto mask = x.size() > 3 ? x[3] : nullptr;
    CHECK(freqs_cis != nullptr) << "freqs_cis is null.";

    // (B, T, C) -> (B, T, (H + 2 * n_kv_head) * D)
    auto qkv = (*modules_[kCAttnLayerName])({x[0]})[0];
    // NOTE(zbl): Acquire full T after AllGather is performed in ColumnParallelLinear
    const auto T = qkv->Dims()[1];
    // NOTE(zbl): torch script uses torch.split({...}, dim) to split tensors into sub-tensors in different sizes
    //            use Slice() to work around here
    const int64_t q_size_local = H_local * D;
    const int64_t kv_size_local = KV_local * D;
    // -> Split into q, k, v
    // q: (B, T, H_local, D)
    auto q = qkv->Slice(2, 0, q_size_local)->View({B, T, H_local, D});
    // k: (B, T, KV_local, D)
    auto k = qkv->Slice(2, q_size_local, q_size_local + kv_size_local)->View({B, T, KV_local, D});
    // v: (B, T, KV_local, D)
    auto v = qkv->Slice(2, q_size_local + kv_size_local, q_size_local + 2 * kv_size_local)->View({B, T, KV_local, D});

    // -> RoPE on q, k
    // q: (B, T, H_local, D)
    // k: (B, T, KV_local, D)
    std::tie(q, k) = ApplyRotaryEmbedding(q, k, freqs_cis);

    // TODO(zbl): use kv cache during inference
    // if (use_kv_) { ... }

    // align n_head in GQA
    // (B, T, KV_local, D) -> (B, T, H_local, D) via RepeatKV
    k = RepeatKV(k, n_rep_);
    v = RepeatKV(v, n_rep_);

    // (B, T, H_local, D) -> (B, H_local, T, D)
    q = q->Transpose(1, 2);
    k = k->Transpose(1, 2);
    v = v->Transpose(1, 2);

    // TODO(zbl): support flash attention later
    // if (flash_) { ... }

    // manual implementation of attention
    // this materializes the large (T,T) matrix for all the queries and keys

    // q: (B, H_local, T, D)
    // k: (B, H_local, T, D) -> (B, H_local, D, T)
    // q @ k.T: (B, H_local, T, T) -> mul 1.0 / sqrt(D) -> (B, H_local, T, T)
    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(static_cast<float>(D)));
    if (mask) {
        // mask: (1, 1, T, T)
        att = att->MaskedFill(mask, std::numeric_limits<float>::lowest());
    }
    // (B, H_local, T, T)
    att = nn::function::Softmax(att, -1);
    // att: (B, H_local, T, T) @ v: (B, H_local, T, D) -> y: (B, H_local, T, D)
    auto y = att->Matmul(v);
    // (B, H_local, T, D) -> Transpose(1, 2) -> (B, T, H_local, D) -> (B, T, C_local)
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, C_local});
    // output projection
    // (B, T, C_local) -> RowParallelLinear(C, C) -> (B, T, C)
    y = (*modules_[kCProjLayerName])({y})[0];
    // (B, H, C) == (bs, seq_len, n_embd)
    return {y};
}

} // namespace infini_train::nn

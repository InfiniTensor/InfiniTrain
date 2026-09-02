#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"

#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

CausalSelfAttention::CausalSelfAttention(const TransformerConfig &config) : CloneableModule(kType), config_(config) {
    SetupAttention(config);

    if (config_.attention_variant == AttentionVariant::kFM9G) {
        modules_[kQAProjLayerName] = std::make_shared<nn::Linear>(n_embd_, q_lora_rank_, false);
        modules_[kQALayerNormLayerName] = std::make_shared<nn::RMSNorm>(q_lora_rank_, config_.norm_eps);
        modules_[kQBProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
            q_lora_rank_, n_head_ * q_head_dim_, false, false, false, false,
            nn::parallel::global::GetSequenceParallelEnabled());
        modules_[kKVAProjWithMQALayerName]
            = std::make_shared<nn::Linear>(n_embd_, kv_lora_rank_ + qk_rope_head_dim_, false);
        modules_[kKVALayerNormLayerName] = std::make_shared<nn::RMSNorm>(kv_lora_rank_, config_.norm_eps);
        modules_[kKVBProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
            kv_lora_rank_, n_head_ * (qk_nope_head_dim_ + v_head_dim_), false, false, false, false,
            nn::parallel::global::GetSequenceParallelEnabled());
        modules_[kOProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
            n_head_ * v_head_dim_, n_embd_, false, true, true, false,
            nn::parallel::global::GetSequenceParallelEnabled());
        return;
    }

    int64_t qkv_dim = (config.n_head + 2 * n_kv_head_) * head_dim_;
    // qkv: ColumnParallel (do not gather output)
    modules_[kCAttnLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/n_embd_,
        /*out_features=*/qkv_dim,
        /*bias=*/config_.add_bias_linear,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // proj: RowParallel (input is parallel and output is full)
    modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
        /*in_features=*/n_embd_,
        /*out_features=*/n_embd_,
        /*bias=*/config_.add_bias_linear,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // FIXME(zbl): Decouple causal-mask ownership from position embedding. For now, only learned-absolute models use
    //             this precomputed buffer; RoPE callers provide a runtime-sized mask.
    if (config_.position_embedding_type == PositionEmbeddingType::kLearnedAbsolute) {
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

    if (config.attention_variant == AttentionVariant::kFM9G) {
        q_lora_rank_ = config.q_lora_rank;
        kv_lora_rank_ = config.kv_lora_rank;
        qk_nope_head_dim_ = config.qk_nope_head_dim;
        qk_rope_head_dim_ = config.qk_rope_head_dim;
        v_head_dim_ = config.v_head_dim;
        q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
        CHECK_GT(q_lora_rank_, 0);
        CHECK_GT(kv_lora_rank_, 0);
        CHECK_GT(qk_nope_head_dim_, 0);
        CHECK_GT(qk_rope_head_dim_, 0);
        CHECK_GT(v_head_dim_, 0);
        n_kv_head_ = n_head_;
        n_rep_ = 1;
        return;
    }

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
CausalSelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    if (config_.attention_variant == AttentionVariant::kFM9G) { return ForwardFM9G(x); }
    const auto B = x[0]->Dims()[0]; // bs
    const auto C = x[0]->Dims()[2]; // n_embd

    const auto tp_size = nn::parallel::global::GetTensorParallelSize();

    const auto C_local = C / tp_size;
    const auto H_local = local_n_head_;
    const auto KV_local = n_kv_head_ / tp_size;
    const auto D = head_dim_; // n_embd / n_head

    const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
    const auto start_pos = x.size() > 2 ? x[2] : nullptr;
    const auto mask = x.size() > 3 ? x[3] : nullptr;
    if (config_.position_embedding_type == PositionEmbeddingType::kRoPE) {
        CHECK(freqs_cis != nullptr) << "freqs_cis is null.";
    }

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

    if (config_.position_embedding_type == PositionEmbeddingType::kRoPE) {
        // q: (B, T, H_local, D), k: (B, T, KV_local, D)
        std::tie(q, k) = ApplyRotaryEmbedding(q, k, freqs_cis);
    }

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
    } else {
        // fallback causal mask: (1, 1, T, T)
        auto causal_mask = buffers_[kParamBiasName]->Slice({0, 0, 0, 0}, {1, 1, T, T}, {1, 1, 1, 1});
        att = att->MaskedFill(causal_mask == 0, -std::numeric_limits<float>::infinity());
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

std::vector<std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::ForwardFM9G(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    const auto B = x[0]->Dims()[0];
    const auto tp_size = nn::parallel::global::GetTensorParallelSize();
    const auto H_local = n_head_ / tp_size;
    const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
    const auto mask = x.size() > 3 ? x[3] : nullptr;
    CHECK(freqs_cis != nullptr) << "FM9G attention requires RoPE frequencies";

    auto q_latent = (*modules_[kQAProjLayerName])({x[0]})[0];
    q_latent = (*modules_[kQALayerNormLayerName])({q_latent})[0];
    auto q = (*modules_[kQBProjLayerName])({q_latent})[0];
    const auto T = q->Dims()[1];
    q = q->View({B, T, H_local, q_head_dim_});
    auto q_nope = q->Slice(-1, 0, qk_nope_head_dim_);
    auto q_pe = q->Slice(-1, qk_nope_head_dim_, q_head_dim_);

    auto kv_a = (*modules_[kKVAProjWithMQALayerName])({x[0]})[0];
    auto compressed_kv = kv_a->Slice(-1, 0, kv_lora_rank_);
    auto k_pe = kv_a->Slice(-1, kv_lora_rank_, kv_lora_rank_ + qk_rope_head_dim_)
                    ->View({B, T, 1, qk_rope_head_dim_});
    auto kv = (*modules_[kKVALayerNormLayerName])({compressed_kv})[0];
    kv = (*modules_[kKVBProjLayerName])({kv})[0]->View({B, T, H_local, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->Slice(-1, 0, qk_nope_head_dim_);
    auto v = kv->Slice(-1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_);

    std::tie(q_pe, k_pe) = ApplyFM9GRotaryEmbedding(q_pe, k_pe, freqs_cis);
    k_pe = RepeatKV(k_pe, H_local);
    q = nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{q_nope, q_pe}, -1);
    auto k = nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{k_nope, k_pe}, -1);
    q = q->Transpose(1, 2);
    k = k->Transpose(1, 2);
    v = v->Transpose(1, 2);

    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(static_cast<float>(q_head_dim_)));
    if (mask) { att = att->MaskedFill(mask, std::numeric_limits<float>::lowest()); }
    att = nn::function::Softmax(att, -1);
    auto y = att->Matmul(v)->Transpose(1, 2)->Contiguous()->View({B, T, H_local * v_head_dim_});
    return {(*modules_[kOProjLayerName])({y})[0]};
}

} // namespace infini_train::nn

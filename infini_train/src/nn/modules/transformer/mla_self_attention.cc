#include "infini_train/include/nn/modules/transformer/mla_self_attention.h"

#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
int64_t DefaultQKVHeadDim(const TransformerConfig &config) {
    CHECK_EQ(config.n_embd % config.n_head, 0) << "n_embd must be divisible by n_head";
    return config.n_embd / config.n_head;
}

int64_t DefaultQKRoPEHeadDim(const TransformerConfig &config) {
    return DefaultQKVHeadDim(config);
}

int64_t DefaultQKNoPEHeadDim(const TransformerConfig &config) {
    return DefaultQKVHeadDim(config);
}
} // namespace

MLASelfAttention::MLASelfAttention(const TransformerConfig &config)
    : MLASelfAttention(config,
                       /*q_lora_rank=*/config.n_embd,
                       /*kv_lora_rank=*/config.n_embd,
                       /*qk_nope_head_dim=*/DefaultQKNoPEHeadDim(config),
                       /*qk_rope_head_dim=*/DefaultQKRoPEHeadDim(config),
                       /*v_head_dim=*/DefaultQKVHeadDim(config)) {}

MLASelfAttention::MLASelfAttention(const TransformerConfig &config, int64_t q_lora_rank, int64_t kv_lora_rank,
                                   int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t v_head_dim)
    : CloneableModule(kType), config_(config) {
    SetupAttention(config, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);

    modules_[kQAProjLayerName] = std::make_shared<nn::Linear>(
        /*in_features=*/n_embd_,
        /*out_features=*/q_lora_rank_,
        /*bias=*/config_.add_bias_linear);
    modules_[kQANormLayerName] = std::make_shared<nn::RMSNorm>(q_lora_rank_, config_.norm_eps);
    modules_[kQBProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/q_lora_rank_,
        /*out_features=*/n_head_ * qk_head_dim_,
        /*bias=*/config_.add_bias_linear,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    modules_[kKVAProjLayerName] = std::make_shared<nn::Linear>(
        /*in_features=*/n_embd_,
        /*out_features=*/kv_lora_rank_ + qk_rope_head_dim_,
        /*bias=*/config_.add_bias_linear);
    modules_[kKVANormLayerName] = std::make_shared<nn::RMSNorm>(kv_lora_rank_, config_.norm_eps);
    modules_[kKVBProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/kv_lora_rank_,
        /*out_features=*/n_head_ * (qk_nope_head_dim_ + v_head_dim_),
        /*bias=*/config_.add_bias_linear,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
        /*in_features=*/n_head_ * v_head_dim_,
        /*out_features=*/n_embd_,
        /*bias=*/config_.add_bias_linear,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    buffers_[kParamBiasName] = function::Tril(nn::function::Ones({config_.block_size, config_.block_size}))
                                   ->View({1, 1, config_.block_size, config_.block_size});
}

void MLASelfAttention::SetupAttention(const TransformerConfig &config, int64_t q_lora_rank, int64_t kv_lora_rank,
                                      int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t v_head_dim) {
    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();

    CHECK_EQ(config.n_embd % config.n_head, 0) << "n_embd must be divisible by n_head";
    CHECK_EQ(config.n_head % tp_world_size, 0) << "n_head must be divisible by TP world size";
    CHECK_GT(q_lora_rank, 0) << "q_lora_rank must be positive";
    CHECK_GT(kv_lora_rank, 0) << "kv_lora_rank must be positive";
    CHECK_GT(qk_nope_head_dim, 0) << "qk_nope_head_dim must be positive";
    CHECK_GT(qk_rope_head_dim, 0) << "qk_rope_head_dim must be positive";
    CHECK_GT(v_head_dim, 0) << "v_head_dim must be positive";
    CHECK_EQ(qk_rope_head_dim % 2, 0) << "qk_rope_head_dim must be even for RoPE";

    n_head_ = config.n_head;
    n_embd_ = config.n_embd;
    local_n_head_ = n_head_ / tp_world_size;

    q_lora_rank_ = q_lora_rank;
    kv_lora_rank_ = kv_lora_rank;
    qk_nope_head_dim_ = qk_nope_head_dim;
    qk_rope_head_dim_ = qk_rope_head_dim;
    qk_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    v_head_dim_ = v_head_dim;
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MLASelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_GE(x.size(), 1) << "MLASelfAttention expects at least hidden states";

    const auto B = x[0]->Dims()[0];
    const auto C = x[0]->Dims()[2];
    CHECK_EQ(C, n_embd_) << "hidden size must match n_embd";

    const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
    const auto external_mask = x.size() > 3 ? x[3] : nullptr;
    if (config_.attention_type == AttentionType::kRoPE) {
        CHECK(freqs_cis != nullptr) << "freqs_cis is null.";
    }

    // (B, T, C) -> q_a -> RMSNorm -> q_b -> (B, T, H_local * (D_nope + D_rope))
    auto q = (*modules_[kQAProjLayerName])({x[0]})[0];
    q = (*modules_[kQANormLayerName])({q})[0];
    q = (*modules_[kQBProjLayerName])({q})[0];
    const auto T = q->Dims()[1];
    q = q->View({B, T, local_n_head_, qk_head_dim_});

    auto q_nope = q->Slice(-1, 0, qk_nope_head_dim_);
    auto q_pe = q->Slice(-1, qk_nope_head_dim_, qk_head_dim_);

    // (B, T, C) -> kv_a -> compressed kv latent and shared RoPE key.
    auto compressed_kv_with_pe = (*modules_[kKVAProjLayerName])({x[0]})[0];
    auto compressed_kv = compressed_kv_with_pe->Slice(-1, 0, kv_lora_rank_);
    auto k_pe = compressed_kv_with_pe->Slice(-1, kv_lora_rank_, kv_lora_rank_ + qk_rope_head_dim_)
                    ->Contiguous();
    if (nn::parallel::global::GetSequenceParallelEnabled()) {
        k_pe = nn::parallel::GatherFromSPRegionFunc(k_pe)[0];
    }
    k_pe = k_pe->View({B, T, 1, qk_rope_head_dim_});

    // (B, T, R_kv) -> RMSNorm -> kv_b -> (B, T, H_local * (D_nope + D_v))
    auto kv = (*modules_[kKVANormLayerName])({compressed_kv})[0];
    kv = (*modules_[kKVBProjLayerName])({kv})[0];
    kv = kv->View({B, T, local_n_head_, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->Slice(-1, 0, qk_nope_head_dim_);
    auto v = kv->Slice(-1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_);

    if (config_.attention_type == AttentionType::kRoPE) {
        std::tie(q_pe, k_pe) = ApplyRotaryEmbedding(q_pe, k_pe, freqs_cis);
    }

    k_pe = k_pe->RepeatInterleave(local_n_head_, 2);
    q = nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{q_nope, q_pe}, -1);
    auto k = nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{k_nope, k_pe}, -1);

    // (B, T, H_local, D) -> (B, H_local, T, D)
    q = q->Transpose(1, 2);
    k = k->Transpose(1, 2);
    v = v->Transpose(1, 2);

    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(static_cast<float>(qk_head_dim_)));
    if (external_mask) {
        att = att->MaskedFill(external_mask, std::numeric_limits<float>::lowest());
    } else {
        auto mask = buffers_[kParamBiasName]->Slice({0, 0, 0, 0}, {1, 1, T, T}, {1, 1, 1, 1});
        att = att->MaskedFill(mask == 0, -std::numeric_limits<float>::infinity());
    }
    att = nn::function::Softmax(att, -1);

    auto y = att->Matmul(v);
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, local_n_head_ * v_head_dim_});
    y = (*modules_[kCProjLayerName])({y})[0];
    return {y};
}

} // namespace infini_train::nn

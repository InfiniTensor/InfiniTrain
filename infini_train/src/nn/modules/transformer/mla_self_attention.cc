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

MLASelfAttention::MLASelfAttention(const TransformerConfig &config) : CloneableModule(kType), config_(config) {
    SetupAttention(config);

    if (use_q_lora_) {
        if (q_down_proj_use_tp_) {
            modules_[kLinearQDownProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
                /*in_features=*/n_embd_,
                /*out_features=*/q_lora_rank_,
                /*bias=*/config_.add_bias_linear,
                /*gather_output=*/false,
                /*input_is_parallel=*/false,
                /*skip_bias_add=*/false,
                /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
        } else {
            modules_[kLinearQDownProjLayerName] = std::make_shared<nn::Linear>(
                /*in_features=*/n_embd_,
                /*out_features=*/q_lora_rank_,
                /*bias=*/config_.add_bias_linear);
        }
        modules_[kQLayerNormLayerName] = std::make_shared<nn::RMSNorm>(q_lora_rank_, config_.norm_eps);
        modules_[kLinearQUpProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
            /*in_features=*/q_lora_rank_,
            /*out_features=*/n_head_ * qk_head_dim_,
            /*bias=*/config_.add_bias_linear,
            /*gather_output=*/false,
            /*input_is_parallel=*/false,
            /*skip_bias_add=*/false,
            /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
    } else {
        modules_[kLinearQProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
            /*in_features=*/n_embd_,
            /*out_features=*/n_head_ * qk_head_dim_,
            /*bias=*/config_.add_bias_linear,
            /*gather_output=*/false,
            /*input_is_parallel=*/false,
            /*skip_bias_add=*/false,
            /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
    }

    if (kv_down_proj_use_tp_) {
        modules_[kLinearKVDownProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
            /*in_features=*/n_embd_,
            /*out_features=*/kv_lora_rank_ + qk_rope_head_dim_,
            /*bias=*/config_.add_bias_linear,
            /*gather_output=*/false,
            /*input_is_parallel=*/false,
            /*skip_bias_add=*/false,
            /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
    } else {
        modules_[kLinearKVDownProjLayerName] = std::make_shared<nn::Linear>(
            /*in_features=*/n_embd_,
            /*out_features=*/kv_lora_rank_ + qk_rope_head_dim_,
            /*bias=*/config_.add_bias_linear);
    }
    modules_[kKVLayerNormLayerName] = std::make_shared<nn::RMSNorm>(kv_lora_rank_, config_.norm_eps);
    modules_[kLinearKVUpProjLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/kv_lora_rank_,
        /*out_features=*/n_head_ * (qk_nope_head_dim_ + v_head_dim_),
        /*bias=*/config_.add_bias_linear,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    modules_[kLinearProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
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

void MLASelfAttention::SetupAttention(const TransformerConfig &config) {
    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();

    CHECK_EQ(config.n_embd % config.n_head, 0) << "n_embd must be divisible by n_head";
    CHECK_EQ(config.n_head % tp_world_size, 0) << "n_head must be divisible by TP world size";
    CHECK(!config.q_lora_rank.has_value() || config.q_lora_rank.value() > 0) << "q_lora_rank must be positive when set";

    const auto default_head_dim = config.n_embd / config.n_head;
    const int64_t kv_lora_rank = config.kv_lora_rank > 0 ? config.kv_lora_rank : config.n_embd;
    const int64_t qk_nope_head_dim = config.qk_nope_head_dim > 0 ? config.qk_nope_head_dim : default_head_dim;
    const int64_t qk_rope_head_dim = config.qk_rope_head_dim > 0 ? config.qk_rope_head_dim : default_head_dim;
    const int64_t v_head_dim = config.v_head_dim > 0 ? config.v_head_dim : default_head_dim;

    CHECK_GT(qk_nope_head_dim, 0) << "qk_nope_head_dim must be positive";
    CHECK_GT(qk_rope_head_dim, 0) << "qk_rope_head_dim must be positive";
    CHECK_GT(v_head_dim, 0) << "v_head_dim must be positive";
    CHECK_EQ(qk_rope_head_dim % 2, 0) << "qk_rope_head_dim must be even for RoPE";

    n_head_ = config.n_head;
    n_embd_ = config.n_embd;
    local_n_head_ = n_head_ / tp_world_size;

    use_q_lora_ = config.q_lora_rank.has_value();
    q_lora_rank_ = config.q_lora_rank.value_or(0);
    kv_lora_rank_ = kv_lora_rank;
    qk_nope_head_dim_ = qk_nope_head_dim;
    qk_rope_head_dim_ = qk_rope_head_dim;
    qk_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    v_head_dim_ = v_head_dim;
    q_down_proj_use_tp_ = config.q_down_proj_use_tp;
    kv_down_proj_use_tp_ = config.kv_down_proj_use_tp;
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MLASelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_GE(x.size(), 1) << "MLASelfAttention expects at least hidden states";

    // x[0]: (B, T_local, C)
    const auto B = x[0]->Dims()[0];
    const auto C = x[0]->Dims()[2];
    CHECK_EQ(C, n_embd_) << "hidden size must match n_embd";

    // freqs_cis: (T, D_rope / 2, 2)
    const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
    // external_mask: (1, 1, T, T)
    const auto external_mask = x.size() > 3 ? x[3] : nullptr;
    if (config_.attention_type == AttentionType::kRoPE) {
        CHECK(freqs_cis != nullptr) << "freqs_cis is null.";
    }

    const bool sequence_parallel_enabled = nn::parallel::global::GetSequenceParallelEnabled();

    // ----------- Q PATH -----------
    // Q path, align with Megatron:
    //     - q_lora_rank == nullopt -> linear_q_proj directly;
    //     - otherwise linear_q_down_proj -> q_layernorm -> linear_q_up_proj.
    std::shared_ptr<Tensor> q;
    if (use_q_lora_) {
        // linear_q_down_proj:
        //   non-TP path: (B, T_local, C) -> (B, T_local, R_q)
        //   TP path before gather: (B, T, C) -> (B, T, R_q / TP)
        //      - Note that ColumnParallelLinear would perform a GatherFromSPRegion in the beginning
        auto q_compressed = (*modules_[kLinearQDownProjLayerName])({x[0]})[0];
        if (q_down_proj_use_tp_ && q_compressed->Dims().back() != q_lora_rank_) {
            // Gather the sharded latent dimension: (B, T, R_q / TP) -> (B, T, R_q).
            q_compressed = nn::parallel::GatherFromTPRegionFunc(q_compressed)[0];
            if (sequence_parallel_enabled) {
                // Keep the q_up input sequence-sharded: (B, T_full, R_q) -> (B, T_local, R_q).
                q_compressed = nn::parallel::ScatterToSPRegionFunc(q_compressed)[0];
            }
        }
        // q_layernorm preserves shape: (B, T_local, R_q)
        q_compressed = (*modules_[kQLayerNormLayerName])({q_compressed})[0];
        // linear_q_up_proj: (B, T_local, R_q) -> (B, T, H_local * (D_nope + D_rope)).
        q = (*modules_[kLinearQUpProjLayerName])({q_compressed})[0];
    } else {
        // linear_q_proj direct path: (B, T, C) -> (B, T, H_local * (D_nope + D_rope)).
        q = (*modules_[kLinearQProjLayerName])({x[0]})[0];
    }

    // T should be the full seqlen after the q projection path gathers sequence-parallel input.
    const auto T = q->Dims()[1];
    // q: (B, T, H_local * D_qk) -> (B, T, H_local, D_qk)
    // qk_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_
    q = q->View({B, T, local_n_head_, qk_head_dim_});

    // q_nope: (B, T, H_local, D_nope), q_pos_emb: (B, T, H_local, D_rope)
    auto q_nope = q->Slice(-1, 0, qk_nope_head_dim_);
    auto q_pos_emb = q->Slice(-1, qk_nope_head_dim_, qk_head_dim_);

    // ----------- KV PATH -----------
    // linear_kv_down_proj:
    //     non-TP path: (B, T_local, C) -> (B, T_local, R_kv + D_rope)
    //     TP path before gather: (B, T, C) -> (B, T, (R_kv + D_rope) / TP)
    auto compressed_kv_with_pe = (*modules_[kLinearKVDownProjLayerName])({x[0]})[0];
    const auto kv_down_proj_out_dim = kv_lora_rank_ + qk_rope_head_dim_;
    const bool kv_down_proj_output_is_sharded = compressed_kv_with_pe->Dims().back() != kv_down_proj_out_dim;
    if (kv_down_proj_use_tp_ && kv_down_proj_output_is_sharded) {
        // Gather latent+RoPE dim: (B, T, (R_kv + D_rope) / TP) -> (B, T, R_kv + D_rope)
        compressed_kv_with_pe = nn::parallel::GatherFromTPRegionFunc(compressed_kv_with_pe)[0];
    }

    // compressed_kv: (B, T_local, R_kv), k_pos_emb: (B, T_local, D_rope)
    auto compressed_kv = compressed_kv_with_pe->Slice(-1, 0, kv_lora_rank_);
    auto k_pos_emb = compressed_kv_with_pe->Slice(-1, kv_lora_rank_, kv_lora_rank_ + qk_rope_head_dim_)->Contiguous();
    const bool k_pos_emb_has_full_sequence
        = kv_down_proj_use_tp_ && kv_down_proj_output_is_sharded && sequence_parallel_enabled;
    if (k_pos_emb_has_full_sequence) {
        // k_pos_emb already has full T; keep only compressed_kv sequence-sharded for linear_kv_up_proj.
        // compressed_kv: (B, T, R_kv) -> (B, T_local, R_kv)
        compressed_kv = nn::parallel::ScatterToSPRegionFunc(compressed_kv)[0];
    } else if (sequence_parallel_enabled) {
        // Replicated down-proj path produces local k_pos_emb; gather it for attention.
        // k_pos_emb: (B, T_local, D_rope) -> (B, T, D_rope)
        k_pos_emb = nn::parallel::GatherFromSPRegionFunc(k_pos_emb)[0];
    }
    // k_pos_emb: (B, T, D_rope) -> (B, T, 1, D_rope), shared across local heads.
    k_pos_emb = k_pos_emb->View({B, T, 1, qk_rope_head_dim_});

    // (B, T, R_kv) -> kv_layernorm -> linear_kv_up_proj -> (B, T, H_local * (D_nope + D_v))
    // kv_layernorm preserves compressed_kv shape: (B, T_local, R_kv)
    auto kv = (*modules_[kKVLayerNormLayerName])({compressed_kv})[0];
    // linear_kv_up_proj: (B, T_local, R_kv) -> (B, T, H_local * (D_nope + D_v))
    kv = (*modules_[kLinearKVUpProjLayerName])({kv})[0];
    // kv: (B, T, H_local * (D_nope + D_v)) -> (B, T, H_local, D_nope + D_v)
    kv = kv->View({B, T, local_n_head_, qk_nope_head_dim_ + v_head_dim_});
    // k_nope: (B, T, H_local, D_nope), v: (B, T, H_local, D_v)
    auto k_nope = kv->Slice(-1, 0, qk_nope_head_dim_);
    auto v = kv->Slice(-1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_);

    if (config_.attention_type == AttentionType::kRoPE) {
        // q_pos_emb: (B, T, H_local, D_rope), k_pos_emb: (B, T, 1, D_rope)
        std::tie(q_pos_emb, k_pos_emb) = ApplyRotaryEmbedding(q_pos_emb, k_pos_emb, freqs_cis);
    }

    // k_pos_emb: (B, T, 1, D_rope) -> (B, T, H_local, D_rope)
    k_pos_emb = k_pos_emb->RepeatInterleave(local_n_head_, 2);
    // q: (B, T, H_local, D_qk), k: (B, T, H_local, D_qk)
    q = nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{q_nope, q_pos_emb}, -1);
    auto k = nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{k_nope, k_pos_emb}, -1);

    // ----------- CORE ATTN -----------
    // q/k: (B, T, H_local, D_qk) -> (B, H_local, T, D_qk)
    // v:   (B, T, H_local, D_v)   -> (B, H_local, T, D_v)
    q = q->Transpose(1, 2);
    k = k->Transpose(1, 2);
    v = v->Transpose(1, 2);

    // att: (B, H_local, T, T)
    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(static_cast<float>(qk_head_dim_)));
    if (external_mask) {
        att = att->MaskedFill(external_mask, std::numeric_limits<float>::lowest());
    } else {
        // mask: (1, 1, T, T)
        auto mask = buffers_[kParamBiasName]->Slice({0, 0, 0, 0}, {1, 1, T, T}, {1, 1, 1, 1});
        att = att->MaskedFill(mask == 0, -std::numeric_limits<float>::infinity());
    }
    // att: (B, H_local, T, T)
    att = nn::function::Softmax(att, -1);

    // y: (B, H_local, T, D_v)
    auto y = att->Matmul(v);
    // y: (B, H_local, T, D_v) -> (B, T, H_local, D_v) -> (B, T, H_local * D_v)
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, local_n_head_ * v_head_dim_});
    // linear_proj: (B, T, H_local * D_v) -> (B, T, C)
    y = (*modules_[kLinearProjLayerName])({y})[0];

    return {y};
}

} // namespace infini_train::nn

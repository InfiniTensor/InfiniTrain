#include "infini_train/include/core/transformer/transformer_builders.h"

#include <cmath>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/core/transformer/activations/gelu.h"
#include "infini_train/include/core/transformer/activations/swiglu.h"
#include "infini_train/include/core/transformer/attention/causal_self_attention.h"
#include "infini_train/include/core/transformer/mlp.h"
#include "infini_train/include/core/transformer/norms/layer_norm.h"
#include "infini_train/include/core/transformer/norms/rms_norm.h"
#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_layer.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"

namespace infini_train::nn {

ModuleSpec BuildNormSpec(const TransformerConfig &config) {
    ModuleSpec spec;
    switch (config.norm_type) {
    case NormType::kLayerNorm:
        spec = ModuleSpec(typeid(LayerNorm));
        spec.with_param(kNormalizedShape, std::vector<int64_t>{config.n_embd});
        break;
    case NormType::kRMSNorm:
        spec = ModuleSpec(typeid(RMSNorm));
        spec.with_param(kDim, static_cast<int>(config.n_embd)).with_param(kEps, config.norm_eps);
        break;
    default:
        LOG(FATAL) << "Unsupported norm type";
    }
    return spec;
}

ModuleSpec BuildAttentionSpec(const TransformerConfig &config) {
    ModuleSpec spec(typeid(CausalSelfAttention));

    // Calculate QKV output dimension based on attention type and GQA
    int64_t qkv_out;
    if (config.use_gqa && config.n_kv_head < config.n_head) {
        // GQA style (LLaMA3 with GQA enabled)
        int64_t head_dim = config.n_embd / config.n_head;
        // qkv_out = config.n_embd + 2 * config.n_kv_head * head_dim;
        qkv_out = (config.n_head + 2 * config.n_kv_head) * head_dim;
    } else {
        // Standard MHA style (GPT2, or models without GQA)
        qkv_out = 3 * config.n_embd;
    }

    // Build c_attn (QKV projection)
    ModuleSpec c_attn_spec(typeid(parallel::ColumnParallelLinear));
    c_attn_spec.with_param(kInFeatures, static_cast<int>(config.n_embd))
        .with_param(kOutFeatures, static_cast<int>(qkv_out))
        .with_param(kBias, config.use_bias);
    spec.with_submodule(CausalSelfAttention::kCAttnLayerName, c_attn_spec);

    // Build c_proj (output projection)
    ModuleSpec c_proj_spec(typeid(parallel::RowParallelLinear));
    c_proj_spec.with_param(kInFeatures, static_cast<int>(config.n_embd))
        .with_param(kOutFeatures, static_cast<int>(config.n_embd))
        .with_param(kBias, config.use_bias);
    spec.with_submodule(CausalSelfAttention::kCProjLayerName, c_proj_spec);

    return spec;
}

ModuleSpec BuildMLPSpec(const TransformerConfig &config) {
    ModuleSpec spec(typeid(MLP));

    // Compute hidden dimension
    // Base dimension: n_embd * ffn_expansion_ratio
    int64_t ffn_hidden = static_cast<int64_t>(config.n_embd * config.ffn_expansion_ratio);

    // Apply SwiGLU adjustment
    if (config.activation_type == MLPType::kSwiGLU) {
        ffn_hidden = int(2 * ffn_hidden) / 3; // SwiGLU intermediate
    }

    // Apply multiplier
    if (config.ffn_dim_multiplier.has_value()) {
        ffn_hidden
            = static_cast<int64_t>(std::llround(static_cast<double>(ffn_hidden) * config.ffn_dim_multiplier.value()));
    }

    // Round up to multiple_of
    int64_t before_round = ffn_hidden;
    ffn_hidden = (ffn_hidden + config.multiple_of - 1) / config.multiple_of * config.multiple_of;

    // Build c_fc (input projection)
    ModuleSpec c_fc_spec(typeid(parallel::ColumnParallelLinear));
    c_fc_spec.with_param(kInFeatures, static_cast<int>(config.n_embd))
        .with_param(kOutFeatures, static_cast<int>(ffn_hidden))
        .with_param(kBias, config.use_bias);
    spec.with_submodule(MLP::kCFcLayerName, c_fc_spec);

    // Build activation based on config
    switch (config.activation_type) {
    case MLPType::kGELU: {
        spec.with_submodule(MLP::kGeluLayerName, ModuleSpec(typeid(NewGELU)));
        break;
    }
    case MLPType::kSwiGLU: {
        // Add second projection for SwiGLU
        ModuleSpec c_fc2_spec(typeid(parallel::ColumnParallelLinear));
        c_fc2_spec.with_param(kInFeatures, static_cast<int>(config.n_embd))
            .with_param(kOutFeatures, static_cast<int>(ffn_hidden))
            .with_param(kBias, config.use_bias);
        spec.with_submodule(MLP::kCFc2LayerName, c_fc2_spec);

        spec.with_submodule(MLP::kSiluLayerName, ModuleSpec(typeid(SwiGLU)));
        break;
    }
    default:
        LOG(FATAL) << "Unsupported MLP type";
    }

    // Build c_proj (output projection)
    ModuleSpec c_proj_spec(typeid(parallel::RowParallelLinear));
    c_proj_spec.with_param(kInFeatures, static_cast<int>(ffn_hidden))
        .with_param(kOutFeatures, static_cast<int>(config.n_embd))
        .with_param(kBias, config.use_bias);
    spec.with_submodule(MLP::kCProjLayerName, c_proj_spec);

    return spec;
}

ModuleSpec BuildTransformerBlockSpec(const TransformerConfig &config) {
    ModuleSpec spec(typeid(TransformerLayer));

    // LayerNorm 1 (before attention)
    spec.with_submodule(TransformerLayer::kLn1LayerName, BuildNormSpec(config));

    // CausalSelfAttention
    spec.with_submodule(TransformerLayer::kAttnLayerName, BuildAttentionSpec(config));

    // LayerNorm 2 (before MLP)
    spec.with_submodule(TransformerLayer::kLn2LayerName, BuildNormSpec(config));

    // MLP
    spec.with_submodule(TransformerLayer::kMlpLayerName, BuildMLPSpec(config));

    return spec;
}

ModuleSpec BuildVocabEmbeddingSpec(const TransformerConfig &config) {
    ModuleSpec spec(typeid(parallel::VocabParallelEmbedding));
    spec.with_param(kNumEmbeddings, static_cast<int>(config.vocab_size))
        .with_param(kEmbeddingDim, static_cast<int>(config.n_embd));
    return spec;
}

ModuleSpec BuildPositionEmbeddingSpec(int64_t num_embeddings, int64_t embedding_dim) {
    ModuleSpec spec(typeid(Embedding));
    spec.with_param(kNumEmbeddings, static_cast<int>(num_embeddings))
        .with_param(kEmbeddingDim, static_cast<int>(embedding_dim));
    return spec;
}

ModuleSpec BuildOutputProjSpec(const TransformerConfig &config, int64_t output_size, bool use_bias) {
    ModuleSpec spec(typeid(parallel::ColumnParallelLinear));
    spec.with_param(kInFeatures, static_cast<int>(config.n_embd))
        .with_param(kOutFeatures, static_cast<int>(output_size))
        .with_param(kBias, use_bias);
    return spec;
}

} // namespace infini_train::nn

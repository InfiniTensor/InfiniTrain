#include "infini_train/include/nn/modules/transformer/layer_specs.h"

#include <cmath>

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/spec_utils.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"

namespace infini_train::nn {
ModuleSpec BuildNormSpec(const TransformerConfig &config) {
    ModuleSpec spec;
    switch (config.norm_type) {
    case NormType::kLayerNorm:
        spec = ModuleSpec(typeid(LayerNorm));
        spec.WithParam(kDim, static_cast<int>(config.n_embd))
            .WithParam(kEps, config.norm_eps)
            .WithParam(kNormalizedShape, std::vector<int64_t>{config.n_embd});
        break;
    case NormType::kRMSNorm:
        spec = ModuleSpec(typeid(RMSNorm));
        spec.WithParam(kDim, static_cast<int>(config.n_embd)).WithParam(kEps, config.norm_eps);
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
    if (config.UseGQA()) {
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
    c_attn_spec.WithParam(kInFeatures, static_cast<int>(config.n_embd))
        .WithParam(kOutFeatures, static_cast<int>(qkv_out))
        .WithParam(kBias, config.use_bias);
    spec.WithSubmodule(CausalSelfAttention::kCAttnLayerName, c_attn_spec);

    // Build c_proj (output projection)
    ModuleSpec c_proj_spec(typeid(parallel::RowParallelLinear));
    c_proj_spec.WithParam(kInFeatures, static_cast<int>(config.n_embd))
        .WithParam(kOutFeatures, static_cast<int>(config.n_embd))
        .WithParam(kBias, config.use_bias);
    spec.WithSubmodule(CausalSelfAttention::kCProjLayerName, c_proj_spec);

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
    c_fc_spec.WithParam(kInFeatures, static_cast<int>(config.n_embd))
        .WithParam(kOutFeatures, static_cast<int>(ffn_hidden))
        .WithParam(kBias, config.use_bias);
    spec.WithSubmodule(MLP::kCFcLayerName, c_fc_spec);

    // Build activation based on config
    switch (config.activation_type) {
    case MLPType::kGELU: {
        spec.WithSubmodule(MLP::kGeluLayerName, ModuleSpec(typeid(NewGELU)));
        break;
    }
    case MLPType::kSwiGLU: {
        // Add second projection for SwiGLU
        ModuleSpec c_fc2_spec(typeid(parallel::ColumnParallelLinear));
        c_fc2_spec.WithParam(kInFeatures, static_cast<int>(config.n_embd))
            .WithParam(kOutFeatures, static_cast<int>(ffn_hidden))
            .WithParam(kBias, config.use_bias);
        spec.WithSubmodule(MLP::kCFc2LayerName, c_fc2_spec);

        spec.WithSubmodule(MLP::kSiluLayerName, ModuleSpec(typeid(SwiGLU)));
        break;
    }
    default:
        LOG(FATAL) << "Unsupported MLP type";
    }

    // Build c_proj (output projection)
    ModuleSpec c_proj_spec(typeid(parallel::RowParallelLinear));
    c_proj_spec.WithParam(kInFeatures, static_cast<int>(ffn_hidden))
        .WithParam(kOutFeatures, static_cast<int>(config.n_embd))
        .WithParam(kBias, config.use_bias);
    spec.WithSubmodule(MLP::kCProjLayerName, c_proj_spec);

    return spec;
}

ModuleSpec BuildVocabEmbeddingSpec(const TransformerConfig &config) {
    ModuleSpec spec(typeid(parallel::VocabParallelEmbedding));
    spec.WithParam(kNumEmbeddings, static_cast<int>(config.vocab_size))
        .WithParam(kEmbeddingDim, static_cast<int>(config.n_embd));
    return spec;
}

ModuleSpec BuildPositionEmbeddingSpec(int64_t num_embeddings, int64_t embedding_dim) {
    ModuleSpec spec(typeid(Embedding));
    spec.WithParam(kNumEmbeddings, static_cast<int>(num_embeddings))
        .WithParam(kEmbeddingDim, static_cast<int>(embedding_dim));
    return spec;
}

ModuleSpec BuildOutputProjSpec(const TransformerConfig &config, int64_t output_size, bool use_bias) {
    ModuleSpec spec(typeid(parallel::ColumnParallelLinear));
    spec.WithParam(kInFeatures, static_cast<int>(config.n_embd))
        .WithParam(kOutFeatures, static_cast<int>(output_size))
        .WithParam(kBias, use_bias);
    return spec;
}

ModuleSpec BuildFirstStageSpec(const TransformerConfig &config) {
    ModuleSpec spec(typeid(TransformerFirstStage));
    spec.WithSubmodule(TransformerFirstStage::kWTELayerName, BuildVocabEmbeddingSpec(config));

    if (config.activation_type == MLPType::kGELU) {
        spec.WithSubmodule(TransformerFirstStage::kWPELayerName,
                           BuildPositionEmbeddingSpec(config.block_size, config.n_embd));
    }

    return spec;
}

ModuleSpec BuildTransformerLayerSpec(const TransformerConfig &config) {
    ModuleSpec spec(typeid(TransformerLayer));

    // LayerNorm 1 (before attention)
    spec.WithSubmodule(TransformerLayer::kLn1LayerName, BuildNormSpec(config));

    // CausalSelfAttention
    spec.WithSubmodule(TransformerLayer::kAttnLayerName, BuildAttentionSpec(config));

    // LayerNorm 2 (before MLP)
    spec.WithSubmodule(TransformerLayer::kLn2LayerName, BuildNormSpec(config));

    // MLP
    spec.WithSubmodule(TransformerLayer::kMlpLayerName, BuildMLPSpec(config));

    return spec;
}

ModuleSpec BuildLastStageSpec(const TransformerConfig &config) {
    ModuleSpec spec(typeid(TransformerLastStage));
    spec.WithSubmodule(TransformerLastStage::kLnFLayerName, BuildNormSpec(config))
        .WithSubmodule(TransformerLastStage::kLMHeadLayerName, BuildOutputProjSpec(config, config.vocab_size, false));

    return spec;
}

ModuleSpec BuildTransformerSpec(const TransformerConfig &config, ModuleSpec first_stage, ModuleSpec layer,
                                ModuleSpec last_stage) {
    ModuleSpec spec(typeid(TransformerModel));
    spec.WithSubmodule(TransformerFirstStage::kType, first_stage)
        .WithSubmodule(TransformerLayer::kType, layer)
        .WithSubmodule(TransformerLastStage::kType, last_stage);

    return spec;
}
} // namespace infini_train::nn

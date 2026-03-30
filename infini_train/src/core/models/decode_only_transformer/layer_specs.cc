#include "infini_train/include/core/models/decode_only_transformer/layer_specs.h"

#include <cmath>
#include <memory>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_builders.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_model.h"

namespace infini_train::nn {

ModuleSpec BuildGPT2Spec(const TransformerConfig &config) {
    // Configure for GPT2 architecture
    TransformerConfig gpt2_config = config;
    ModuleSpec spec;

    // ===== First Stage =====
    ModuleSpec first_stage;
    first_stage.WithSubmodule(TransformerFirstStage::kWTELayerName, BuildVocabEmbeddingSpec(gpt2_config))
        .WithSubmodule(TransformerFirstStage::kWPELayerName,
                       BuildPositionEmbeddingSpec(gpt2_config.block_size, gpt2_config.n_embd));
    spec.WithSubmodule(TransformerFirstStage::kType, first_stage);

    // ===== Transformer Layer =====
    ModuleSpec block = BuildTransformerLayerSpec(gpt2_config);
    spec.WithSubmodule(TransformerLayer::kType, block);

    // ===== Last Stage =====
    ModuleSpec last_stage;
    last_stage.WithSubmodule(TransformerLastStage::kLnFLayerName, BuildNormSpec(gpt2_config))
        .WithSubmodule(TransformerLastStage::kLMHeadLayerName,
                       BuildOutputProjSpec(gpt2_config, gpt2_config.vocab_size, false));
    spec.WithSubmodule(TransformerLastStage::kType, last_stage);

    return spec;
}

ModuleSpec BuildLLaMA3Spec(const TransformerConfig &config) {
    // Configure for LLaMA3 architecture
    TransformerConfig llama3_config = config;
    ModuleSpec spec;

    // ===== First Stage =====
    ModuleSpec first_stage;
    // LLaMA3 only has token embedding, no position embedding (uses RoPE)
    first_stage.WithSubmodule(TransformerFirstStage::kWTELayerName, BuildVocabEmbeddingSpec(llama3_config));
    spec.WithSubmodule(TransformerFirstStage::kType, first_stage);

    // ===== Transformer Layer =====
    ModuleSpec block = BuildTransformerLayerSpec(llama3_config);
    spec.WithSubmodule(TransformerLayer::kType, block);

    // ===== Last Stage =====
    ModuleSpec last_stage;
    last_stage.WithSubmodule(TransformerLastStage::kLnFLayerName, BuildNormSpec(llama3_config))
        .WithSubmodule(TransformerLastStage::kLMHeadLayerName,
                       BuildOutputProjSpec(llama3_config, llama3_config.vocab_size, false));
    spec.WithSubmodule(TransformerLastStage::kType, last_stage);

    return spec;
}

} // namespace infini_train::nn

#include "infini_train/include/core/models/decode_only_transformer/layer_specs.h"

#include <cmath>
#include <memory>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_builders.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_layer.h"

namespace infini_train::nn {

ModuleSpec BuildGPT2Spec(const TransformerConfig &config) {
    // Configure for GPT2 architecture
    TransformerConfig gpt2_config = config;
    ModuleSpec spec;

    // ===== First Stage =====
    ModuleSpec first_stage;
    first_stage.with_submodule(TransformerFirstStage::kWTELayerName, BuildVocabEmbeddingSpec(gpt2_config))
        .with_submodule(TransformerFirstStage::kWPELayerName,
                        BuildPositionEmbeddingSpec(gpt2_config.block_size, gpt2_config.n_embd));
    spec.with_submodule(TransformerFirstStage::kType, first_stage);

    // ===== Transformer Block =====
    ModuleSpec block = BuildTransformerBlockSpec(gpt2_config);
    spec.with_submodule(TransformerBlock::kType, block);

    // ===== Last Stage =====
    ModuleSpec last_stage;
    last_stage.with_submodule(TransformerLastStage::kLnFLayerName, BuildNormSpec(gpt2_config))
        .with_submodule(TransformerLastStage::kLMHeadLayerName,
                        BuildOutputProjSpec(gpt2_config, gpt2_config.vocab_size, false));
    spec.with_submodule(TransformerLastStage::kType, last_stage);

    return spec;
}

ModuleSpec BuildLLaMA3Spec(const TransformerConfig &config) {
    // Configure for LLaMA3 architecture
    TransformerConfig llama3_config = config;
    ModuleSpec spec;

    // ===== First Stage =====
    ModuleSpec first_stage;
    // LLaMA3 only has token embedding, no position embedding (uses RoPE)
    first_stage.with_submodule(TransformerFirstStage::kWTELayerName, BuildVocabEmbeddingSpec(llama3_config));
    spec.with_submodule(TransformerFirstStage::kType, first_stage);

    // ===== Transformer Block =====
    ModuleSpec block = BuildTransformerBlockSpec(llama3_config);
    spec.with_submodule(TransformerBlock::kType, block);

    // ===== Last Stage =====
    ModuleSpec last_stage;
    last_stage.with_submodule(TransformerLastStage::kLnFLayerName, BuildNormSpec(llama3_config))
        .with_submodule(TransformerLastStage::kLMHeadLayerName,
                        BuildOutputProjSpec(llama3_config, llama3_config.vocab_size, false));
    spec.with_submodule(TransformerLastStage::kType, last_stage);

    return spec;
}

} // namespace infini_train::nn

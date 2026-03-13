#pragma once

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn {

/**
 * @brief Build complete GPT2 model spec
 *
 * GPT2 architecture:
 * - First stage: VocabParallelEmbedding (wte) + Embedding (wpe)
 * - Blocks: LayerNorm + CausalSelfAttention + LayerNorm + MLP (GELU)
 * - Last stage: LayerNorm (ln_f) + ColumnParallelLinear (lm_head)
 */
ModuleSpec BuildGPT2Spec(const TransformerConfig &config);

/**
 * @brief Build complete LLaMA3 model spec
 *
 * LLaMA3 architecture:
 * - First stage: VocabParallelEmbedding (wte) - no position embedding (uses RoPE)
 * - Blocks: RMSNorm + CausalSelfAttention (with RoPE, GQA) + RMSNorm + MLP (SwiGLU)
 * - Last stage: RMSNorm (ln_f) + ColumnParallelLinear (lm_head)
 */
ModuleSpec BuildLLaMA3Spec(const TransformerConfig &config);

} // namespace infini_train::nn

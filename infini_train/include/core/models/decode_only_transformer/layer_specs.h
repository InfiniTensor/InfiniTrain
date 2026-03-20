#pragma once

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn {
// Build GPT2 model spec: LayerNorm + GELU + standard attention
ModuleSpec BuildGPT2Spec(const TransformerConfig &config);

// Build LLaMA3 model spec: RMSNorm + SwiGLU + RoPE + GQA
ModuleSpec BuildLLaMA3Spec(const TransformerConfig &config);
} // namespace infini_train::nn

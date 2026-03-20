#pragma once

#include <cstdint>
#include <memory>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn {

// Embedding
inline constexpr char kNumEmbeddings[] = "num_embeddings";
inline constexpr char kEmbeddingDim[] = "embedding_dim";

// Normalization
inline constexpr char kNormalizedShape[] = "normalized_shape";
inline constexpr char kDim[] = "dim";
inline constexpr char kEps[] = "eps";

// Linear
inline constexpr char kInFeatures[] = "in_features";
inline constexpr char kOutFeatures[] = "out_features";
inline constexpr char kBias[] = "bias";

// Attention
inline constexpr char kNumHeads[] = "num_heads";
inline constexpr char kNumKVHeads[] = "num_kv_heads";

// Build LayerNorm or RMSNorm spec based on config
ModuleSpec BuildNormSpec(const TransformerConfig &config);

// Build CausalSelfAttention spec
ModuleSpec BuildAttentionSpec(const TransformerConfig &config);

// Build MLP spec (supports GELU and SwiGLU)
ModuleSpec BuildMLPSpec(const TransformerConfig &config);

// Build TransformerBlock spec
ModuleSpec BuildTransformerBlockSpec(const TransformerConfig &config);

// Build VocabParallelEmbedding spec for token embeddings
ModuleSpec BuildVocabEmbeddingSpec(const TransformerConfig &config);

// Build Embedding spec for position embeddings
ModuleSpec BuildPositionEmbeddingSpec(int64_t num_embeddings, int64_t embedding_dim);

// Build ColumnParallelLinear spec for output projection (lm_head)
ModuleSpec BuildOutputProjSpec(const TransformerConfig &config, int64_t output_size, bool use_bias);

} // namespace infini_train::nn

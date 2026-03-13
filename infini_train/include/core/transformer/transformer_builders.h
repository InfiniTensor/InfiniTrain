#pragma once

#include <cstdint>
#include <memory>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn {

/**
 * @brief Build LayerNorm or RMSNorm spec based on config
 * @param config Transformer config
 */
ModuleSpec BuildNormSpec(const TransformerConfig &config);

/**
 * @brief Build CausalSelfAttention spec
 * @param config Transformer config
 */
ModuleSpec BuildAttentionSpec(const TransformerConfig &config);

/**
 * @brief Build MLP spec based on config (supports GELU and SwiGLU)
 * @param config Transformer config
 * @param ffn_hidden_opt Optional hidden dimension (computed from config if not provided)
 */
ModuleSpec BuildMLPSpec(const TransformerConfig &config);

/**
 * @brief Build TransformerBlock spec based on config
 * @param config Transformer config
 */
ModuleSpec BuildTransformerBlockSpec(const TransformerConfig &config);

/**
 * @brief Build VocabParallelEmbedding spec
 * @param config Transformer config
 */
ModuleSpec BuildVocabEmbeddingSpec(const TransformerConfig &config);

/**
 * @brief Build standard Embedding spec (for position embeddings)
 * @param num_embeddings Number of embeddings (e.g., block_size)
 * @param embedding_dim Embedding dimension (e.g., n_embd)
 */
ModuleSpec BuildPositionEmbeddingSpec(int64_t num_embeddings, int64_t embedding_dim);

/**
 * @brief Build ColumnParallelLinear spec for output projection (lm_head)
 * @param config Transformer config
 * @param output_size Output dimension (usually vocab_size)
 * @param use_bias Whether to use bias
 */
ModuleSpec BuildOutputProjSpec(const TransformerConfig &config, int64_t output_size, bool use_bias);

} // namespace infini_train::nn

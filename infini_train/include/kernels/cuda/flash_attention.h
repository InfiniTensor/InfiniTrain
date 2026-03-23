#pragma once

#include <memory>

namespace infini_train {
class Tensor;
}

namespace infini_train::kernels::cuda {

/**
 * FlashAttention Forward Output Structure
 *
 * This structure holds the output tensors from FlashAttention forward pass.
 *
 * Args:
 *   output: Output tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
 *   logsumexp: Logsumexp tensor for backward pass [batch_size, num_heads, seq_len_q]
 *   dropout_seed: Dropout seed for backward pass [1]
 */
struct FlashAttentionForwardOutput {
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> logsumexp;
    std::shared_ptr<Tensor> dropout_seed;
};

} // namespace infini_train::kernels::cuda

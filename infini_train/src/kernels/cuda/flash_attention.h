#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::kernels::cuda {

/**
 * @brief FlashAttention Forward Kernel (Placeholder)
 *
 * @param q Query tensor (B, T, H, D)
 * @param k Key tensor (B, T, H, D)
 * @param v Value tensor (B, T, H, D)
 * @param output Output tensor (B, T, H, D)
 * @param softmax_lse LogSumExp of softmax, for backward (B, H, T)
 * @param dropout_p Dropout probability
 * @param softmax_scale Scaling factor for QK^T
 * @param is_causal Whether to apply causal mask
 * @param device Device info
 */
void FlashAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output, Tensor &softmax_lse,
                           float dropout_p, float softmax_scale, bool is_causal, const Device &device);

} // namespace infini_train::kernels::cuda

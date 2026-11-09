#pragma once

#include <cstdint>
#include <memory>
#include <tuple>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
// RoPE helper method
std::shared_ptr<Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end,
                                           float theta = 10000.0f,
                                           bool use_scaled = false,
                                           Device device = Device());

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ApplyRotaryEmbedding(const std::shared_ptr<Tensor> &xq,
                     const std::shared_ptr<Tensor> &xk,
                     const std::shared_ptr<Tensor> &freqs_cis,
                     bool rotary_interleaved);

std::shared_ptr<Tensor> RepeatKVHeads(const std::shared_ptr<Tensor> &x,
                                      int64_t n_rep);

std::shared_ptr<Tensor> ScaledDotProductAttention(
    const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v, int64_t num_gqa_groups,
    const std::shared_ptr<Tensor> &mask);
}  // namespace infini_train

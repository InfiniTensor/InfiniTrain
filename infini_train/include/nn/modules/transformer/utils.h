#pragma once

#include <cstdint>

#include "infini_train/include/tensor.h"

namespace infini_train {
// RoPE helper method
std::shared_ptr<Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end, float theta = 10000.0f, bool use_scaled = false,
                                           Device device = Device());
} // namespace infini_train

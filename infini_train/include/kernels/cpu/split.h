#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::vector<std::shared_ptr<Tensor>> SplitForward(const std::shared_ptr<Tensor> &input, int64_t split_size, int dim);
std::shared_ptr<Tensor> SplitBackward(const std::vector<int64_t> &input_dims, int64_t split_size, int dim,
                                      const std::vector<std::shared_ptr<Tensor>> &grad_outputs);
} // namespace infini_train::kernels::cpu

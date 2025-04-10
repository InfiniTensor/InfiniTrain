#pragma once

#include <memory>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> NoOpForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &dims);
std::shared_ptr<Tensor> NoOpBackward(const std::vector<int64_t> &dims, const std::shared_ptr<Tensor> &grad_output);
} // namespace infini_train::kernels::cpu

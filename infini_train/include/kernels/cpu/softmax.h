#pragma once

#include <cstdint>
#include <memory>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> SoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim);
std::shared_ptr<Tensor> SoftmaxBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &output, int64_t dim);
} // namespace infini_train::kernels::cpu

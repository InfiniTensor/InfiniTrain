#pragma once

#include "infini_train/include/tensor.h"
#include <cstdint>
#include <memory>

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> SoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim);
std::shared_ptr<Tensor> SoftmaxBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &output, int64_t dim);
} // namespace infini_train::kernels::cuda

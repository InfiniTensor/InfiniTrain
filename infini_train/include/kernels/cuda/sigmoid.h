#pragma once

#include <memory>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> SigmoidForward(const std::shared_ptr<Tensor> &input);
std::shared_ptr<Tensor> SigmoidBackward(const std::shared_ptr<Tensor> &input,
                                        const std::shared_ptr<Tensor> &grad_output);
} // namespace infini_train::kernels::cuda

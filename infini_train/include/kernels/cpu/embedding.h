#pragma once

#include <memory>
#include <tuple>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight);
std::shared_ptr<Tensor> EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                          const std::shared_ptr<Tensor> &grad_output);
} // namespace infini_train::kernels::cpu

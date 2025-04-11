#pragma once

#include <memory>
#include <tuple>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight);
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                  const std::shared_ptr<Tensor> &grad_output);
} // namespace infini_train::kernels::cuda

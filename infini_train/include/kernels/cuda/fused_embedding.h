#pragma once

#include <memory>
#include <tuple>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> FusedEmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &wte,
                                         const std::shared_ptr<Tensor> &wpe);
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FusedEmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &wte,
                  const std::shared_ptr<Tensor> &wpe, const std::shared_ptr<Tensor> &grad_output);
} // namespace infini_train::kernels::cuda

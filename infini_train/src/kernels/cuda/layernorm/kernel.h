#pragma once

#include <memory>
#include <tuple>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
StableLayerNormForward32(const std::shared_ptr<Tensor>& input,
                         const std::shared_ptr<Tensor>& weight,
                         const std::shared_ptr<Tensor>& bias, float eps);

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
StableLayerNormBackward32(const std::shared_ptr<Tensor>& input,
                          const std::shared_ptr<Tensor>& weight,
                          const std::shared_ptr<Tensor>& bias,
                          const std::shared_ptr<Tensor>& rstd,
                          const std::shared_ptr<Tensor>& grad_output);

}  // namespace infini_train::kernels::cuda

#pragma once

#include <memory>
#include <utility>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> NewGELUForward(
    const std::shared_ptr<Tensor>& input);

std::shared_ptr<Tensor> NewGELUBackward(
    const std::shared_ptr<Tensor>& input,
    const std::shared_ptr<Tensor>& saved_tanh,
    const std::shared_ptr<Tensor>& gradient);

}  // namespace infini_train::kernels::cuda

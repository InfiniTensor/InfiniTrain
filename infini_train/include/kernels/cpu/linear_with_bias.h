#pragma once

#include <memory>
#include <tuple>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> LinearWithBiasForward(const std::shared_ptr<Tensor> &input,
                                              const std::shared_ptr<Tensor> &weight,
                                              const std::shared_ptr<Tensor> &bias);
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearWithBiasBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                       int64_t out_features, const std::shared_ptr<Tensor> &grad_output);
} // namespace infini_train::kernels::cpu

#pragma once

#include <cstdint>
#include <memory>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> TrilForward(const std::shared_ptr<Tensor> &input, int64_t diagonal);
std::shared_ptr<Tensor> TrilBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal);

std::shared_ptr<Tensor> TransposeForward(const std::shared_ptr<Tensor> &input, int64_t dim0, int64_t dim1);
std::shared_ptr<Tensor> TransposeBackward(const std::shared_ptr<Tensor> &grad_output, int64_t dim0, int64_t dim1);

std::shared_ptr<Tensor> MaskForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &mask,
                                    float value);
std::shared_ptr<Tensor> MaskBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask);
} // namespace infini_train::kernels::cpu

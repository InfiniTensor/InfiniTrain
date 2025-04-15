#pragma once

#include <memory>
#include <utility>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> TanhForward(const std::shared_ptr<Tensor> &input);
std::shared_ptr<Tensor> TanhBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &output);

std::shared_ptr<Tensor> PowForward(const std::shared_ptr<Tensor> &input, float exponent);
std::shared_ptr<Tensor> PowBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &output,
                                    float exponent);

std::shared_ptr<Tensor> EqualsScalarForward(const std::shared_ptr<Tensor> &a, float scalar);

std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims);

std::shared_ptr<Tensor> AddScalarForward(const std::shared_ptr<Tensor> &a, float scalar);
std::shared_ptr<Tensor> AddScalarBackward(const std::shared_ptr<Tensor> &grad_output);

std::shared_ptr<Tensor> MulForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b);
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MulBackward(const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b,
                                                                        const std::shared_ptr<Tensor> &grad_output);

std::shared_ptr<Tensor> MulScalarForward(const std::shared_ptr<Tensor> &a, float scalar);
std::shared_ptr<Tensor> MulScalarBackward(const std::shared_ptr<Tensor> &grad_output, float scalar);
} // namespace infini_train::kernels::cpu

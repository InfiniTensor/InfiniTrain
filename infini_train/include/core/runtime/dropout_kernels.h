#pragma once

#include <optional>

#include "infini_train/include/generator.h"

namespace infini_train {

class Tensor;

void dropout_forward_kernel(Tensor &output, Tensor &mask, const Tensor &input, double p,
                            const std::optional<Generator> &generator);

void dropout_backward_kernel(Tensor &grad_input, const Tensor &grad_output, const Tensor &mask, double p);

} // namespace infini_train

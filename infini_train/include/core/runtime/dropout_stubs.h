#pragma once

#include <optional>

#include "infini_train/include/core/runtime/dispatch_stub.h"
#include "infini_train/include/generator.h"

namespace infini_train {

class Tensor;

DECLARE_DISPATCH(void (*)(Tensor &output, Tensor &mask, const Tensor &input, double p,
                          const std::optional<Generator> &generator),
                 dropout_forward_stub);

DECLARE_DISPATCH(void (*)(Tensor &grad_input, const Tensor &grad_output, const Tensor &mask, double p),
                 dropout_backward_stub);

} // namespace infini_train

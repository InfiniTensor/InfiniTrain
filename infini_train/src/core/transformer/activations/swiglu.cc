#include "infini_train/include/core/transformer/activations/swiglu.h"

#include <cmath>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

std::vector<std::shared_ptr<infini_train::Tensor>>
SwiGLU::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    return {x[0] * nn::function::Sigmoid(x[0])};
}

} // namespace infini_train::nn
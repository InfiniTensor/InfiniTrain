#include "infini_train/include/core/transformer/activations/gelu.h"

#include <cmath>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

std::vector<std::shared_ptr<infini_train::Tensor>>
NewGELU::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto &input = x[0];
    return {0.5 * input
            * (1.0 + nn::function::Tanh(std::sqrt(2.0 / M_PI) * (input + 0.044715 * nn::function::Pow(input, 3.0))))};
}

} // namespace infini_train::nn
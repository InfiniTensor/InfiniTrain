#include "infini_train/include/nn/modules/activations.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
std::vector<std::shared_ptr<Tensor>> Sigmoid::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::Sigmoid>()->Apply(input_tensors);
}

std::vector<std::shared_ptr<Tensor>> NewGELU::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto &input = x[0];
    return {0.5 * input
            * (1.0 + function::Tanh(std::sqrt(2.0 / M_PI) * (input + 0.044715 * function::Pow(input, 3.0))))};
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    return {x[0] * function::Sigmoid(x[0])};
}
} // namespace infini_train::nn

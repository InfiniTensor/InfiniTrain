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
    return std::make_shared<autograd::GELU>()->Apply(x);
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    return std::make_shared<autograd::SiLU>()->Apply(x);
}
} // namespace infini_train::nn

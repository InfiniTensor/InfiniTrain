#include "infini_train/include/nn/modules/loss.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/loss.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
std::vector<std::shared_ptr<Tensor>>
CrossEntropyLoss::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::CrossEntropy>()->Apply(input_tensors);
}
} // namespace infini_train::nn

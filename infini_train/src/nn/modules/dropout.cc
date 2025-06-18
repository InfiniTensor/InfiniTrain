#include "infini_train/include/nn/modules/dropout.h"

#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/dropout.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
std::vector<std::shared_ptr<Tensor>> Dropout::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::Dropout>(p_, training_, inplace_)->Apply({input_tensors[0]});
}
} // namespace infini_train::nn
#include "infini_train/include/nn/modules/linear.h"

#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Linear::Linear(int64_t in_features, int64_t out_features, bool bias, Device device)
    : CloneableModule(kType), bias_(bias) {
    device_ = device ? device : Device();

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features, in_features}, DataType::kFLOAT32, device_)
              ->RequiresGrad();
    if (bias) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32, device_)->RequiresGrad();
    }
    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> Linear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::Linear>()->Apply(
        bias_ ? std::vector<std::shared_ptr<Tensor>>{input_tensors[0], parameters_[kParamWeightName],
                                                     parameters_[kParamBiasName]}
              : std::vector<std::shared_ptr<Tensor>>{input_tensors[0], parameters_[kParamWeightName]});
}

void Linear::ResetParameters() {
    init::KaimingUniform(parameters_[kParamWeightName], sqrt(5.0f));
    if (bias_) {
        const auto [fan_in, _] = init::CalculateFanInAndFanOut(parameters_[kParamWeightName]);
        const float bound = fan_in > 0 ? 1.0 / sqrt(fan_in) : 0.0;
        init::Uniform(parameters_[kParamBiasName], -bound, bound);
    }
}
} // namespace infini_train::nn

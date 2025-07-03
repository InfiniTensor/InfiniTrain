#include "infini_train/include/nn/modules/linear.h"

#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Linear::Linear(int64_t in_features, int64_t out_features, bool bias, Device device) : Module(kType), bias_(bias) {
    device_ = device;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features, in_features}, DataType::kFLOAT32, device)
              ->RequiresGrad();
    if (bias) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32, device)->RequiresGrad();
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
    parameters_[kParamWeightName]->Fill(0.01f);
    if (bias_) {
        parameters_[kParamBiasName]->Fill(0.01f);
    }
}
} // namespace infini_train::nn

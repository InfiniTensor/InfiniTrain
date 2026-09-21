#include "infini_train/include/nn/modules/normalization.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
LayerNorm::LayerNorm(const std::vector<int64_t> &normalized_shape, float eps, Device device)
    : CloneableModule(kType), eps_(eps) {
    device_ = device;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(normalized_shape, DataType::kFLOAT32, device_)->RequiresGrad();
    parameters_[kParamBiasName]
        = std::make_shared<Tensor>(normalized_shape, DataType::kFLOAT32, device_)->RequiresGrad();
    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto outputs = std::make_shared<autograd::LayerNorm>(eps_)->Apply(
        {input_tensors[0], parameters_[kParamWeightName], parameters_[kParamBiasName]});
    return {outputs[0]};
}

void LayerNorm::ResetParameters() {
    init::Ones(parameters_[kParamWeightName]);
    init::Zeros(parameters_[kParamBiasName]);
}

RMSNorm::RMSNorm(int64_t dim, float eps, Device device) : CloneableModule(kType), eps_(eps) {
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{dim}, DataType::kFLOAT32, device)->RequiresGrad();
    nn::init::Ones(parameters_[kParamWeightName]);
}

std::vector<std::shared_ptr<Tensor>> RMSNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto outputs = std::make_shared<autograd::RMSNorm>(eps_)->Apply({x[0], parameters_[kParamWeightName]});
    return {outputs[0]};
}
} // namespace infini_train::nn

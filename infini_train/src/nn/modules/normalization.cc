#include "infini_train/include/nn/modules/normalization.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
LayerNorm::LayerNorm(const std::vector<int64_t> &normalized_shape, float eps, Device device) : eps_(eps) {
    device_ = device;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(normalized_shape, DataType::kFLOAT32, device)->RequiresGrad();
    parameters_[kParamBiasName]
        = std::make_shared<Tensor>(normalized_shape, DataType::kFLOAT32, device)->RequiresGrad();
    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::LayerNorm>(eps_)->Apply(
        {input_tensors[0], parameters_[kParamWeightName], parameters_[kParamBiasName]});
}

void LayerNorm::ResetParameters() {
    init::Ones(parameters_[kParamWeightName]);
    init::Zeros(parameters_[kParamBiasName]);
}
} // namespace infini_train::nn

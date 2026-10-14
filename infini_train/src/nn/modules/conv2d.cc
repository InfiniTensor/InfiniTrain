#include "infini_train/include/nn/modules/conv2d.h"

#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/conv2d.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
               int64_t stride, int64_t padding, bool bias, Device device)
    : CloneableModule(kType), in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding), bias_(bias) {
    device_ = device;

    parameters_[kParamWeightName] = std::make_shared<Tensor>(
        std::vector<int64_t>{out_channels, in_channels, kernel_size, kernel_size},
        DataType::kFLOAT32, device_)->RequiresGrad();

    if (bias) {
        parameters_[kParamBiasName] = std::make_shared<Tensor>(
            std::vector<int64_t>{out_channels}, DataType::kFLOAT32, device_)->RequiresGrad();
    }
    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto op = std::make_shared<autograd::Conv2d>(stride_, padding_);
    if (bias_) {
        return op->Apply({input_tensors[0], parameters_[kParamWeightName], parameters_[kParamBiasName]});
    } else {
        return op->Apply({input_tensors[0], parameters_[kParamWeightName]});
    }
}

void Conv2d::ResetParameters() {
    init::KaimingUniform(parameters_[kParamWeightName], sqrt(5.0f));
    if (bias_) {
        const auto [fan_in, _] = init::CalculateFanInAndFanOut(parameters_[kParamWeightName]);
        const float bound = fan_in > 0 ? 1.0f / sqrt(fan_in) : 0.0f;
        init::Uniform(parameters_[kParamBiasName], -bound, bound);
    }
}
} // namespace infini_train::nn
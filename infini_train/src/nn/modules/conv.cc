#include "infini_train/include/nn/modules/conv.h"

#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding,
               bool bias, Device device)
    : CloneableModule(kType), stride_(stride), padding_(padding), bias_(bias) {
    device_ = device;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_channels, in_channels, kernel_size, kernel_size},
                                   DataType::kFLOAT32, device_)
              ->RequiresGrad();
    if (bias) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32, device_)->RequiresGrad();
    }
    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::Conv2d>(stride_, padding_)
        ->Apply(bias_ ? std::vector<std::shared_ptr<Tensor>>{input_tensors[0], parameters_[kParamWeightName],
                                                             parameters_[kParamBiasName]}
                      : std::vector<std::shared_ptr<Tensor>>{input_tensors[0], parameters_[kParamWeightName]});
}

void Conv2d::ResetParameters() {
    init::KaimingUniform(parameters_[kParamWeightName], sqrt(5.0f));
    if (bias_) {
        const auto [fan_in, _] = init::CalculateFanInAndFanOut(parameters_[kParamWeightName]);
        const float bound = fan_in > 0 ? 1.0 / sqrt(fan_in) : 0.0;
        init::Uniform(parameters_[kParamBiasName], -bound, bound);
    }
}
} // namespace infini_train::nn

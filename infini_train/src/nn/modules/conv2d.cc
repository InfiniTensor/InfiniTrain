#include "infini_train/include/nn/modules/conv2d.h"

#include <cmath>

#include "glog/logging.h"

#include "infini_train/include/autograd/conv2d.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding,
               bool bias, Device device)
    : CloneableModule(kType), stride_(stride), padding_(padding), bias_(bias) {
    CHECK_GT(in_channels, 0);
    CHECK_GT(out_channels, 0);
    CHECK_GT(kernel_size, 0);
    CHECK_GT(stride, 0);
    CHECK_GE(padding, 0);
    device_ = device;
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_channels, in_channels, kernel_size, kernel_size},
                                   DataType::kFLOAT32, device_)
              ->RequiresGrad();
    if (bias_) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32, device_)->RequiresGrad();
    }
    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    return std::make_shared<autograd::Conv2d>(stride_, padding_)
        ->Apply(bias_ ? std::vector<std::shared_ptr<Tensor>>{input_tensors[0], parameters_[kParamWeightName],
                                                             parameters_[kParamBiasName]}
                      : std::vector<std::shared_ptr<Tensor>>{input_tensors[0], parameters_[kParamWeightName]});
}

void Conv2d::ResetParameters() {
    init::KaimingUniform(parameters_[kParamWeightName], 0.0f, init::KaimingMode::kFanIn, init::NonLinearityType::kReLU);
    if (bias_) {
        const auto [fan_in, fan_out] = init::CalculateFanInAndFanOut(parameters_[kParamWeightName]);
        static_cast<void>(fan_out);
        const float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        init::Uniform(parameters_[kParamBiasName], -bound, bound);
    }
}

} // namespace infini_train::nn

#include "infini_train/include/nn/modules/convolution.h"

#include <cmath>

#include "infini_train/include/nn/functional.h"
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
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{out_channels, in_channels, kernel_size, kernel_size},
                                           DataType::kFLOAT32, device, true);
    parameters_[kParamWeightName] = weight;
    init::KaimingUniform(weight, std::sqrt(5.0f));
    if (bias) {
        auto b = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32, device, true);
        parameters_[kParamBiasName] = b;
        const float bound = 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size));
        init::Uniform(b, -bound, bound);
    }
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 1);
    return {function::Conv2d(inputs[0], parameters_.at(kParamWeightName),
                             bias_ ? parameters_.at(kParamBiasName) : nullptr, stride_, padding_)};
}
} // namespace infini_train::nn

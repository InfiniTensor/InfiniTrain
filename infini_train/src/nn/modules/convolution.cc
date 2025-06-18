#include "infini_train/include/nn/modules/convolution.h"

#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/convolution.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Conv2D::Conv2D(int64_t in_channels, int64_t out_channels, size_t kernel_size, size_t stride, size_t padding,
               size_t dilation, size_t groups, bool bias, std::string padding_mode, Device device)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride),
      padding_(padding), dilation_(dilation), groups_(groups), bias_(bias), padding_mode_(padding_mode) {
    device_ = device;

    parameters_[kParamKernelName]
        = std::make_shared<Tensor>(
              std::vector<int64_t>{out_channels, in_channels, (int64_t)kernel_size, (int64_t)kernel_size},
              DataType::kFLOAT32, device)
              ->RequiresGrad();

    if (bias_) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32)->RequiresGrad();
    }

    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> Conv2D::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::Conv2D>(in_channels_, out_channels_, kernel_size_, stride_, padding_, dilation_,
                                              groups_, bias_)
        ->Apply(bias_ ? std::vector<std::shared_ptr<Tensor>>{input_tensors[0], parameters_[kParamKernelName],
                                                             parameters_[kParamBiasName]}
                      : std::vector<std::shared_ptr<Tensor>>{input_tensors[0], parameters_[kParamKernelName]});
}

void Conv2D::ResetParameters() {
    init::KaimingUniform(parameters_[kParamKernelName], sqrt(0.0f));
    if (bias_) {
        const auto [fan_in, _] = init::CalculateFanInAndFanOut(parameters_[kParamKernelName]);
        const float bound = fan_in > 0 ? 1.0 / sqrt(fan_in) : 0.0;
        init::Uniform(parameters_[kParamBiasName], -bound, bound);
    }
}
} // namespace infini_train::nn
#include "infini_train/include/nn/modules/conv.h"

#include <cmath>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
               int64_t stride, int64_t padding, int64_t dilation,
               int64_t groups, bool bias, Device device)
    : Conv2d(in_channels, out_channels, {kernel_size, kernel_size},
             {stride, stride}, {padding, padding}, {dilation, dilation}, groups,
             bias, device) {}

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels,
               std::pair<int64_t, int64_t> kernel_size,
               std::pair<int64_t, int64_t> stride,
               std::pair<int64_t, int64_t> padding,
               std::pair<int64_t, int64_t> dilation, int64_t groups, bool bias,
               Device device)
    : CloneableModule(kType),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      groups_(groups),
      bias_(bias) {
  CHECK_GT(groups_, 0);
  CHECK_EQ(in_channels_ % groups_, 0)
      << "in_channels must be divisible by groups";
  CHECK_EQ(out_channels_ % groups_, 0)
      << "out_channels must be divisible by groups";
  CHECK_EQ(dilation_.first, 1) << "Conv2d currently supports dilation=1 only";
  CHECK_EQ(dilation_.second, 1) << "Conv2d currently supports dilation=1 only";

  device_ = device;
  parameters_[kParamWeightName] =
      std::make_shared<Tensor>(
          std::vector<int64_t>{out_channels_, in_channels_ / groups_,
                               kernel_size_.first, kernel_size_.second},
          DataType::kFLOAT32, device_)
          ->RequiresGrad();
  if (bias_) {
    parameters_[kParamBiasName] =
        std::make_shared<Tensor>(std::vector<int64_t>{out_channels_},
                                 DataType::kFLOAT32, device_)
            ->RequiresGrad();
  }
  ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  return std::make_shared<autograd::Conv2d>(stride_, padding_, dilation_,
                                            groups_)
      ->Apply(bias_
                  ? std::vector<
                        std::shared_ptr<Tensor>>{input_tensors[0],
                                                 parameters_[kParamWeightName],
                                                 parameters_[kParamBiasName]}
                  : std::vector<std::shared_ptr<Tensor>>{
                        input_tensors[0], parameters_[kParamWeightName]});
}

bool Conv2d::has_bias() const { return bias_; }

void Conv2d::ResetParameters() {
  init::KaimingUniform(parameters_[kParamWeightName], std::sqrt(5.0f));
  if (bias_) {
    const auto [fan_in, _] =
        init::CalculateFanInAndFanOut(parameters_[kParamWeightName]);
    const float bound =
        fan_in > 0 ? 1.0f / std::sqrt(static_cast<float>(fan_in)) : 0.0f;
    init::Uniform(parameters_[kParamBiasName], -bound, bound);
  }
}

}  // namespace infini_train::nn

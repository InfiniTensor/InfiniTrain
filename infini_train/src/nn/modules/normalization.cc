#include "infini_train/include/nn/modules/normalization.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
LayerNorm::LayerNorm(const std::vector<int64_t> &normalized_shape, float eps,
                     Device device)
    : CloneableModule(kType), eps_(eps) {
  device_ = device;

  parameters_[kParamWeightName] =
      std::make_shared<Tensor>(normalized_shape, DataType::kFLOAT32, device_)
          ->RequiresGrad();
  parameters_[kParamBiasName] =
      std::make_shared<Tensor>(normalized_shape, DataType::kFLOAT32, device_)
          ->RequiresGrad();
  if (parallel::global::GetSequenceParallelEnabled()) {
    parameters_[kParamWeightName]->set_sequence_parallel(true);
    parameters_[kParamBiasName]->set_sequence_parallel(true);
  }
  ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  auto outputs = std::make_shared<autograd::LayerNorm>(eps_)->Apply(
      {input_tensors[0], parameters_[kParamWeightName],
       parameters_[kParamBiasName]});
  return {outputs[0]};
}

void LayerNorm::ResetParameters() {
  init::Ones(parameters_[kParamWeightName]);
  init::Zeros(parameters_[kParamBiasName]);
}

RMSNorm::RMSNorm(int64_t dim, float eps, Device device)
    : CloneableModule(kType), eps_(eps) {
  parameters_[kParamWeightName] =
      std::make_shared<Tensor>(std::vector<int64_t>{dim}, DataType::kFLOAT32,
                               device)
          ->RequiresGrad();
  if (parallel::global::GetSequenceParallelEnabled()) {
    parameters_[kParamWeightName]->set_sequence_parallel(true);
  }
  nn::init::Ones(parameters_[kParamWeightName]);
}

std::vector<std::shared_ptr<Tensor>> RMSNorm::Forward(
    const std::vector<std::shared_ptr<Tensor>> &x) {
  // broadcasted Mul([4, 64, 2048] * [4, 64, 1])
  auto norm =
      x[0] *
      function::Rsqrt(function::Mean(function::Pow(x[0], 2), -1, true) + eps_);
  return {norm * parameters_[kParamWeightName]};
}

BatchNorm2d::BatchNorm2d(int64_t num_features, float eps, float momentum,
                         Device device)
    : CloneableModule(kType),
      num_features_(num_features),
      eps_(eps),
      momentum_(momentum) {
  device_ = device;
  const std::vector<int64_t> shape{num_features_};
  parameters_[kParamWeightName] =
      std::make_shared<Tensor>(shape, DataType::kFLOAT32, device_)
          ->RequiresGrad();
  parameters_[kParamBiasName] =
      std::make_shared<Tensor>(shape, DataType::kFLOAT32, device_)
          ->RequiresGrad();
  buffers_[kBufferRunningMeanName] =
      std::make_shared<Tensor>(shape, DataType::kFLOAT32, device_);
  buffers_[kBufferRunningVarName] =
      std::make_shared<Tensor>(shape, DataType::kFLOAT32, device_);
  ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> BatchNorm2d::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  auto outputs =
      std::make_shared<autograd::BatchNorm2d>(buffers_[kBufferRunningMeanName],
                                              buffers_[kBufferRunningVarName],
                                              is_training(), momentum_, eps_)
          ->Apply({input_tensors[0], parameters_[kParamWeightName],
                   parameters_[kParamBiasName]});
  return {outputs[0]};
}

void BatchNorm2d::ResetParameters() {
  init::Ones(parameters_[kParamWeightName]);
  init::Zeros(parameters_[kParamBiasName]);
  init::Zeros(buffers_[kBufferRunningMeanName]);
  init::Ones(buffers_[kBufferRunningVarName]);
}
}  // namespace infini_train::nn

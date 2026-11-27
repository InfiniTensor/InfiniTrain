#include "infini_train/include/autograd/pooling.h"

#include <tuple>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

MaxPool2d::MaxPool2d(std::pair<int64_t, int64_t> kernel_size,
                     std::pair<int64_t, int64_t> stride,
                     std::pair<int64_t, int64_t> padding)
    : Function(kType),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {}

std::vector<std::shared_ptr<Tensor>> MaxPool2d::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK_EQ(input_tensors.size(), 1);
  const auto &input = input_tensors[0]->Contiguous();
  auto device = input->GetDevice().type();
  auto [output, indices] =
      Dispatcher::Instance()
          .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
              {device, "MaxPool2dForward"}, input, kernel_size_.first,
              kernel_size_.second, stride_.first, stride_.second,
              padding_.first, padding_.second);
  return {output, indices};
}

void MaxPool2d::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
  CHECK_EQ(output_tensors.size(), 2);
  input_dims_ = input_tensors[0]->Dims();
  const auto &indices = output_tensors[1];
  ctx_.MarkNonDifferentiable({indices});
  ctx_.SaveForBackward({indices});
}

std::vector<std::shared_ptr<Tensor>> MaxPool2d::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  auto saved_tensors = ctx_.GetSavedTensors();
  CHECK_EQ(saved_tensors.size(), 1);
  CHECK_GE(grad_outputs.size(), 1);
  const auto &indices = saved_tensors[0];
  const auto &grad_output = grad_outputs[0]->Contiguous();

  auto device = grad_output->GetDevice().type();
  auto grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {device, "MaxPool2dBackward"}, grad_output, indices, input_dims_);
  return {grad_input};
}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(std::pair<int64_t, int64_t> output_size)
    : Function(kType), output_size_(output_size) {}

std::vector<std::shared_ptr<Tensor>> AdaptiveAvgPool2d::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK_EQ(input_tensors.size(), 1);
  const auto &input = input_tensors[0]->Contiguous();
  auto device = input->GetDevice().type();
  auto output = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {device, "AdaptiveAvgPool2dForward"}, input, output_size_.first,
      output_size_.second);
  return {output};
}

void AdaptiveAvgPool2d::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &) {
  input_dims_ = input_tensors[0]->Dims();
}

std::vector<std::shared_ptr<Tensor>> AdaptiveAvgPool2d::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  CHECK_EQ(grad_outputs.size(), 1);
  const auto &grad_output = grad_outputs[0]->Contiguous();
  auto device = grad_output->GetDevice().type();
  auto grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {device, "AdaptiveAvgPool2dBackward"}, grad_output, input_dims_);
  return {grad_input};
}

}  // namespace infini_train::autograd

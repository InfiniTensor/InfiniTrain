#include "infini_train/include/autograd/activations.h"

#include <cassert>
#include <ostream>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Sigmoid::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK_EQ(input_tensors.size(), 1);
  const auto &input = input_tensors[0];

  auto device = input->GetDevice().type();
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {device, "SigmoidForward"}, input)};
}

void Sigmoid::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &,
    const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
  const auto &output = output_tensors[0];
  ctx_.SaveForBackward({output});
}

std::vector<std::shared_ptr<Tensor>> Sigmoid::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  auto saved_tensors = ctx_.GetSavedTensors();
  CHECK_EQ(saved_tensors.size(), 1);
  const auto &output = saved_tensors[0];
  CHECK_EQ(grad_outputs.size(), 1);
  const auto &grad_output = grad_outputs[0];

  auto device = output->GetDevice().type();
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {device, "SigmoidBackward"}, output, grad_output)};
}

std::vector<std::shared_ptr<Tensor>> NewGELU::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
  assert(inputs.size() == 1);
  const auto &input = inputs[0];
  auto [output, saved_tanh] =
      Dispatcher::Instance()
          .Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
              {input->GetDevice().type(), "NewGELUForward"}, input);
  saved_tanh_ = std::move(saved_tanh);
  return {output};
}

void NewGELU::SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                           const std::vector<std::shared_ptr<Tensor>> &) {
  ctx_.SaveForBackward({inputs[0], saved_tanh_});
  saved_tanh_.reset();
}

std::vector<std::shared_ptr<Tensor>> NewGELU::Backward(
    const std::vector<std::shared_ptr<Tensor>> &gradients) {
  assert(gradients.size() == 1);
  const auto saved = ctx_.GetSavedTensors();
  auto gradient = gradients[0]->Contiguous();
  if (gradient->Dtype() != saved[0]->Dtype()) {
    gradient = std::make_shared<Tensor>(gradient->To(saved[0]->Dtype()));
  }
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {saved[0]->GetDevice().type(), "NewGELUBackward"}, saved[0], saved[1],
      gradient)};
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK_EQ(input_tensors.size(), 1);
  const auto &input = input_tensors[0];
  CHECK_GT(input->Dims().size(), 0);
  CHECK_EQ(input->Dims().back() % 2, 0)
      << "SwiGLU expects an even last dimension";

  auto device = input->GetDevice().type();
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {device, "SwiGLUForward"}, input)};
}

void SwiGLU::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &) {
  ctx_.SaveForBackward({input_tensors[0]});
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  auto saved_tensors = ctx_.GetSavedTensors();
  CHECK_EQ(saved_tensors.size(), 1);
  CHECK_EQ(grad_outputs.size(), 1);
  const auto &input = saved_tensors[0];
  const auto &grad_output = grad_outputs[0];

  auto device = input->GetDevice().type();
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {device, "SwiGLUBackward"}, input, grad_output)};
}
}  // namespace infini_train::autograd

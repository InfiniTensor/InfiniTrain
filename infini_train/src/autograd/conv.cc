#include "infini_train/include/autograd/conv.h"

#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

Conv2d::Conv2d(std::pair<int64_t, int64_t> stride,
               std::pair<int64_t, int64_t> padding,
               std::pair<int64_t, int64_t> dilation, int64_t groups)
    : Function(kType),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      groups_(groups) {
  CHECK_EQ(dilation_.first, 1) << "Conv2d currently supports dilation=1 only";
  CHECK_EQ(dilation_.second, 1) << "Conv2d currently supports dilation=1 only";
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK(input_tensors.size() == 2 || input_tensors.size() == 3);
  const auto &input = input_tensors[0]->Contiguous();
  const auto &weight = input_tensors[1]->Contiguous();
  const auto &bias =
      input_tensors.size() == 3 ? input_tensors[2]->Contiguous() : nullptr;

  auto device = input->GetDevice().type();
  auto output = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {device, "Conv2dForward"}, input, weight, bias, stride_.first,
      stride_.second, padding_.first, padding_.second, groups_);
  return {output};
}

void Conv2d::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &) {
  const auto &input = input_tensors[0];
  const auto &weight = input_tensors[1];
  const bool need_input =
      ctx_.needs_input_grad().size() > 0 && ctx_.needs_input_grad()[0];
  const bool need_weight =
      ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];

  ctx_.SaveForBackward(
      {need_weight ? input : nullptr, need_input ? weight : nullptr});
  has_bias_ = (input_tensors.size() == 3);
  input_dims_ = input->Dims();
  weight_dims_ = weight->Dims();
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  auto saved_tensors = ctx_.GetSavedTensors();
  CHECK_EQ(saved_tensors.size(), 2);
  CHECK_EQ(grad_outputs.size(), 1);
  const auto &input = saved_tensors[0];
  const auto &weight = saved_tensors[1];
  const auto &grad_output = grad_outputs[0]->Contiguous();

  CHECK(!ctx_.needs_input_grad().empty())
      << "needs_input_grad not populated in Conv2d::Backward";
  const bool need_grad_input = ctx_.needs_input_grad()[0];
  const bool need_grad_weight =
      ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];
  const bool need_grad_bias = has_bias_ && ctx_.needs_input_grad().size() > 2 &&
                              ctx_.needs_input_grad()[2];

  auto device = grad_output->GetDevice().type();
  std::shared_ptr<Tensor> grad_input = nullptr;
  std::shared_ptr<Tensor> grad_weight = nullptr;
  std::shared_ptr<Tensor> grad_bias = nullptr;

  if (need_grad_input) {
    grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "Conv2dBackwardInput"}, weight->Contiguous(), grad_output,
        input_dims_, stride_.first, stride_.second, padding_.first,
        padding_.second, groups_);
  }
  if (need_grad_weight) {
    grad_weight = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "Conv2dBackwardWeight"}, input->Contiguous(), grad_output,
        weight_dims_, stride_.first, stride_.second, padding_.first,
        padding_.second, groups_);
  }
  if (need_grad_bias) {
    grad_bias = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "Conv2dBackwardBias"}, grad_output);
  }

  if (has_bias_) {
    return {grad_input, grad_weight, grad_bias};
  }
  return {grad_input, grad_weight};
}

}  // namespace infini_train::autograd

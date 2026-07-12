#include "infini_train/include/autograd/normalization.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> LayerNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors[2];

    auto device = input->GetDevice().type();
    auto [output, mean, rstd]
        = Dispatcher::Instance()
              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                  {device, "LayerNormForward"}, input, weight, bias, eps_);
    return {output, mean, rstd};
}

void LayerNorm::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    CHECK_EQ(output_tensors.size(), 3);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors[2];
    const auto &mean = output_tensors[1];
    const auto &rstd = output_tensors[2];
    ctx_.MarkNonDifferentiable({mean, rstd});
    ctx_.SaveForBackward({input, weight, bias, mean, rstd});
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 5);
    const auto &input = saved_tensors[0];
    const auto &weight = saved_tensors[1];
    const auto &bias = saved_tensors[2];
    const auto &mean = saved_tensors[3];
    const auto &rstd = saved_tensors[4];
    CHECK_GE(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice().type();
    auto [grad_input, grad_weight, grad_bias]
        = Dispatcher::Instance()
              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                  {device, "LayerNormBackward"}, input, weight, bias, mean, rstd, grad_output);
    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::autograd

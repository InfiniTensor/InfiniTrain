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
    SaveForBackward({input, weight, bias, mean, rstd});
    return {output};
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(SavedTensorsSize(), 5);
    const auto &input = GetSavedTensor(0);
    const auto &weight = GetSavedTensor(1);
    const auto &bias = GetSavedTensor(2);
    const auto &mean = GetSavedTensor(3);
    const auto &rstd = GetSavedTensor(4);
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice().type();
    auto [grad_input, grad_weight, grad_bias]
        = Dispatcher::Instance()
              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                  {device, "LayerNormBackward"}, input, weight, bias, mean, rstd, grad_output);
    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::autograd

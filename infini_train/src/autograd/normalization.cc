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

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "LayerNormForward"});
    auto [output, mean, rstd]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
            input, weight, bias, eps_);
    saved_tensors_ = {mean, rstd};
    return {output};
}

void LayerNorm::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                             const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors[2];
    saved_tensors_.insert(saved_tensors_.begin(), {input, weight, bias});
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 5);
    const auto &input = saved_tensors_[0];
    const auto &weight = saved_tensors_[1];
    const auto &bias = saved_tensors_[2];
    const auto &mean = saved_tensors_[3];
    const auto &rstd = saved_tensors_[4];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "LayerNormBackward"});
    auto [grad_input, grad_weight, grad_bias]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
            input, weight, bias, mean, rstd, grad_output);
    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::autograd

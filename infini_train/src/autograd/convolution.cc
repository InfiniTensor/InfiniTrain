#include "infini_train/include/autograd/convolution.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Conv2D::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 2);
    const auto &input_tensor = input_tensors[0];
    const auto &kernel_tensor = input_tensors[1];
    const auto &bias_tensor = input_tensors.size() == 3 ? input_tensors[2] : nullptr;

    auto device = input_tensor->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "Conv2DForward"});

    return {kernel.Call<std::shared_ptr<Tensor>>(input_tensor, kernel_tensor, bias_tensor, kernel_size_, in_channels_,
                                                 out_channels_, stride_, padding_, dilation_, groups_, bias_)};
}

void Conv2D::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    const auto &kernel = input_tensors[1];
    saved_tensors_ = {input, kernel};
    bias_ = input_tensors.size() == 3;
}

std::vector<std::shared_ptr<Tensor>> Conv2D::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input_tensor = saved_tensors_[0];

    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input_tensor->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "Conv2DBackward"});

    auto [grad_input, grad_kernel, grad_bias]
        = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
            grad_output, input_tensor, kernel_size_, in_channels_, out_channels_, stride_, padding_, dilation_, groups_,
            bias_);

    return bias_ ? std::vector<std::shared_ptr<Tensor>>{grad_input, grad_kernel, grad_bias}
                 : std::vector<std::shared_ptr<Tensor>>{grad_input, grad_kernel};
}
} // namespace infini_train::autograd
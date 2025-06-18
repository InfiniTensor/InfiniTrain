#include "infini_train/include/autograd/maxpool.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> MaxPool2D::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);

    auto input_tensor = input_tensors[0];
    auto device = input_tensor->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MaxPool2DForward"});

    auto [output, mask] = kernel.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
        input_tensor, kernel_size_, stride_, padding_, dilation_);

    return {output, mask};
}

void MaxPool2D::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    saved_tensors_ = {output_tensors[1]};
}

std::vector<std::shared_ptr<Tensor>> MaxPool2D::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);

    auto device = grad_outputs[0]->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MaxPool2DBackward"});

    return {kernel.Call<std::shared_ptr<Tensor>>(grad_outputs[0], saved_tensors_[0], kernel_size_, stride_)};
}
} // namespace infini_train::autograd
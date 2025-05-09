#include "infini_train/include/autograd/sparse.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Embedding::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "EmbeddingForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, weight)};
}

void Embedding::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    weight_dims_ = weight->Dims();
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Embedding::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &input = saved_tensors_[0];
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "EmbeddingBackward"});
    auto grad_weight = kernel.Call<std::shared_ptr<Tensor>>(input, weight_dims_, grad_output);
    return {nullptr, grad_weight};
}
} // namespace infini_train::autograd

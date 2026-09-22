#include "infini_train/include/autograd/scatter_src.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>>
ScatterSrc::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &input = input_tensors[0];
    const auto &indices = input_tensors[1];
    const auto &src = input_tensors[2];
    const auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "ScatterSrcForward"}, input, dim_, indices, src)};
}

void ScatterSrc::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                              const std::vector<std::shared_ptr<Tensor>> &) {
    ctx_.SaveForBackward({input_tensors[1]});
}

std::vector<std::shared_ptr<Tensor>>
ScatterSrc::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];
    const auto indices = ctx_.GetSavedTensors()[0];
    const auto device = grad_output->GetDevice().type();
    auto gradients = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "ScatterSrcBackward"}, grad_output, dim_, indices);
    return {gradients[0], nullptr, gradients[1]};
}

} // namespace infini_train::autograd

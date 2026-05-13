#include "infini_train/include/autograd/gather.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Gather::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &index = input_tensors[1];

    auto device = input->GetDevice().type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "GatherForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, index, dim_)};
}

void Gather::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &index = input_tensors[1];
    input_dims_ = input->Dims();
    SaveForBackward({index});
}

std::vector<std::shared_ptr<Tensor>> Gather::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];
    const auto &index = GetSavedTensor(0);

    auto device = grad_outputs[0]->GetDevice();
    auto kernel = Dispatcher::Instance().GetKernel({device.type(), "GatherBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, index, dim_, input_dims_), nullptr};
}

} // namespace infini_train::autograd

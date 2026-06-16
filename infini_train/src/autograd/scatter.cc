#include "infini_train/include/autograd/scatter.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> Scatter::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &values = input_tensors[0];
    const auto &indices = input_tensors[1];
    auto device = values->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "ScatterForward"}, values, indices,
                                                                 output_dims_)};
}

void Scatter::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                           const std::vector<std::shared_ptr<Tensor>> &) {
    ctx_.SaveForBackward({input_tensors[1]});
}

std::vector<std::shared_ptr<Tensor>> Scatter::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];
    const auto &indices = ctx_.saved_tensors()[0];
    auto device = grad_output->GetDevice().type();
    auto grad_values
        = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "ScatterBackward"}, grad_output, indices);
    return {grad_values, nullptr};
}

} // namespace infini_train::autograd

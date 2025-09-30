#include "infini_train/include/autograd/reduction.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Mean::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MeanForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, dim_, keep_dim_)};
}

void Mean::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Mean::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MeanBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input_dims_, dim_, keep_dim_)};
}

std::vector<std::shared_ptr<Tensor>> Sum::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "SumForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, dim_, keep_dim_)};
}

void Sum::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Sum::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "SumBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input_dims_, dim_, keep_dim_)};
}

std::vector<std::shared_ptr<Tensor>> Max::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MaxForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, dim_, keep_dim_)};
}

void Max::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    const auto &output = output_tensors[0];
    saved_tensors_ = {input, output};
}

std::vector<std::shared_ptr<Tensor>> Max::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &grad_output = grad_outputs[0];
    const auto &input = saved_tensors_[0];
    const auto &reduced = saved_tensors_[1];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MaxBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input, reduced, dim_, keep_dim_)};
}

std::vector<std::shared_ptr<Tensor>> Min::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MinForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, dim_, keep_dim_)};
}

void Min::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    const auto &output = output_tensors[0];
    saved_tensors_ = {input, output};
}

std::vector<std::shared_ptr<Tensor>> Min::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &grad_output = grad_outputs[0];
    const auto &input = saved_tensors_[0];
    const auto &reduced = saved_tensors_[1];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MinBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input, reduced, dim_, keep_dim_)};
}
} // namespace infini_train::autograd

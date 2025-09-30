#include "infini_train/include/autograd/elementwise.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}

std::vector<std::shared_ptr<Tensor>> Reciprocal::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "ReciprocalForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

void Reciprocal::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                              const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Reciprocal::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "ReciprocalBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input)};
}

std::vector<std::shared_ptr<Tensor>> Sin::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "SinForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

void Sin::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Sin::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "SinBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input)};
}

std::vector<std::shared_ptr<Tensor>> Cos::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CosForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

void Cos::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Cos::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CosBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input)};
}

std::vector<std::shared_ptr<Tensor>> Tanh::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TanhForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

void Tanh::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                        const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &output = output_tensors[0];
    saved_tensors_ = {output};
}

std::vector<std::shared_ptr<Tensor>> Tanh::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &output = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TanhBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, output)};
}

std::vector<std::shared_ptr<Tensor>> Pow::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "PowForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, exponent_, scalar_is_base_)};
}

void Pow::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Pow::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "PowBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input, exponent_, scalar_is_base_)};
}

std::vector<std::shared_ptr<Tensor>> Rsqrt::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "RsqrtForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

void Rsqrt::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Rsqrt::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "RsqrtBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input)};
}

std::vector<std::shared_ptr<Tensor>> EqualsScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "EqualsScalarForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> EqualsScalar::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    LOG(FATAL) << "EqualsScalar::Backward shall not be called anytime";
    return {};
}

std::vector<std::shared_ptr<Tensor>> Add::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "AddForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(a, b)};
}

void Add::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    a_dims_ = input_tensors[0]->Dims();
    b_dims_ = input_tensors[1]->Dims();
}

std::vector<std::shared_ptr<Tensor>> Add::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "AddBackward"});
    auto [grad_a, grad_b]
        = kernel.Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(grad_output, a_dims_, b_dims_);
    return {grad_a, grad_b};
}

std::vector<std::shared_ptr<Tensor>> AddScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "AddScalarForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> AddScalar::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "AddScalarBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}

std::vector<std::shared_ptr<Tensor>> Sub::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "SubForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(a, b)};
}

void Sub::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    a_dims_ = input_tensors[0]->Dims();
    b_dims_ = input_tensors[1]->Dims();
}

std::vector<std::shared_ptr<Tensor>> Sub::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "SubBackward"});
    auto [grad_a, grad_b]
        = kernel.Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(grad_output, a_dims_, b_dims_);
    return {grad_a, grad_b};
}

std::vector<std::shared_ptr<Tensor>> Mul::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MulForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(a, b)};
}

void Mul::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];
    saved_tensors_ = {a, b};
}

std::vector<std::shared_ptr<Tensor>> Mul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &a = saved_tensors_[0];
    const auto &b = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MulBackward"});
    auto [grad_a, grad_b] = kernel.Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(grad_output, a, b);
    return {grad_a, grad_b};
}

std::vector<std::shared_ptr<Tensor>> MulScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MulScalarForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> MulScalar::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MulScalarBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> Div::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "DivForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(a, b)};
}

void Div::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                       const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];
    saved_tensors_ = {a, b};
}

std::vector<std::shared_ptr<Tensor>> Div::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &a = saved_tensors_[0];
    const auto &b = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "DivBackward"});
    auto [grad_a, grad_b] = kernel.Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(grad_output, a, b);
    return {grad_a, grad_b};
}
} // namespace infini_train::autograd

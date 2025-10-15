#include "infini_train/include/autograd/elementwise.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "NegForward"}, input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "NegBackward"}, grad_output)};
}

std::vector<std::shared_ptr<Tensor>> Reciprocal::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "ReciprocalForward"}, input)};
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
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "ReciprocalBackward"}, grad_output, input)};
}

std::vector<std::shared_ptr<Tensor>> Sin::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SinForward"}, input)};
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
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SinBackward"}, grad_output, input)};
}

std::vector<std::shared_ptr<Tensor>> Cos::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "CosForward"}, input)};
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
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "CosBackward"}, grad_output, input)};
}

std::vector<std::shared_ptr<Tensor>> Tanh::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TanhForward"}, input)};
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
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TanhBackward"}, grad_output, output)};
}

std::vector<std::shared_ptr<Tensor>> Pow::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "PowForward"}, input, exponent_,
                                                                 scalar_is_base_)};
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
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "PowBackward"}, grad_output, input, exponent_,
                                                                 scalar_is_base_)};
}

std::vector<std::shared_ptr<Tensor>> Rsqrt::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "RsqrtForward"}, input)};
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
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "RsqrtBackward"}, grad_output, input)};
}

std::vector<std::shared_ptr<Tensor>> EqualsScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "EqualsScalarForward"}, input, scalar_)};
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
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "AddForward"}, a, b)};
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
    auto [grad_a, grad_b] = Dispatcher::Instance().Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
        {device, "AddBackward"}, grad_output, a_dims_, b_dims_);
    return {grad_a, grad_b};
}

std::vector<std::shared_ptr<Tensor>> AddScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "AddScalarForward"}, input, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> AddScalar::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "AddScalarBackward"}, grad_output)};
}

std::vector<std::shared_ptr<Tensor>> Sub::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SubForward"}, a, b)};
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
    auto [grad_a, grad_b] = Dispatcher::Instance().Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
        {device, "SubBackward"}, grad_output, a_dims_, b_dims_);
    return {grad_a, grad_b};
}

std::vector<std::shared_ptr<Tensor>> Mul::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MulForward"}, a, b)};
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
    auto [grad_a, grad_b] = Dispatcher::Instance().Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
        {device, "MulBackward"}, grad_output, a, b);
    return {grad_a, grad_b};
}

std::vector<std::shared_ptr<Tensor>> MulScalar::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MulScalarForward"}, input, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> MulScalar::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MulScalarBackward"}, grad_output, scalar_)};
}

std::vector<std::shared_ptr<Tensor>> Div::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &a = input_tensors[0];
    const auto &b = input_tensors[1];

    auto device = a->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "DivForward"}, a, b)};
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
    auto [grad_a, grad_b] = Dispatcher::Instance().Call<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
        {device, "DivBackward"}, grad_output, a, b);
    return {grad_a, grad_b};
}
} // namespace infini_train::autograd

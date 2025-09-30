#include "infini_train/include/autograd/transform.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Tril::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TrilForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, diagonal_)};
}

std::vector<std::shared_ptr<Tensor>> Tril::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TrilBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, diagonal_)};
}

std::vector<std::shared_ptr<Tensor>> Triu::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TriuForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, diagonal_)};
}

std::vector<std::shared_ptr<Tensor>> Triu::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TriuBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, diagonal_)};
}

std::vector<std::shared_ptr<Tensor>> Transpose::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TransposeForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, dim0_, dim1_)};
}

std::vector<std::shared_ptr<Tensor>> Transpose::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "TransposeBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, dim0_, dim1_)};
}

std::vector<std::shared_ptr<Tensor>> Mask::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MaskForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, mask_, value_)};
}

std::vector<std::shared_ptr<Tensor>> Mask::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "MaskBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, mask_)};
}

std::vector<std::shared_ptr<Tensor>>
RepeatInterleave::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "RepeatInterleaveForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input, repeat_, dim_)};
}

void RepeatInterleave::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                    const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>>
RepeatInterleave::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "RepeatInterleaveBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output, input_dims_, dim_)};
}
} // namespace infini_train::autograd

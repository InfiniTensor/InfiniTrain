#include "infini_train/include/autograd/transform.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Tril::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TrilForward"}, input, diagonal_)};
}

std::vector<std::shared_ptr<Tensor>> Tril::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TrilBackward"}, grad_output, diagonal_)};
}

std::vector<std::shared_ptr<Tensor>> Triu::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TriuForward"}, input, diagonal_)};
}

std::vector<std::shared_ptr<Tensor>> Triu::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TriuBackward"}, grad_output, diagonal_)};
}

std::vector<std::shared_ptr<Tensor>> Transpose::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TransposeForward"}, input, dim0_, dim1_)};
}

std::vector<std::shared_ptr<Tensor>> Transpose::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    return {
        Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TransposeBackward"}, grad_output, dim0_, dim1_)};
}

std::vector<std::shared_ptr<Tensor>> Mask::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MaskForward"}, input, mask_, value_)};
}

std::vector<std::shared_ptr<Tensor>> Mask::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MaskBackward"}, grad_output, mask_)};
}

std::vector<std::shared_ptr<Tensor>>
RepeatInterleave::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice()->Type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "RepeatInterleaveForward"}, input, repeat_,
                                                                 dim_)};
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
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "RepeatInterleaveBackward"}, grad_output,
                                                                 input_dims_, dim_)};
}
} // namespace infini_train::autograd

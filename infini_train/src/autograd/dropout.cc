#include "infini_train/include/autograd/dropout.h"

#include "infini_train/include/core/runtime/dropout_kernels.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> Dropout::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    mask_ = std::make_shared<Tensor>(input->Dims(), DataType::kUINT8, input->GetDevice());
    dropout_forward_kernel(*output, *mask_, *input, p_, generator_);
    return {output};
}

void Dropout::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                           const std::vector<std::shared_ptr<Tensor>> &) {
    if (!needs_input_grad_.empty() && needs_input_grad_[0]) {
        saved_tensors_ = {mask_};
    }
    mask_.reset();
}

std::vector<std::shared_ptr<Tensor>> Dropout::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &grad_output = grad_outputs[0];
    const auto &mask = saved_tensors_[0];

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    dropout_backward_kernel(*grad_input, *grad_output, *mask, p_);
    return {grad_input};
}

} // namespace infini_train::autograd

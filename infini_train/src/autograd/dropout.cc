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
    if (!ctx_.needs_input_grad().empty() && ctx_.needs_input_grad()[0]) {
        ctx_.SaveForBackward({mask_});
    }
    mask_.reset();
}

std::vector<std::shared_ptr<Tensor>> Dropout::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 1);
    const auto &grad_output = grad_outputs[0];
    const auto &mask = saved_tensors[0];

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    dropout_backward_kernel(*grad_input, *grad_output, *mask, p_);
    return {grad_input};
}

} // namespace infini_train::autograd

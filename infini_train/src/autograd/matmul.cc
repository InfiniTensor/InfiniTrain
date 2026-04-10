#include "infini_train/include/autograd/matmul.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Matmul::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];

    auto device = input1->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MatmulForward"}, input1, input2)};
}

void Matmul::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];
    const auto &output = output_tensors[0];
    // Cast saved tensors to forward compute dtype (output dtype) so backward
    // computes in the same precision as forward, matching PyTorch's behavior.

    // FIXME: An extra cast (input1/input2 -> compute_dtype) is performed here because
    // autocast runs before autograd. The correct approach is to adjust the ordering or
    // integration of autocast and autograd so that autograd receives already-cast tensors,
    // avoiding the redundant cast.

    // FIXME: compute_dtype is not necessarily the dtype of output_tensor; it should be
    // determined by autocast, not derived from output->Dtype().
    auto compute_dtype = output->Dtype();

    // grad_input1 = grad_output @ input2^T, so input2 is needed
    // grad_input2 = grad_output^T @ input1, so input1 is needed
    bool need_grad_input1 = needs_input_grad_.size() > 0 && needs_input_grad_[0];
    bool need_grad_input2 = needs_input_grad_.size() > 1 && needs_input_grad_[1];

    auto cast = [&](const std::shared_ptr<Tensor> &t) {
        return t->Dtype() == compute_dtype ? t : std::make_shared<Tensor>(t->To(compute_dtype));
    };

    saved_tensors_ = {need_grad_input2 ? cast(input1) : nullptr, need_grad_input1 ? cast(input2) : nullptr};
    out_features_ = output->Dims()[0];
}

std::vector<std::shared_ptr<Tensor>> Matmul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input1 = saved_tensors_[0];
    const auto &input2 = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    CHECK(!needs_input_grad_.empty()) << "needs_input_grad_ not populated in Matmul::Backward";
    bool need_grad_input1 = needs_input_grad_.size() > 0 && needs_input_grad_[0];
    bool need_grad_input2 = needs_input_grad_.size() > 1 && needs_input_grad_[1];

    auto device = input1->GetDevice().type();

    std::shared_ptr<Tensor> grad_input1 = nullptr;
    std::shared_ptr<Tensor> grad_input2 = nullptr;

    if (need_grad_input1) {
        grad_input1 = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MatmulBackwardInput1"}, input2,
                                                                           grad_output, input1->Dims());
    }
    if (need_grad_input2) {
        grad_input2 = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MatmulBackwardInput2"}, input1,
                                                                           grad_output, input2->Dims());
    }

    return {grad_input1, grad_input2};
}
} // namespace infini_train::autograd

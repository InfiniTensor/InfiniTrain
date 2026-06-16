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
    bool need_grad_input1 = ctx_.needs_input_grad().size() > 0 && ctx_.needs_input_grad()[0];
    bool need_grad_input2 = ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];

    auto cast = [&](const std::shared_ptr<Tensor> &t) {
        return t->Dtype() == compute_dtype ? t : std::make_shared<Tensor>(t->To(compute_dtype));
    };

    ctx_.SaveForBackward({need_grad_input2 ? cast(input1) : nullptr, need_grad_input1 ? cast(input2) : nullptr});
    input1_dims_ = input1->Dims();
    input2_dims_ = input2->Dims();
    out_features_ = output->Dims()[0];
}

std::vector<std::shared_ptr<Tensor>> Matmul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(ctx_.saved_tensors().size(), 2);
    const auto &input1 = ctx_.saved_tensors()[0];
    const auto &input2 = ctx_.saved_tensors()[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    CHECK(!ctx_.needs_input_grad().empty()) << "needs_input_grad not populated in Matmul::Backward";
    bool need_grad_input1 = ctx_.needs_input_grad().size() > 0 && ctx_.needs_input_grad()[0];
    bool need_grad_input2 = ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];

    auto device = grad_output->GetDevice().type();

    std::shared_ptr<Tensor> grad_input = nullptr;
    std::shared_ptr<Tensor> grad_other = nullptr;

    if (need_grad_input1) {
        CHECK(input2 != nullptr) << "input2 not saved but need_grad_input1 is true";
        grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MatmulBackwardInput"}, input2,
                                                                          grad_output, input1_dims_);
    }
    if (need_grad_input2) {
        CHECK(input1 != nullptr) << "input1 not saved but need_grad_input2 is true";
        grad_other = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MatmulBackwardOther"}, input1,
                                                                          grad_output, input2_dims_);
    }

    return {grad_input, grad_other};
}
} // namespace infini_train::autograd

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
    saved_tensors_ = {
        input1->Dtype() == compute_dtype ? input1 : std::make_shared<Tensor>(input1->To(compute_dtype)),
        input2->Dtype() == compute_dtype ? input2 : std::make_shared<Tensor>(input2->To(compute_dtype)),
    };
    out_features_ = output->Dims()[0];
}

std::vector<std::shared_ptr<Tensor>> Matmul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input1 = saved_tensors_[0];
    const auto &input2 = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input1->GetDevice().type();
    auto [grad_input1, grad_input2]
        = Dispatcher::Instance().Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
            {device, "MatmulBackward"}, input1, input2, grad_output);
    return {grad_input1, grad_input2};
}
} // namespace infini_train::autograd

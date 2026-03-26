#include "infini_train/include/autograd/linear.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Linear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors.size() == 3 ? input_tensors[2] : nullptr;

    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "LinearForward"}, input, weight, true, bias)};
}

void Linear::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    // Cast saved tensors to forward compute dtype (output dtype) so backward
    // computes in the same precision as forward, matching PyTorch's behavior.

    // FIXME: An extra cast (input/weight -> compute_dtype) is performed here because
    // autocast runs before autograd. The correct approach is to adjust the ordering or
    // integration of autocast and autograd so that autograd receives already-cast tensors,
    // avoiding the redundant cast.

    // FIXME: compute_dtype is not necessarily the dtype of output_tensor; it should be
    // determined by autocast, not derived from output_tensors[0]->Dtype().
    auto compute_dtype = output_tensors[0]->Dtype();
    saved_tensors_ = {
        input->Dtype() == compute_dtype ? input : std::make_shared<Tensor>(input->To(compute_dtype)),
        weight->Dtype() == compute_dtype ? weight : std::make_shared<Tensor>(weight->To(compute_dtype)),
    };
    bias_ = input_tensors.size() == 3;
    out_features_ = weight->Dims()[0];

    bool need_input = needs_input_grad_.size() > 0 && needs_input_grad_[0];
    bool need_weight = needs_input_grad_.size() > 1 && needs_input_grad_[1];

    // grad_input needs weight, grad_weight needs input
    saved_tensors_ = {need_weight ? input : nullptr, need_input ? weight : nullptr};

    meta_ = {.transpose = true,
             .has_bias = input_tensors.size() == 3,
             .in_features = weight->Dims()[1],
             .out_features = weight->Dims()[0],
             .input_dims = input->Dims()};
}

std::vector<std::shared_ptr<Tensor>> Linear::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input = saved_tensors_[0];
    const auto &weight = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    CHECK(!needs_input_grad_.empty()) << "needs_input_grad_ not populated in Linear::Backward";
    LinearGradFlags grad_flags = {.input = needs_input_grad_[0],
                                  .weight = needs_input_grad_.size() > 1 && needs_input_grad_[1],
                                  .bias = meta_.has_bias && needs_input_grad_.size() > 2 && needs_input_grad_[2]};

    auto device = grad_output->GetDevice().type();
    auto [grad_input, grad_weight, grad_bias]
        = Dispatcher::Instance()
              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                  {device, "LinearBackward"}, input, weight, true, out_features_, grad_output, bias_);
    return bias_ ? std::vector<std::shared_ptr<Tensor>>{grad_input, grad_weight, grad_bias}
                 : std::vector<std::shared_ptr<Tensor>>{grad_input, grad_weight};
}
} // namespace infini_train::autograd

#include "infini_train/include/autograd/scaled_dot_product_attention.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK(input_tensors.size() == 3 || input_tensors.size() == 4);
    const auto &q = input_tensors[0];
    const auto &k = input_tensors[1];
    const auto &v = input_tensors[2];
    const auto mask = input_tensors.size() == 4 ? input_tensors[3] : nullptr;

    auto device = q->GetDevice().type();
    // Call device kernel. Kernel name: ScaledDotProductAttentionForward
    auto out_and_lse = Dispatcher::Instance().Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
        {device, "ScaledDotProductAttentionForward"}, q, k, v, mask, dropout_p_, is_causal_, scale_,
        enable_gqa_);
    forward_out_ = std::get<0>(out_and_lse);
    forward_lse_ = std::get<1>(out_and_lse);
    auto out = forward_out_;
    return {out};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    (void)output_tensors;
    // Save q,k,v and mask (mask may be nullptr)
    const auto &q = input_tensors[0];
    const auto &k = input_tensors[1];
    const auto &v = input_tensors[2];
    std::shared_ptr<Tensor> mask = nullptr;
    has_attn_mask_input_ = (input_tensors.size() == 4);
    if (input_tensors.size() == 4) {
        mask = input_tensors[3];
    }
    saved_tensors_ = {q, k, v, mask};
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK(saved_tensors_.size() == 4);
    const auto &q = saved_tensors_[0];
    const auto &k = saved_tensors_[1];
    const auto &v = saved_tensors_[2];
    const auto &mask = saved_tensors_[3];

    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().type();

    CHECK(forward_out_ != nullptr);
    CHECK(forward_lse_ != nullptr);

    auto grads = Dispatcher::Instance().Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                                                       std::shared_ptr<Tensor>>>(
        {device, "ScaledDotProductAttentionBackward"}, grad_output, q, k, v, mask, forward_out_, forward_lse_,
        dropout_p_, is_causal_, scale_, enable_gqa_);

    forward_out_ = nullptr;
    forward_lse_ = nullptr;

    if (has_attn_mask_input_) {
        return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads), nullptr};
    }

    return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads)};
}

} // namespace infini_train::autograd

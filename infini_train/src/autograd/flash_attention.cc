#include "infini_train/include/autograd/flash_attention.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> FlashAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);  // Q, K, V
    const auto &Q = input_tensors[0];
    const auto &K = input_tensors[1];
    const auto &V = input_tensors[2];

    auto device = Q->GetDevice().type();
    // FlashAttentionForward returns [O, L]
    auto outputs = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionForward"}, Q, K, V, is_causal_);
    CHECK_EQ(outputs.size(), 2);

    l_cache_ = outputs[1];  // save L for SetupContext
    return {outputs[0]};    // return only O to the user
}

void FlashAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                   const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    // Save Q, K, V, O, L for backward
    saved_tensors_ = {
        input_tensors[0],  // Q
        input_tensors[1],  // K
        input_tensors[2],  // V
        output_tensors[0], // O
        l_cache_           // L
    };
    l_cache_ = nullptr;
}

std::vector<std::shared_ptr<Tensor>> FlashAttention::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_EQ(saved_tensors_.size(), 5);

    const auto &Q = saved_tensors_[0];
    const auto &K = saved_tensors_[1];
    const auto &V = saved_tensors_[2];
    const auto &O = saved_tensors_[3];
    const auto &L = saved_tensors_[4];
    const auto &dO = grad_outputs[0];

    auto device = Q->GetDevice().type();
    // FlashAttentionBackward returns [dQ, dK, dV]
    return Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionBackward"}, Q, K, V, O, L, dO, is_causal_);
}

} // namespace infini_train::autograd

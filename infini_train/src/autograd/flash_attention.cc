#include "infini_train/include/autograd/flash_attention.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

// Forward: inputs = {Q, K, V}.
// The kernel returns {O, L}, but we expose only {O} to the caller so that the
// autograd engine waits for exactly one upstream gradient (dO) before calling
// Backward. L is stashed in l_ for use in SetupContext.
std::vector<std::shared_ptr<Tensor>> FlashAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &q = input_tensors[0];
    auto device = q->GetDevice().type();
    auto result = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionForward"}, q, input_tensors[1], input_tensors[2], is_causal_);
    CHECK_EQ(result.size(), 2);
    l_ = result[1];  // save L for SetupContext
    return {result[0]};  // expose only O
}

// SetupContext: inputs = {Q, K, V}, outputs = {O}.
// l_ was populated by Forward and is included in saved_tensors_.
void FlashAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                  const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    CHECK(l_ != nullptr);
    // saved: {Q, K, V, O, L}
    saved_tensors_ = {input_tensors[0], input_tensors[1], input_tensors[2],
                      output_tensors[0], l_};
    l_ = nullptr;  // release the temporary reference
}

// Backward: grad_outputs = {dO}, returns {dQ, dK, dV}.
std::vector<std::shared_ptr<Tensor>> FlashAttention::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 5);
    const auto &q  = saved_tensors_[0];
    const auto &k  = saved_tensors_[1];
    const auto &v  = saved_tensors_[2];
    const auto &o  = saved_tensors_[3];
    const auto &l  = saved_tensors_[4];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &do_ = grad_outputs[0];
    auto device = q->GetDevice().type();
    return Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionBackward"}, q, k, v, o, l, do_, is_causal_);
}

} // namespace infini_train::autograd

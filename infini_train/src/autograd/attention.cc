#include "infini_train/include/autograd/attention.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK(input_tensors.size() == 3 || input_tensors.size() == 4);

    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];
    const auto &attn_mask = input_tensors.size() == 4 ? input_tensors[3] : nullptr;

    auto device = query->GetDevice().type();
    auto [output, lse, rng_seed, rng_offset]
        = Dispatcher::Instance().Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, uint64_t, uint64_t>>(
            {device, "FlashAttentionForward"}, query, key, value, attn_mask, dropout_p_, is_causal_, scale_,
            enable_gqa_);
    lse_ = lse;
    rng_seed_ = rng_seed;
    rng_offset_ = rng_offset;
    return {output};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                             const std::vector<std::shared_ptr<Tensor>> &) {
    CHECK(input_tensors.size() == 3 || input_tensors.size() == 4);

    has_attn_mask_ = input_tensors.size() == 4;
    saved_tensors_.clear();
    saved_tensors_.push_back(input_tensors[0]);
    saved_tensors_.push_back(input_tensors[1]);
    saved_tensors_.push_back(input_tensors[2]);
    if (has_attn_mask_) {
        saved_tensors_.push_back(input_tensors[3]);
    }
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK(grad_outputs.size() == 1);
    CHECK(saved_tensors_.size() == 3 || saved_tensors_.size() == 4);

    const auto &query = saved_tensors_[0];
    const auto &key = saved_tensors_[1];
    const auto &value = saved_tensors_[2];
    const auto &attn_mask = has_attn_mask_ ? saved_tensors_[3] : nullptr;
    const auto &grad_output = grad_outputs[0];

    auto device = query->GetDevice().type();
    auto [grad_query, grad_key, grad_value]
        = Dispatcher::Instance()
              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                  {device, "FlashAttentionBackward"}, query, key, value, attn_mask, lse_, grad_output, dropout_p_,
                  is_causal_, scale_, enable_gqa_, rng_seed_, rng_offset_);

    lse_ = nullptr;

    if (has_attn_mask_) {
        return {grad_query, grad_key, grad_value, nullptr};
    }
    return {grad_query, grad_key, grad_value};
}
} // namespace infini_train::autograd

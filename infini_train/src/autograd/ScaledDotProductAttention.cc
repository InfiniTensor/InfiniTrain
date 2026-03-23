#include "infini_train/include/autograd/ScaledDotProductAttention.h"

#include <cmath>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/kernels/cuda/flash_attention.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);

    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];

    CHECK_GE(query->Dims().size(), 2);
    CHECK_GE(key->Dims().size(), 2);
    CHECK_GE(value->Dims().size(), 2);

    const int64_t query_last_dim = query->Dims().back();
    const float scale
        = static_cast<float>(scale_.has_value() ? *scale_ : 1.0 / std::sqrt(static_cast<double>(query_last_dim)));

    // ScaledDotProductAttention only supports CUDA device
    auto device = query->GetDevice().type();
    CHECK(device == Device::Device::DeviceType::kCUDA)
        << "ScaledDotProductAttention only supports CUDA device. "
        << "For CPU or other devices, please use traditional attention implementation.";

    // Note: is_causal and enable_gqa are supported in FlashAttention kernel
    // Dropout is now supported using seed-based approach for reproducibility
    flash_output_ = Dispatcher::Instance().Call<kernels::cuda::FlashAttentionForwardOutput>(
        {device, "FlashAttentionForward"}, query, key, value, attn_mask_, scale, is_causal_, dropout_p_, enable_gqa_);

    // Return output tensor only; saved_tensors_ will be set in SetupContext
    return {flash_output_.output};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];
    const auto &output = output_tensors[0];

    // Save tensors for backward pass:
    // - query, key, value: input tensors
    // - output: forward output
    // - logsumexp: logsumexp tensor for backward pass (from flash_output_)
    // - dropout_seed: dropout seed tensor for backward pass (from flash_output_)
    saved_tensors_ = {query, key, value, output, flash_output_.logsumexp, flash_output_.dropout_seed};
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 6) << "FlashAttention backward expects 6 saved tensors";
    CHECK_EQ(grad_outputs.size(), 1);

    const auto &query = saved_tensors_[0];
    const auto &key = saved_tensors_[1];
    const auto &value = saved_tensors_[2];
    const auto &output = saved_tensors_[3];
    const auto &logsumexp = saved_tensors_[4];
    const auto &dropout_seed = saved_tensors_[5];
    const auto &grad_output = grad_outputs[0];

    const float scale
        = static_cast<float>(scale_.has_value() ? *scale_ : 1.0 / std::sqrt(static_cast<double>(query->Dims().back())));

    auto device = query->GetDevice().type();
    auto grad_tensors = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionBackward"}, query, key, value, output, grad_output, logsumexp, dropout_seed, attn_mask_,
        scale, is_causal_, dropout_p_, enable_gqa_);

    return grad_tensors;
}
} // namespace infini_train::autograd

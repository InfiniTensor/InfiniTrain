#include "infini_train/include/autograd/scaled_dot_product_attention.h"

#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3) << "ScaledDotProductAttention expects 3 inputs: Q, K, V";

    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];

    // Q: [B, H_q, N, d], K: [B, H_kv, N, d], V: [B, H_kv, N, d]
    CHECK_EQ(query->Dims().size(), 4) << "Query must be 4D [B, H, N, d]";
    CHECK_EQ(key->Dims().size(), 4) << "Key must be 4D [B, H, N, d]";
    CHECK_EQ(value->Dims().size(), 4) << "Value must be 4D [B, H, N, d]";

    const auto B = query->Dims()[0];
    const auto H_q = query->Dims()[1];
    const auto N = query->Dims()[2];
    const auto d = query->Dims()[3];
    const auto H_kv = key->Dims()[1];

    CHECK_EQ(key->Dims()[0], B);
    CHECK_EQ(value->Dims()[0], B);
    CHECK_EQ(key->Dims()[2], N);
    CHECK_EQ(value->Dims()[2], N);
    CHECK_EQ(key->Dims()[3], d);
    CHECK_EQ(value->Dims()[3], d);
    CHECK_EQ(H_q % H_kv, 0) << "H_q must be divisible by H_kv for GQA";

    // Compute scale
    float scale = scale_.has_value() ? scale_.value() : (1.0f / std::sqrt(static_cast<float>(d)));

    auto device = query->GetDevice().type();

    // Call the fused FlashAttention forward kernel
    // Returns: {output [B, H_q, N, d], logsumexp [B, H_q, N]}
    auto results = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionForward"}, query, key, value, scale, is_causal_, dropout_p_);

    return results;
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    // Save inputs and forward outputs needed for backward
    // output_tensors[0] = O, output_tensors[1] = L (logsumexp)
    saved_tensors_ = {input_tensors[0], input_tensors[1], input_tensors[2], output_tensors[0], output_tensors[1]};
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1) << "Expected 1 gradient output (dO)";
    CHECK_EQ(saved_tensors_.size(), 5) << "Expected 5 saved tensors: Q, K, V, O, L";

    const auto &query = saved_tensors_[0];
    const auto &key = saved_tensors_[1];
    const auto &value = saved_tensors_[2];
    const auto &output = saved_tensors_[3];
    const auto &logsumexp = saved_tensors_[4];
    const auto &grad_output = grad_outputs[0];

    const auto d = query->Dims()[3];
    float scale = scale_.has_value() ? scale_.value() : (1.0f / std::sqrt(static_cast<float>(d)));

    auto device = query->GetDevice().type();

    // Call the fused FlashAttention backward kernel
    // Returns: {dQ, dK, dV}
    auto grads = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "FlashAttentionBackward"}, grad_output, query, key, value, output, logsumexp, scale, is_causal_,
        dropout_p_);

    return grads;
}

} // namespace infini_train::autograd

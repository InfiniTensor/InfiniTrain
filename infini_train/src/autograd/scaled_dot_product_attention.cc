//------modify-start------------------------------------------
#include "infini_train/include/autograd/scaled_dot_product_attention.h"

#include <cmath>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];

    CHECK_EQ(dropout_p_, 0.0) << "dropout is not supported in current SDPA backend";

    CHECK_EQ(query->Dims().size(), 4) << "query must be 4D: (B, H, S, D)";
    const auto head_dim = query->Dims().back();
    CHECK_GT(head_dim, 0);

    const float attn_scale = scale_.has_value() ? static_cast<float>(*scale_)
                                                : static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

    auto device = query->GetDevice().type();

    auto [output, stats] = Dispatcher::Instance().Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
        {device, "ScaledDotProductAttentionForward"}, query, key, value, attn_scale, is_causal_);

    saved_stats_ = stats;
    return {output};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    CHECK_EQ(output_tensors.size(), 1);

    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];
    const auto &output = output_tensors[0];

    CHECK(saved_stats_ != nullptr) << "SDPA forward must save stats for backward";

    saved_tensors_ = {query, key, value, output, saved_stats_};
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 5);
    CHECK_EQ(grad_outputs.size(), 1);

    const auto &query = saved_tensors_[0];
    const auto &key = saved_tensors_[1];
    const auto &value = saved_tensors_[2];
    const auto &output = saved_tensors_[3];
    const auto &stats = saved_tensors_[4];
    //------modify-start------------------------------------------
    // cuDNN-frontend SDPA backward assumes dO has the same dtype/layout as O.
    // In practice, upstream may produce non-contiguous and/or FP32 grad tensors.
    // Force dtype match + contiguous to avoid incorrect memory interpretation and NaNs/corruption.
    auto grad_output = grad_outputs[0];
    if (grad_output->Dtype() != query->Dtype()) {
        grad_output = std::make_shared<Tensor>(grad_output->To(query->Dtype()));
    }
    grad_output = grad_output->Contiguous();
    //---------modify-end-----------------------------------------

    CHECK_EQ(query->Dims().size(), 4);
    const auto head_dim = query->Dims().back();
    CHECK_GT(head_dim, 0);

    const float attn_scale = scale_.has_value() ? static_cast<float>(*scale_)
                                                : static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

    auto device = query->GetDevice().type();

    auto [grad_query, grad_key, grad_value]
        = Dispatcher::Instance()
              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                  {device, "ScaledDotProductAttentionBackward"}, query, key, value, output, stats, grad_output,
                  attn_scale, is_causal_);

    return {grad_query, grad_key, grad_value};
}

} // namespace infini_train::autograd
//---------modify-end-----------------------------------------

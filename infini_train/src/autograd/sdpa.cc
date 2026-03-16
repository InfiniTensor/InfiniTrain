#include "infini_train/include/autograd/sdpa.h"

#include <cmath>
#include <limits>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"

#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

namespace {
std::shared_ptr<Tensor> RepeatHeads(const std::shared_ptr<Tensor> &x, int64_t n_rep) {
    // x: (B, H, T, D)
    if (n_rep == 1) {
        return x;
    }
    CHECK_EQ(x->Dims().size(), 4);
    return x->RepeatInterleave(n_rep, 1)->Contiguous();
}

std::shared_ptr<Tensor> MakeCausalMask(int64_t t, Device device) {
    // Returns mask of shape (1, 1, T, T) with 1s in upper triangle (excluding diagonal).
    // NOTE: construct on CPU then move to device, matching existing helper behavior.
    auto ones_cpu = std::make_shared<Tensor>(std::vector<int64_t>{t, t}, DataType::kFLOAT32);
    ones_cpu->Fill<float>(1.0f);
    auto ones = std::make_shared<Tensor>(ones_cpu->To(device));
    auto tri = std::make_shared<autograd::Triu>(1)->Apply({ones})[0];
    return tri->View({1, 1, t, t});
}
} // namespace

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &query = input_tensors[0];
    const auto &key = input_tensors[1];
    const auto &value = input_tensors[2];

    CHECK(query != nullptr);
    CHECK(key != nullptr);
    CHECK(value != nullptr);

    const auto &q_shape = query->Dims();
    const auto &k_shape = key->Dims();
    const auto &v_shape = value->Dims();
    CHECK_EQ(q_shape.size(), 4);
    CHECK_EQ(k_shape.size(), 4);
    CHECK_EQ(v_shape.size(), 4);

    const int64_t b = q_shape[0];
    const int64_t hq = q_shape[1];
    const int64_t t = q_shape[2];
    const int64_t d = q_shape[3];

    CHECK_EQ(k_shape[0], b);
    CHECK_EQ(v_shape[0], b);
    CHECK_EQ(k_shape[2], t);
    CHECK_EQ(v_shape[2], t);
    CHECK_EQ(k_shape[3], d);
    CHECK_EQ(v_shape[3], d);

    n_rep_ = 1;
    if (enable_gqa_) {
        const int64_t hk = k_shape[1];
        const int64_t hv = v_shape[1];
        CHECK_EQ(hk, hv) << "GQA expects key/value to have the same #heads";
        CHECK_EQ(hq % hk, 0) << "query heads must be divisible by kv heads when enable_gqa";
        n_rep_ = hq / hk;
    }

    scale_value_ = scale_.has_value() ? *scale_ : (1.0 / std::sqrt(static_cast<double>(d)));

    auto k_used = (enable_gqa_ ? RepeatHeads(key, n_rep_) : key);
    auto v_used = (enable_gqa_ ? RepeatHeads(value, n_rep_) : value);

    // scores: (B, H, T, T)
    auto scores = query->Matmul(k_used->Transpose(-2, -1)) * static_cast<float>(scale_value_);

    std::shared_ptr<Tensor> mask = nullptr;
    if (attn_mask_ != nullptr) {
        auto attn_mask = attn_mask_;
        if (attn_mask->GetDevice() != query->GetDevice()) {
            attn_mask = std::make_shared<Tensor>(attn_mask->To(query->GetDevice()));
        }
        mask = (attn_mask > 0);
    }

    if (is_causal_) {
        auto causal = MakeCausalMask(t, query->GetDevice());
        mask = (mask != nullptr) ? ((mask > 0) | (causal > 0)) : (causal > 0);
    }

    has_mask_ = (mask != nullptr);

    if (mask != nullptr) {
        scores = scores->MaskedFill(mask, std::numeric_limits<float>::lowest());
    }

    auto probs = std::make_shared<autograd::Softmax>(-1)->Apply({scores})[0];

    CHECK_EQ(dropout_p_, 0.0) << "ScaledDotProductAttention autograd path currently requires dropout_p=0";

    // CUDA fused path (float32 only for now); otherwise fall back to reference composition.
    std::shared_ptr<Tensor> out = nullptr;
#ifdef USE_CUDA
    if (query->GetDevice().IsCUDA() && query->Dtype() == DataType::kFLOAT32 && k_used->Dtype() == DataType::kFLOAT32
        && v_used->Dtype() == DataType::kFLOAT32) {
        out = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({Device::DeviceType::kCUDA, "SdpaForward"},
                                                                  query, k_used, v_used, attn_mask_, is_causal_,
                                                                  static_cast<float>(scale_value_));
    }
#endif
    if (out == nullptr) {
        out = probs->Matmul(v_used);
    }

    // Save for backward: query, key, value, probs, [mask]
    saved_tensors_.clear();
    saved_tensors_.push_back(query);
    saved_tensors_.push_back(key);
    saved_tensors_.push_back(value);
    saved_tensors_.push_back(probs);
    if (has_mask_) {
        saved_tensors_.push_back(mask);
    }

    return {out};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                            const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    (void)input_tensors;
    (void)output_tensors;
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_out = grad_outputs[0];

    CHECK_GE(saved_tensors_.size(), 4);
    const auto &query = saved_tensors_[0];
    const auto &key = saved_tensors_[1];
    const auto &value = saved_tensors_[2];
    const auto &probs = saved_tensors_[3];

    std::shared_ptr<Tensor> mask = nullptr;
    if (has_mask_) {
        CHECK_EQ(saved_tensors_.size(), 5);
        mask = saved_tensors_[4];
    }

    auto k_used = (enable_gqa_ ? RepeatHeads(key, n_rep_) : key);
    auto v_used = (enable_gqa_ ? RepeatHeads(value, n_rep_) : value);

    std::shared_ptr<Tensor> grad_q = nullptr;
    std::shared_ptr<Tensor> grad_k_used = nullptr;
    std::shared_ptr<Tensor> grad_v_used = nullptr;

#ifdef USE_CUDA
    if (query->GetDevice().IsCUDA() && query->Dtype() == DataType::kFLOAT32 && k_used->Dtype() == DataType::kFLOAT32
        && v_used->Dtype() == DataType::kFLOAT32 && grad_out->Dtype() == DataType::kFLOAT32) {
        auto [gq, gk, gv] = Dispatcher::Instance()
                              .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                                  {Device::DeviceType::kCUDA, "SdpaBackward"}, query, k_used, v_used, grad_out, attn_mask_,
                                  is_causal_, static_cast<float>(scale_value_));
        grad_q = gq;
        grad_k_used = gk;
        grad_v_used = gv;
    }
#endif

    if (grad_q == nullptr) {
        // Reference backward (materialize probs via composed ops)
        // grad_v_used: (B, H, T, D)
        grad_v_used = probs->Transpose(-2, -1)->Matmul(grad_out);

        // grad_probs: (B, H, T, T)
        auto grad_probs = grad_out->Matmul(v_used->Transpose(-2, -1));

        // softmax backward: grad_scores = probs * (grad_probs - sum(grad_probs * probs))
        auto dot = (grad_probs * probs)->Sum(-1, true);
        auto grad_scores = probs * (grad_probs - dot);

        if (mask != nullptr) {
            grad_scores = grad_scores->MaskedFill(mask, 0.0f);
        }

        const float s = static_cast<float>(scale_value_);

        // grad_q: (B, H, T, D)
        grad_q = grad_scores->Matmul(k_used) * s;

        // grad_k_used: (B, H, T, D)
        grad_k_used = grad_scores->Transpose(-2, -1)->Matmul(query) * s;
    }

    auto grad_v = grad_v_used;
    auto grad_k = grad_k_used;

    if (enable_gqa_ && n_rep_ != 1) {
        const auto &k_shape = key->Dims();
        const int64_t b = k_shape[0];
        const int64_t hk = k_shape[1];
        const int64_t t = k_shape[2];
        const int64_t d = k_shape[3];

        grad_k = grad_k_used->View({b, hk, n_rep_, t, d})->Sum(2, false);
        grad_v = grad_v_used->View({b, hk, n_rep_, t, d})->Sum(2, false);
    }

    return {grad_q, grad_k, grad_v};
}

} // namespace infini_train::autograd

#include "infini_train/include/autograd/scaled_dot_product_attention.h"

#include <cmath>
#include <optional>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"
#include "infini_train/src/kernels/cuda/flash_attention.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // Inputs: query, key, value, [attn_mask]
    CHECK(input_tensors.size() >= 3 && input_tensors.size() <= 4);
    auto q = input_tensors[0]->Contiguous();
    auto k = input_tensors[1]->Contiguous();
    auto v = input_tensors[2]->Contiguous();
    std::shared_ptr<Tensor> attn_mask = (input_tensors.size() == 4) ? input_tensors[3] : nullptr;

    // Check dimensions
    // Expected: (B, T, H, D) or similar, depending on kernel requirements
    // For now, assume FlashAttention kernel expects (B, T, H, D)
    
    // Calculate scale factor if not provided
    double softmax_scale = 0.0;
    if (scale_.has_value()) {
        softmax_scale = scale_.value();
    } else {
        softmax_scale = 1.0 / std::sqrt(static_cast<double>(q->Dims().back()));
    }

    // Prepare outputs
    auto output = std::make_shared<Tensor>(q->Dims(), q->Dtype(), q->GetDevice());
    
    // Prepare softmax_lse tensor for backward: (B, H, T) - assuming H is dim 2, T is dim 1
    // q shape: (B, T, H, D)
    int64_t B = q->Dims()[0];
    int64_t T = q->Dims()[1];
    int64_t H = q->Dims()[2];
    // LSE buffer needs to be float32 even for fp16 inputs
    auto softmax_lse = std::make_shared<Tensor>(std::vector<int64_t>{B, H, T}, DataType::kFLOAT32, q->GetDevice());

    // Call FlashAttention Forward Kernel
    // LOG(INFO) << "Calling FlashAttention Kernel...";
    kernels::cuda::FlashAttentionForward(*q, *k, *v, *output, *softmax_lse, 
                                         dropout_p_, softmax_scale, is_causal_, q->GetDevice());
    
    // Store LSE for SetupContext
    softmax_lse_ = softmax_lse;
    
    return {output};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    // Save inputs for backward
    // 0: q, 1: k, 2: v, 3: attn_mask (optional)
    saved_tensors_ = input_tensors; 
    
    // Save softmax_lse computed in Forward
    if (softmax_lse_) {
        saved_tensors_.push_back(softmax_lse_);
        softmax_lse_ = nullptr; // Clear reference
    }
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    auto grad_output = grad_outputs[0];
    
    // Determine if LSE is present at the end
    bool has_lse = false;
    if (!saved_tensors_.empty() && saved_tensors_.back()->Dims().size() == 3) {
        has_lse = true;
    }
    
    size_t num_inputs = saved_tensors_.size() - (has_lse ? 1 : 0);
    
    auto q = saved_tensors_[0]->Contiguous();
    auto k = saved_tensors_[1]->Contiguous();
    auto v = saved_tensors_[2]->Contiguous();
    std::shared_ptr<Tensor> attn_mask = (num_inputs >= 4) ? saved_tensors_[3] : nullptr;
    
    std::shared_ptr<Tensor> softmax_lse = has_lse ? saved_tensors_.back() : nullptr;

    auto dq = std::make_shared<Tensor>(q->Dims(), q->Dtype(), q->GetDevice());
    auto dk = std::make_shared<Tensor>(k->Dims(), k->Dtype(), k->GetDevice());
    auto dv = std::make_shared<Tensor>(v->Dims(), v->Dtype(), v->GetDevice());

    double softmax_scale = 0.0;
    if (scale_.has_value()) {
        softmax_scale = scale_.value();
    } else {
        softmax_scale = 1.0 / std::sqrt(static_cast<double>(q->Dims().back()));
    }
    
    if (softmax_lse) {
         kernels::cuda::FlashAttentionBackward(*q, *k, *v, *grad_output->Contiguous(), *dq, *dk, *dv, *softmax_lse, softmax_scale, is_causal_,
                                          q->GetDevice());
    } else {
         LOG(FATAL) << "Softmax LSE missing";
    }

    std::vector<std::shared_ptr<Tensor>> grads = {dq, dk, dv};
    if (attn_mask) {
        // Mask usually doesn't require grad
        grads.push_back(nullptr); 
    }
    
    return grads;
}

} // namespace infini_train::autograd

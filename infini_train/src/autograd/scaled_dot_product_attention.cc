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
    // (B, T, H, D)
    auto q_in = input_tensors[0]->Contiguous();
    auto k_in = input_tensors[1]->Contiguous();
    auto v_in = input_tensors[2]->Contiguous();
    std::shared_ptr<Tensor> attn_mask = (input_tensors.size() == 4) ? input_tensors[3] : nullptr;

    // Check dimensions
    // Expected: (B, T, H, D)
    // FlashAttention Kernel expects (B, H, T, D) layout.
    // Transpose inputs: (B, T, H, D) -> (B, H, T, D)
    auto q_trans = q_in->Transpose(1, 2)->Contiguous();
    auto k_trans = k_in->Transpose(1, 2)->Contiguous();
    auto v_trans = v_in->Transpose(1, 2)->Contiguous();
    
    // Calculate scale factor if not provided
    double softmax_scale = 0.0;
    if (scale_.has_value()) {
        softmax_scale = scale_.value();
    } else {
        softmax_scale = 1.0 / std::sqrt(static_cast<double>(q_in->Dims().back()));
    }

    // Prepare outputs
    // Kernel writes to (B, H, T, D)
    auto output_trans = std::make_shared<Tensor>(q_trans->Dims(), q_trans->Dtype(), q_trans->GetDevice());
    
    // Prepare softmax_lse tensor for backward: (B, H, T)
    int64_t B = q_trans->Dims()[0];
    int64_t H = q_trans->Dims()[1];
    int64_t T = q_trans->Dims()[2];
    // LSE buffer needs to be float32 even for fp16 inputs
    auto softmax_lse = std::make_shared<Tensor>(std::vector<int64_t>{B, H, T}, DataType::kFLOAT32, q_in->GetDevice());

    // Call FlashAttention Forward Kernel
    // LOG(INFO) << "Calling FlashAttention Kernel...";
    kernels::cuda::FlashAttentionForward(*q_trans, *k_trans, *v_trans, *output_trans, *softmax_lse, 
                                         dropout_p_, softmax_scale, is_causal_, q_in->GetDevice());
    
    // Transpose output back to (B, T, H, D)
    auto output = output_trans->Transpose(1, 2)->Contiguous();

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
    
    // Original Inputs (B, T, H, D)
    auto q_in = saved_tensors_[0]->Contiguous();
    auto k_in = saved_tensors_[1]->Contiguous();
    auto v_in = saved_tensors_[2]->Contiguous();
    std::shared_ptr<Tensor> attn_mask = (num_inputs >= 4) ? saved_tensors_[3] : nullptr;
    
    std::shared_ptr<Tensor> softmax_lse = has_lse ? saved_tensors_.back() : nullptr;

    // Transpose inputs for Kernel: (B, T, H, D) -> (B, H, T, D)
    auto q_trans = q_in->Transpose(1, 2)->Contiguous();
    auto k_trans = k_in->Transpose(1, 2)->Contiguous();
    auto v_trans = v_in->Transpose(1, 2)->Contiguous();
    
    // Transpose grad_output: (B, T, H, D) -> (B, H, T, D)
    auto grad_output_trans = grad_output->Transpose(1, 2)->Contiguous();

    // Prepare gradients (B, H, T, D)
    auto dq_trans = std::make_shared<Tensor>(q_trans->Dims(), q_trans->Dtype(), q_trans->GetDevice());
    auto dk_trans = std::make_shared<Tensor>(k_trans->Dims(), k_trans->Dtype(), k_trans->GetDevice());
    auto dv_trans = std::make_shared<Tensor>(v_trans->Dims(), v_trans->Dtype(), v_trans->GetDevice());

    double softmax_scale = 0.0;
    if (scale_.has_value()) {
        softmax_scale = scale_.value();
    } else {
        softmax_scale = 1.0 / std::sqrt(static_cast<double>(q_in->Dims().back()));
    }
    
    if (softmax_lse) {
         kernels::cuda::FlashAttentionBackward(*q_trans, *k_trans, *v_trans, *grad_output_trans, *dq_trans, *dk_trans, *dv_trans, *softmax_lse, softmax_scale, is_causal_,
                                          q_in->GetDevice());
    } else {
         LOG(FATAL) << "Softmax LSE missing";
    }

    // Transpose gradients back to (B, T, H, D)
    auto dq = dq_trans->Transpose(1, 2)->Contiguous();
    auto dk = dk_trans->Transpose(1, 2)->Contiguous();
    auto dv = dv_trans->Transpose(1, 2)->Contiguous();

    std::vector<std::shared_ptr<Tensor>> grads = {dq, dk, dv};
    if (attn_mask) {
        // Mask usually doesn't require grad
        grads.push_back(nullptr); 
    }
    
    return grads;
}

} // namespace infini_train::autograd

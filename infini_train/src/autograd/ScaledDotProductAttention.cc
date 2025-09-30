#include "infini_train/include/autograd/ScaledDotProductAttention.h"
#include "glog/logging.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK(input_tensors.size() == 3 || input_tensors.size() == 4)
        << "ScaledDotProductAttention expects {Q,K,V[,attn_mask]}";

    const std::shared_ptr<Tensor> &q = input_tensors[0];
    const std::shared_ptr<Tensor> &k = input_tensors[1];
    const std::shared_ptr<Tensor> &v = input_tensors[2];
    const std::shared_ptr<Tensor> mask = (input_tensors.size() == 4 ? input_tensors[3] : nullptr);

    auto device = q->GetDevice()->Type();

    auto kernel = Dispatcher::Instance().GetKernel({device, "ScaledDotProductAttentionForward"});

    std::shared_ptr<Tensor> y = kernel.template Call<std::shared_ptr<Tensor>>(q, k, v, mask,
                                                                              /*is_causal=*/is_causal_,
                                                                              /*dropout_p=*/dropout_p_,
                                                                              /*scale_has=*/scale_.has_value(),
                                                                              /*scale_val=*/scale_.value_or(0.0),
                                                                              /*enable_gqa=*/enable_gqa_);

    return {y};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    (void)output_tensors;
    saved_tensors_.clear();
    for (const auto &t : input_tensors) { saved_tensors_.push_back(t); } // Q,K,V,[mask]
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const std::shared_ptr<Tensor> &dY = grad_outputs[0];

    CHECK(saved_tensors_.size() == 3 || saved_tensors_.size() == 4);
    const std::shared_ptr<Tensor> &Q = saved_tensors_[0];
    const std::shared_ptr<Tensor> &K = saved_tensors_[1];
    const std::shared_ptr<Tensor> &V = saved_tensors_[2];
    const std::shared_ptr<Tensor> mask = (saved_tensors_.size() == 4 ? saved_tensors_[3] : nullptr);

    auto device = dY->GetDevice()->Type();

    auto kernel = Dispatcher::Instance().GetKernel({device, "ScaledDotProductAttentionBackward"});

    std::vector<std::shared_ptr<Tensor>> grads
        = kernel.template Call<std::vector<std::shared_ptr<Tensor>>>(dY, Q, K, V, mask,
                                                                     /*is_causal=*/is_causal_,
                                                                     /*dropout_p=*/dropout_p_,
                                                                     /*scale_has=*/scale_.has_value(),
                                                                     /*scale_val=*/scale_.value_or(0.0),
                                                                     /*enable_gqa=*/enable_gqa_);
    CHECK_EQ(grads.size(), 3);
    return grads; // {dQ, dK, dV}
}

} // namespace infini_train::autograd

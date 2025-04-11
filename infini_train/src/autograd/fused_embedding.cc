#include "infini_train/include/autograd/fused_embedding.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/kernels/cpu/fused_embedding.h"
#include "infini_train/include/tensor.h"

#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/fused_embedding.h"
#endif

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> FusedEmbedding::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &input = input_tensors[0];
    const auto &tok_emb = input_tensors[1];
    const auto &pos_emb = input_tensors[2];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::FusedEmbeddingForward(input, tok_emb, pos_emb);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::FusedEmbeddingForward(input, tok_emb, pos_emb);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void FusedEmbedding::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                             const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &tok_emb = input_tensors[1];
    const auto &pos_emb = input_tensors[2];
    saved_tensors_ = {input, tok_emb, pos_emb};
}

std::vector<std::shared_ptr<Tensor>> FusedEmbedding::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 3);
    const auto &input = saved_tensors_[0];
    const auto &tok_emb = saved_tensors_[1];
    const auto &pos_emb = saved_tensors_[2];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        auto [grad_input, grad_tok_emb, grad_pos_emb]
            = kernels::cpu::FusedEmbeddingBackward(input, tok_emb, pos_emb, grad_output);
        return {grad_input, grad_tok_emb, grad_pos_emb};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto [grad_input, grad_tok_emb, grad_pos_emb]
            = kernels::cuda::FusedEmbeddingBackward(input, tok_emb, pos_emb, grad_output);
        return {grad_input, grad_tok_emb, grad_pos_emb};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {};
}
} // namespace infini_train::autograd

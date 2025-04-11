#include "infini_train/include/autograd/embedding.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/kernels/cpu/embedding.h"
#include "infini_train/include/tensor.h"

#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/embedding.h"
#endif

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Embedding::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::EmbeddingForward(input, weight);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::EmbeddingForward(input, weight);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Embedding::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                             const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    saved_tensors_ = {input, weight};
}

std::vector<std::shared_ptr<Tensor>> Embedding::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input = saved_tensors_[0];
    const auto &weight = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        auto grad_weight = kernels::cpu::EmbeddingBackward(input, weight, grad_output);
        return {grad_weight};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto grad_weight = kernels::cuda::EmbeddingBackward(input, weight, grad_output);
        return {grad_weight};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {};
}
} // namespace infini_train::autograd

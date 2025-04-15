#include "infini_train/include/autograd/normalization.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/kernels/cpu/layernorm.h"
#include "infini_train/include/tensor.h"

#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/layernorm.h"
#endif

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> LayerNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors[2];

    std::shared_ptr<Tensor> output = nullptr;
    std::shared_ptr<Tensor> mean = nullptr;
    std::shared_ptr<Tensor> rstd = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::LayerNormForward(input, weight, bias, mean, rstd, eps_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::LayerNormForward(input, weight, bias, mean, rstd, eps_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    saved_tensors_ = {mean, rstd};
    return {output};
}

void LayerNorm::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                             const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors[2];
    saved_tensors_.insert(saved_tensors_.begin(), {input, weight, bias});
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 5);
    const auto &input = saved_tensors_[0];
    const auto &weight = saved_tensors_[1];
    const auto &bias = saved_tensors_[2];
    const auto &mean = saved_tensors_[3];
    const auto &rstd = saved_tensors_[4];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        auto [grad_input, grad_weight, grad_bias]
            = kernels::cpu::LayerNormBackward(input, weight, bias, mean, rstd, grad_output);
        return {grad_input, grad_weight, grad_bias};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto [grad_input, grad_weight, grad_bias]
            = kernels::cuda::LayerNormBackward(input, weight, bias, mean, rstd, grad_output);
        return {grad_input, grad_weight, grad_bias};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {};
}
} // namespace infini_train::autograd

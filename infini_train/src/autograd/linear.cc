#include "infini_train/include/autograd/linear.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/kernels/cpu/linear.h"
#include "infini_train/include/tensor.h"

#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/linear.h"
#endif

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Linear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors[2];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::LinearForward(input, weight, true, bias);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::LinearForward(input, weight, bias);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Linear::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    saved_tensors_ = {input, weight};
    const auto &bias = input_tensors[2];
    out_features_ = bias->Dims()[0];
}

std::vector<std::shared_ptr<Tensor>> Linear::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input = saved_tensors_[0];
    const auto &weight = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        auto [grad_input, grad_weight, grad_bias]
            = kernels::cpu::LinearBackward(input, weight, true, out_features_, grad_output);
        return {grad_input, grad_weight, grad_bias};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto [grad_input, grad_weight, grad_bias]
            = kernels::cuda::LinearBackward(input, weight, out_features_, grad_output, true);
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

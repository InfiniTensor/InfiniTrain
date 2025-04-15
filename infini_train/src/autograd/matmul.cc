#include "infini_train/include/autograd/matmul.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/kernels/cpu/linear.h"
#include "infini_train/include/tensor.h"

#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/linear.h"
#endif

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Matmul::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input1->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::MatmulForward(input1, input2);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::LinearForward(input1, input2, nullptr);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input1->GetDevice().Type());
        break;
    }
    return {output};
}

void Matmul::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];
    const auto &output = output_tensors[0];
    saved_tensors_ = {input1, input2};
    out_features_ = output->Dims()[0];
}

std::vector<std::shared_ptr<Tensor>> Matmul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 2);
    const auto &input1 = saved_tensors_[0];
    const auto &input2 = saved_tensors_[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    switch (input1->GetDevice().Type()) {
    case DeviceType::kCPU: {
        auto [grad_input1, grad_input2] = kernels::cpu::MatmulBackward(input1, input2, grad_output);
        return {grad_input1, grad_input2};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto [grad_input1, grad_input2, _]
            = kernels::cuda::LinearBackward(input1, input2, out_features_, grad_output, false);
        return {grad_input1, grad_input2};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input1->GetDevice().Type());
        break;
    }
    return {};
}
} // namespace infini_train::autograd

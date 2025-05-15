#include "infini_train/include/autograd/misc.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/kernels/cpu/no_op.h"
#include "infini_train/include/kernels/cpu/slice.h"
#include "infini_train/include/kernels/cpu/split.h"
#include "infini_train/include/kernels/cpu/stack.h"
#include "infini_train/include/tensor.h"
#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/no_op.h"
#include "infini_train/include/kernels/cuda/slice.h"
#include "infini_train/include/kernels/cuda/split.h"
#include "infini_train/include/kernels/cuda/stack.h"
#endif

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Split::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        auto output_tensors = kernels::cpu::SplitForward(input, split_size_, dim_);
        return output_tensors;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto output_tensors = kernels::cuda::SplitForward(input, split_size_, dim_);
        return output_tensors;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {};
}

void Split::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Split::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto device = grad_outputs[0]->GetDevice();

    switch (device.Type()) {
    case DeviceType::kCPU: {
        auto grad_input = kernels::cpu::SplitBackward(input_dims_, split_size_, dim_, grad_outputs);
        return {grad_input};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto grad_input = kernels::cuda::SplitBackward(input_dims_, split_size_, dim_, grad_outputs);
        return {grad_input};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
        break;
    }
    return {};
}

std::vector<std::shared_ptr<Tensor>> NoOp::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::NoOpForward(input, output_dims_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::NoOpForward(input, output_dims_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void NoOp::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> NoOp::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    const auto device = grad_outputs[0]->GetDevice();
    switch (device.Type()) {
    case DeviceType::kCPU: {
        auto grad_input = kernels::cpu::NoOpBackward(input_dims_, grad_output);
        return {grad_input};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto grad_input = kernels::cuda::NoOpBackward(input_dims_, grad_output);
        return {grad_input};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
        break;
    }
    return {};
}

std::vector<std::shared_ptr<Tensor>> Slice::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::SliceForward(input, starts_, ends_, steps_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::SliceForward(input, starts_, ends_, steps_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Slice::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &) {
    // FIXME(dcj): only input's dim need to be saved
    const auto &input = input_tensors[0];
    saved_tensors_ = {input};
}

std::vector<std::shared_ptr<Tensor>> Slice::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &input = saved_tensors_[0];
    const auto &grad_output = grad_outputs[0];
    std::shared_ptr<Tensor> grad_input = nullptr;

    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::SliceBackward(grad_output, input, starts_, ends_, steps_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::SliceBackward(grad_output, input, starts_, ends_, steps_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Stack::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 2);
    const auto device = input_tensors[0]->GetDevice();
    std::shared_ptr<Tensor> output = nullptr;

    switch (device.Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::StackForward(input_tensors, dim_);
        return {output};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::StackForward(input_tensors, dim_);
        return {output};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
        break;
    }
    return {output};
}

void Stack::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Stack::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        auto grad_inputs = kernels::cpu::StackBackward(input_dims_, dim_, grad_output);
        return grad_inputs;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto grad_inputs = kernels::cuda::StackBackward(input_dims_, dim_, grad_output);
        return grad_inputs;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {};
}
} // namespace infini_train::autograd

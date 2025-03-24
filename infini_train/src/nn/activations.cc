#include "infini_train/include/nn/activations.h"

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Sigmoid::Sigmoid() : sigmoid_op_(std::make_unique<ops::Sigmoid>()) {}

std::vector<std::shared_ptr<Tensor>> Sigmoid::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return sigmoid_op_->Forward(input_tensors);
}

void Sigmoid::ToImpl(Device device) {
    if (device_ == device) {
        return;
    }

    switch (device.Type()) {
    case DeviceType::kCPU:
        sigmoid_op_ = std::make_unique<ops::Sigmoid>();
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        sigmoid_op_ = std::make_unique<ops::CUDASigmoid>();
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
    }
}
} // namespace infini_train::nn

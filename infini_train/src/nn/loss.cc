#include "infini_train/include/nn/loss.h"

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
CrossEntropyLoss::CrossEntropyLoss() : cross_entropy_op_(std::make_unique<ops::CrossEntropy>()) {}

std::vector<std::shared_ptr<Tensor>>
CrossEntropyLoss::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return cross_entropy_op_->Forward(input_tensors);
}

void CrossEntropyLoss::ToImpl(Device device) {
    if (device_ == device) {
        return;
    }

    switch (device.Type()) {
    case DeviceType::kCPU:
        cross_entropy_op_ = std::make_unique<ops::CrossEntropy>();
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        cross_entropy_op_ = std::make_unique<ops::CUDACrossEntropy>();
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
    }
}
} // namespace infini_train::nn

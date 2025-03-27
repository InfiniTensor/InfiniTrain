#include "infini_train/include/nn/linear.h"

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
constexpr char kParamWeightName[] = "weight";
constexpr char kParamBiasName[] = "bias";
} // namespace

Linear::Linear(int64_t in_dim, int64_t out_dim) {
    AddNamedParameter(kParamWeightName, {in_dim, out_dim}, DataType::kFLOAT32);
    AddNamedParameter(kParamBiasName, {out_dim}, DataType::kFLOAT32);
    linear_op_ = std::make_unique<ops::Linear>(GetParameter(kParamWeightName), GetParameter(kParamBiasName));
}

std::vector<std::shared_ptr<Tensor>> Linear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return linear_op_->Forward(input_tensors);
}

void Linear::ToImpl(Device device) {
    if (device_ == device) {
        return;
    }

    auto *w = GetParameter(kParamWeightName);
    auto *b = GetParameter(kParamBiasName);
    switch (device.Type()) {
    case DeviceType::kCPU:
        linear_op_ = std::make_unique<ops::Linear>(w, b);
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        linear_op_ = std::make_unique<ops::CUDALinear>(w, b);
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
    }
}
} // namespace infini_train::nn

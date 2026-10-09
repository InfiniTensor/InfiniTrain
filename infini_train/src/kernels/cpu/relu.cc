#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> ReLUForward(const std::shared_ptr<Tensor> &x) {
    auto y = std::make_shared<Tensor>(x->Dims(), x->Dtype(), x->GetDevice());
    const auto xp = static_cast<const float *>(x->DataPtr());
    auto yp = static_cast<float *>(y->DataPtr());
    for (size_t i = 0; i < x->NumElements(); ++i) { yp[i] = xp[i] <= 0.0f ? 0.0f : xp[i]; }
    return y;
}

std::shared_ptr<Tensor> ReLUBackward(const std::shared_ptr<Tensor> &x, const std::shared_ptr<Tensor> &dy) {
    auto dx = std::make_shared<Tensor>(x->Dims(), x->Dtype(), x->GetDevice());
    const auto xp = static_cast<const float *>(x->DataPtr());
    const auto gp = static_cast<const float *>(dy->DataPtr());
    auto p = static_cast<float *>(dx->DataPtr());
    for (size_t i = 0; i < x->NumElements(); ++i) { p[i] = xp[i] <= 0.0f ? 0.0f : gp[i]; }
    return dx;
}
} // namespace infini_train::kernels::cpu

REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, ReLUForward, infini_train::kernels::cpu::ReLUForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, ReLUBackward, infini_train::kernels::cpu::ReLUBackward)

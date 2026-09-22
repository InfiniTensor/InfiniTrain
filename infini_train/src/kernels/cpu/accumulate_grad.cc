#include <cstddef>
#include <memory>
#include <vector>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

template <typename T> void ScaleInplaceTyped(const std::shared_ptr<Tensor> &tensor, float scale) {
    auto *data = static_cast<T *>(tensor->DataPtr());
#pragma omp parallel for
    for (size_t i = 0; i < tensor->NumElements(); ++i) { data[i] = T(static_cast<float>(data[i]) * scale); }
}

void ScaleInplace(const std::shared_ptr<Tensor> &tensor, float scale) {
    switch (tensor->Dtype()) {
    case DataType::kFLOAT16:
        ScaleInplaceTyped<FP16>(tensor, scale);
        return;
    case DataType::kBFLOAT16:
        ScaleInplaceTyped<BF16>(tensor, scale);
        return;
    case DataType::kFLOAT32:
        ScaleInplaceTyped<float>(tensor, scale);
        return;
    case DataType::kFLOAT64:
        ScaleInplaceTyped<double>(tensor, scale);
        return;
    default:
        LOG(FATAL) << "ScaleInplace only supports floating point gradients.";
    }
}

template <typename T>
void ScaleInplaceMultiTyped(const std::vector<std::shared_ptr<Tensor>> &tensors, float scale) {
    size_t total = 0;
    for (const auto &tensor : tensors) {
        if (tensor && tensor->NumElements() != 0) {
            total += tensor->NumElements();
        }
    }
#pragma omp parallel for
    for (int64_t flat = 0; flat < static_cast<int64_t>(total); ++flat) {
        size_t offset = static_cast<size_t>(flat);
        for (const auto &tensor : tensors) {
            if (!tensor || tensor->NumElements() == 0 || offset >= tensor->NumElements()) {
                if (tensor && tensor->NumElements() != 0) {
                    offset -= tensor->NumElements();
                }
                continue;
            }
            auto *data = static_cast<T *>(tensor->DataPtr());
            data[offset] = T(static_cast<float>(data[offset]) * scale);
            break;
        }
    }
}

void ScaleInplaceMulti(std::vector<std::shared_ptr<Tensor>> tensors, float scale) {
    if (tensors.empty() || scale == 1.0f) {
        return;
    }
    switch (tensors.front()->Dtype()) {
    case DataType::kFLOAT16:
        ScaleInplaceMultiTyped<FP16>(tensors, scale);
        return;
    case DataType::kBFLOAT16:
        ScaleInplaceMultiTyped<BF16>(tensors, scale);
        return;
    case DataType::kFLOAT32:
        ScaleInplaceMultiTyped<float>(tensors, scale);
        return;
    case DataType::kFLOAT64:
        ScaleInplaceMultiTyped<double>(tensors, scale);
        return;
    default:
        LOG(FATAL) << "ScaleInplaceMulti only supports floating point gradients.";
    }
}
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    const float *grad_data = static_cast<const float *>(grad->DataPtr());
    float *m_data = static_cast<float *>(m->DataPtr());
    float *v_data = static_cast<float *>(v->DataPtr());
    float *param_data = static_cast<float *>(param->DataPtr());

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

#pragma omp parallel for
    for (size_t idx = 0; idx < grad->NumElements(); ++idx) {
        m_data[idx] = beta1 * m_data[idx] + (1 - beta1) * grad_data[idx];
        v_data[idx] = beta2 * v_data[idx] + (1 - beta2) * grad_data[idx] * grad_data[idx];

        const float m_hat = m_data[idx] / bias_correction_m;
        const float v_hat = v_data[idx] / bias_correction_v;

        param_data[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(ScaleInplace)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(ScaleInplaceMulti)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL

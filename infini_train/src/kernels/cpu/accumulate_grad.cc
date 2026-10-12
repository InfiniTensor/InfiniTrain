#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void L2NormAccumulate(const std::shared_ptr<Tensor> &gradient, const std::shared_ptr<Tensor> &total_square) {
    CHECK(gradient->Dtype() == DataType::kFLOAT32);
    CHECK(total_square->Dtype() == DataType::kFLOAT32);
    CHECK_EQ(total_square->NumElements(), 1);
    const auto *gradient_data = static_cast<const float *>(gradient->DataPtr());
    float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        sum += gradient_data[idx] * gradient_data[idx];
    }
    *static_cast<float *>(total_square->DataPtr()) += sum;
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

void AdamWFP32MasterStep(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                         const std::shared_ptr<Tensor> &master, const std::shared_ptr<Tensor> &m,
                         const std::shared_ptr<Tensor> &v, float learning_rate, float beta1, float beta2,
                         float eps, int64_t t, float gradient_scale, float weight_decay) {
    CHECK(grad->Dtype() == DataType::kFLOAT32 && param->Dtype() == DataType::kFLOAT32);
    const auto *g = static_cast<const float *>(grad->DataPtr());
    auto *p = static_cast<float *>(param->DataPtr());
    auto *master_data = static_cast<float *>(master->DataPtr());
    auto *m_data = static_cast<float *>(m->DataPtr());
    auto *v_data = static_cast<float *>(v->DataPtr());
    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);
#pragma omp parallel for
    for (size_t idx = 0; idx < grad->NumElements(); ++idx) {
        const float scaled_grad = g[idx] * gradient_scale;
        float value = master_data[idx];
        if (weight_decay != 0.0f) {
            value += value * (-learning_rate * weight_decay);
        }
        const float next_m = beta1 * m_data[idx] + (1.0f - beta1) * scaled_grad;
        const float next_v = beta2 * v_data[idx] + (1.0f - beta2) * scaled_grad * scaled_grad;
        m_data[idx] = next_m;
        v_data[idx] = next_v;
        value -= learning_rate * (next_m / bias_correction_m) /
                 (std::sqrt(next_v / bias_correction_v) + eps);
        master_data[idx] = value;
        p[idx] = value;
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(L2NormAccumulate)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamWFP32MasterStep)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL

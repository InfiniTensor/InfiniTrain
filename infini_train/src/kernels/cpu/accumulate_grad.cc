#include <stdint.h>

#include <cmath>
#include <cstddef>
#include <memory>

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate,
                    const std::shared_ptr<Tensor> &tensor) {
  for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
    static_cast<float *>(tensor->DataPtr())[idx] +=
        rate * static_cast<const float *>(gradient->DataPtr())[idx];
  }
}

void SGDAccumulateGrad(const std::shared_ptr<Tensor> &grad,
                       const std::shared_ptr<Tensor> &param,
                       const std::shared_ptr<Tensor> &momentum_buffer,
                       float learning_rate, float momentum, float weight_decay,
                       bool is_first_step) {
  const float *grad_data = static_cast<const float *>(grad->DataPtr());
  float *param_data = static_cast<float *>(param->DataPtr());
  float *buf_data = momentum_buffer
                        ? static_cast<float *>(momentum_buffer->DataPtr())
                        : nullptr;

#pragma omp parallel for
  for (size_t idx = 0; idx < grad->NumElements(); ++idx) {
    float d_p = grad_data[idx];
    if (weight_decay != 0.0f) {
      d_p += weight_decay * param_data[idx];
    }
    if (momentum != 0.0f && buf_data != nullptr) {
      if (is_first_step) {
        buf_data[idx] = d_p;
      } else {
        buf_data[idx] = momentum * buf_data[idx] + d_p;
      }
      d_p = buf_data[idx];
    }
    param_data[idx] -= learning_rate * d_p;
  }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad,
                        const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m,
                        const std::shared_ptr<Tensor> &v, float learning_rate,
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
    v_data[idx] =
        beta2 * v_data[idx] + (1 - beta2) * grad_data[idx] * grad_data[idx];

    const float m_hat = m_data[idx] / bias_correction_m;
    const float v_hat = v_data[idx] / bias_correction_v;

    param_data[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
  }
}

}  // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)               \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, \
                  infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(SGDAccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL

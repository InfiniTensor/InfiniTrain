#include <cmath>
#include <memory>

#include "infini_train/include/common/dcu/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {

template <typename T>
__global__ void AccumulateGradKernel(const T *grad_ptr, float rate, T *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += common::dcu::Mul(grad_ptr[idx], common::dcu::Cast<T>(rate));
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    auto device = tensor->GetDevice();
    const auto &hip_stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->hip_stream();

    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(
        gradient->Dtype(),
        [=]<typename T>() {
            AccumulateGradKernel<<<num_blocks, threads_per_block, 0, hip_stream>>>(
                static_cast<const T *>(gradient->DataPtr()), rate, static_cast<T *>(tensor->DataPtr()), num_elements);
        },
        "HIP AccumulateGrad");
}

template <typename T>
__global__ void AdamAccumulateGradKernel(const T *grad_data, T *param_data, size_t num_elements, T *m_data, T *v_data,
                                         float learning_rate, float beta1, float beta2, float eps,
                                         const float bias_correction_m, const float bias_correction_v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        m_data[idx] = common::dcu::Fma(common::dcu::Cast<T>(beta1), m_data[idx],
                                        common::dcu::Cast<T>(1 - beta1) * grad_data[idx]);
        v_data[idx] = common::dcu::Fma(common::dcu::Cast<T>(beta2), v_data[idx],
                                        common::dcu::Cast<T>(1 - beta2) * grad_data[idx] * grad_data[idx]);

        const float m_hat = common::dcu::Cast<float>(m_data[idx]) / bias_correction_m;
        const float v_hat = common::dcu::Cast<float>(v_data[idx]) / bias_correction_v;

        param_data[idx] = common::dcu::Sub(
            param_data[idx], common::dcu::Cast<T>(learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps)));
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    size_t num_elements = grad->NumElements();

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    auto device = grad->GetDevice();
    const auto &hip_stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->hip_stream();

    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, hip_stream>>>(
                static_cast<const T *>(grad->DataPtr()), static_cast<T *>(param->DataPtr()), num_elements,
                static_cast<T *>(m->DataPtr()), static_cast<T *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v);
        },
        "HIP AdamAccumulateGrad");
}
} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, kernel_name, infini_train::kernels::dcu::kernel_name)

REGISTER_HIP_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_HIP_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_HIP_ACCUMULATE_GRAD_KERNEL

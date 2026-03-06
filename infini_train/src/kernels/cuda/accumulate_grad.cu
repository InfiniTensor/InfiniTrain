#include <cmath>
#include <memory>

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/cuda/cuda_dispatch.h"
#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void AccumulateGradKernel(const T *grad_ptr, float rate, T *tensor_ptr, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx]
            = common::cuda::Add(tensor_ptr[idx], common::cuda::Mul(grad_ptr[idx], common::cuda::Cast<T>(rate)));
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    const size_t num_elements = gradient->NumElements();

    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    auto device = tensor->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    infini_train::core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        gradient->Dtype(),
        [=]<typename T>() {
            AccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(gradient->DataPtr()), rate, static_cast<T *>(tensor->DataPtr()), num_elements);
        },
        "CUDA AccumulateGrad");
}

template <typename T>
__global__ void AdamAccumulateGradKernel(const T *grad_data, T *param_data, size_t num_elements, T *m_data, T *v_data,
                                         float learning_rate, float beta1, float beta2, float eps,
                                         float bias_correction_m, float bias_correction_v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        const T beta1_t = common::cuda::Cast<T>(beta1);
        const T beta2_t = common::cuda::Cast<T>(beta2);
        const T one_minus_beta1_t = common::cuda::Cast<T>(1.0f - beta1);
        const T one_minus_beta2_t = common::cuda::Cast<T>(1.0f - beta2);

        m_data[idx] = common::cuda::Fma(beta1_t, m_data[idx], common::cuda::Mul(one_minus_beta1_t, grad_data[idx]));

        v_data[idx] = common::cuda::Fma(
            beta2_t, v_data[idx],
            common::cuda::Mul(one_minus_beta2_t, common::cuda::Mul(grad_data[idx], grad_data[idx])));

        const float m_hat = common::cuda::Cast<float>(m_data[idx]) / bias_correction_m;
        const float v_hat = common::cuda::Cast<float>(v_data[idx]) / bias_correction_v;

        const float update = learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps);

        param_data[idx] = common::cuda::Sub(param_data[idx], common::cuda::Cast<T>(update));
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    const size_t num_elements = grad->NumElements();

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    auto device = grad->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    infini_train::core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(grad->DataPtr()), static_cast<T *>(param->DataPtr()), num_elements,
                static_cast<T *>(m->DataPtr()), static_cast<T *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v);
        },
        "CUDA AdamAccumulateGrad");
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL

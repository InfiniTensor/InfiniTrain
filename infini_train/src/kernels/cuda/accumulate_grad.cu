#include <cmath>
#include <memory>

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void AccumulateGradKernel(const T *grad_ptr, float rate, T *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += common::cuda::Mul(grad_ptr[idx], common::cuda::Cast<T>(rate));
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(tensor->GetDevice());

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        gradient->Dtype(),
        [=]<typename T>() {
            AccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<const T *>(gradient->DataPtr()), rate, static_cast<T *>(tensor->DataPtr()), num_elements);
        },
        "CUDA AccumulateGrad");
}

template <typename T>
__global__ void AdamAccumulateGradKernel(const T *grad_data, T *param_data, size_t num_elements, T *m_data, T *v_data,
                                         float learning_rate, float beta1, float beta2, float eps,
                                         const float bias_correction_m, const float bias_correction_v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        m_data[idx] = common::cuda::Fma(common::cuda::Cast<T>(beta1), m_data[idx],
                                        common::cuda::Cast<T>(1 - beta1) * grad_data[idx]);
        v_data[idx] = common::cuda::Fma(common::cuda::Cast<T>(beta2), v_data[idx],
                                        common::cuda::Cast<T>(1 - beta2) * grad_data[idx] * grad_data[idx]);

        const float m_hat = common::cuda::Cast<float>(m_data[idx]) / bias_correction_m;
        const float v_hat = common::cuda::Cast<float>(v_data[idx]) / bias_correction_v;

        param_data[idx] = common::cuda::Sub(
            param_data[idx], common::cuda::Cast<T>(learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps)));
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
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad->GetDevice());

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<const T *>(grad->DataPtr()), static_cast<T *>(param->DataPtr()), num_elements,
                static_cast<T *>(m->DataPtr()), static_cast<T *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v);
        },
        "CUDA AdamAccumulateGrad");
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL

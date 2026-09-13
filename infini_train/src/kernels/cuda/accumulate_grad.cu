#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

// Loop-invariant Adam scalars. beta1/beta2 are cast to T once per thread instead of once per element;
// the cast is deterministic, so the values fed to the math are unchanged.
template <typename T> struct AdamParams {
    T beta1;
    T one_minus_beta1;
    T beta2;
    T one_minus_beta2;
    float learning_rate;
    float bias_correction_m;
    float bias_correction_v;
    float eps;
};

template <typename T>
__device__ __forceinline__ AdamParams<T> MakeAdamParams(float learning_rate, float beta1, float beta2, float eps,
                                                        float bias_correction_m, float bias_correction_v) {
    return AdamParams<T>{common::cuda::Cast<T>(beta1), common::cuda::Cast<T>(1 - beta1), common::cuda::Cast<T>(beta2),
                         common::cuda::Cast<T>(1 - beta2), learning_rate, bias_correction_m, bias_correction_v, eps};
}

// One Adam update step for a single element: m/v EMAs, bias correction, then the param update.
// Shared by the plain and shadow-weight kernels so both paths stay bit-identical.
template <typename T>
__device__ __forceinline__ void AdamUpdateElement(const T &grad, T &param, T &m, T &v, const AdamParams<T> &p) {
    m = common::cuda::Fma(p.beta1, m, p.one_minus_beta1 * grad);
    v = common::cuda::Fma(p.beta2, v, p.one_minus_beta2 * grad * grad);

    const float m_hat = common::cuda::Cast<float>(m) / p.bias_correction_m;
    const float v_hat = common::cuda::Cast<float>(v) / p.bias_correction_v;
    const float rcp = __frcp_rn(__fsqrt_rn(v_hat) + p.eps);

    if constexpr (std::is_same_v<T, float>) {
        // Spell out the contraction of "param - (lr * m_hat) * rcp" instead of leaving it to nvcc, so
        // the rounded result does not depend on whether the compiler fuses it into a single FFMA.
        param = std::fma(-(p.learning_rate * m_hat), rcp, param);
    } else {
        param = common::cuda::Sub(param, common::cuda::Cast<T>((p.learning_rate * m_hat) * rcp));
    }
}

} // namespace

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

    auto device = tensor->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
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
                                         const float bias_correction_m, const float bias_correction_v) {
    const AdamParams<T> p = MakeAdamParams<T>(learning_rate, beta1, beta2, eps, bias_correction_m, bias_correction_v);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        AdamUpdateElement<T>(grad_data[idx], param_data[idx], m_data[idx], v_data[idx], p);
    }
}

template <typename T, typename TShadow>
__global__ void AdamAccumulateGradShadowKernel(const T *grad_data, T *param_data, TShadow *shadow_data,
                                               size_t num_elements, T *m_data, T *v_data, float learning_rate,
                                               float beta1, float beta2, float eps, const float bias_correction_m,
                                               const float bias_correction_v) {
    const AdamParams<T> p = MakeAdamParams<T>(learning_rate, beta1, beta2, eps, bias_correction_m, bias_correction_v);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        AdamUpdateElement<T>(grad_data[idx], param_data[idx], m_data[idx], v_data[idx], p);
        shadow_data[idx] = common::cuda::Cast<TShadow>(param_data[idx]);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    size_t num_elements = grad->NumElements();

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    int threads_per_block = 256;

    auto device = grad->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            const T *grad_ptr = static_cast<const T *>(grad->DataPtr());
            T *param_ptr = static_cast<T *>(param->DataPtr());
            T *m_ptr = static_cast<T *>(m->DataPtr());
            T *v_ptr = static_cast<T *>(v->DataPtr());

            int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
            AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                grad_ptr, param_ptr, num_elements, m_ptr, v_ptr, learning_rate, beta1, beta2, eps, bias_correction_m,
                bias_correction_v);
        },
        "CUDA AdamAccumulateGrad");
}

void AdamAccumulateGradShadow(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                              const std::shared_ptr<Tensor> &shadow, const std::shared_ptr<Tensor> &m,
                              const std::shared_ptr<Tensor> &v, float learning_rate, float beta1, float beta2,
                              float eps, int64_t t) {
    size_t num_elements = grad->NumElements();

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    int threads_per_block = 256;

    auto device = grad->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
                shadow->Dtype(),
                [=]<typename TShadow>() {
                    const T *grad_ptr = static_cast<const T *>(grad->DataPtr());
                    T *param_ptr = static_cast<T *>(param->DataPtr());
                    TShadow *shadow_ptr = static_cast<TShadow *>(shadow->DataPtr());
                    T *m_ptr = static_cast<T *>(m->DataPtr());
                    T *v_ptr = static_cast<T *>(v->DataPtr());

                    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
                    AdamAccumulateGradShadowKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                        grad_ptr, param_ptr, shadow_ptr, num_elements, m_ptr, v_ptr, learning_rate, beta1, beta2, eps,
                        bias_correction_m, bias_correction_v);
                },
                "CUDA AdamAccumulateGradShadow");
        },
        "CUDA AdamAccumulateGradShadow");
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGradShadow)
#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL

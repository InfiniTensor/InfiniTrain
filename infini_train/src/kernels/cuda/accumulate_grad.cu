#include <algorithm>
#include <cmath>
#include <memory>

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

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
__global__ void L2NormAccumulateKernel(const T *__restrict__ gradient, size_t num_elements,
                                       float *__restrict__ total_square) {
    __shared__ float partial[256];
    float thread_sum = 0.0f;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < num_elements;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float value = common::cuda::Cast<float>(gradient[idx]);
        thread_sum += value * value;
    }
    partial[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(total_square, partial[0]);
    }
}

void L2NormAccumulate(const std::shared_ptr<Tensor> &gradient,
                      const std::shared_ptr<Tensor> &total_square) {
    CHECK(total_square->Dtype() == DataType::kFLOAT32);
    CHECK_EQ(total_square->NumElements(), 1);
    CHECK(gradient->GetDevice() == total_square->GetDevice());
    constexpr int threads = 256;
    const int blocks = std::min<int64_t>(256, (gradient->NumElements() + threads - 1) / threads);
    const auto device = gradient->GetDevice();
    const auto stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                            infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                            ->cuda_stream();
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        gradient->Dtype(),
        [=]<typename T>() {
            L2NormAccumulateKernel<<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(gradient->DataPtr()), gradient->NumElements(),
                static_cast<float *>(total_square->DataPtr()));
        },
        "CUDA L2NormAccumulate");
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

    auto device = grad->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(grad->DataPtr()), static_cast<T *>(param->DataPtr()), num_elements,
                static_cast<T *>(m->DataPtr()), static_cast<T *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v);
        },
        "CUDA AdamAccumulateGrad");
}

template <typename TGrad, typename TParam>
__global__ void AdamWFP32MasterStepKernel(const TGrad *__restrict__ grad, TParam *__restrict__ param,
                                          float *__restrict__ master, float *__restrict__ m,
                                          float *__restrict__ v, size_t num_elements, float learning_rate,
                                          float beta1, float beta2, float eps, float bias_correction_m,
                                          float bias_correction_v, float gradient_scale, float weight_decay) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }
    const TGrad scaled_grad = common::cuda::Mul(grad[idx], common::cuda::Cast<TGrad>(gradient_scale));
    const float g = common::cuda::Cast<float>(scaled_grad);
    float p = master[idx];
    if (weight_decay != 0.0f) {
        p += p * (-learning_rate * weight_decay);
    }
    const float next_m = common::cuda::Fma(beta1, m[idx], (1.0f - beta1) * g);
    const float next_v = common::cuda::Fma(beta2, v[idx], (1.0f - beta2) * g * g);
    m[idx] = next_m;
    v[idx] = next_v;
    const float m_hat = next_m / bias_correction_m;
    const float v_hat = next_v / bias_correction_v;
    p -= learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps);
    master[idx] = p;
    param[idx] = common::cuda::Cast<TParam>(p);
}

void AdamWFP32MasterStep(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                         const std::shared_ptr<Tensor> &master, const std::shared_ptr<Tensor> &m,
                         const std::shared_ptr<Tensor> &v, float learning_rate, float beta1, float beta2,
                         float eps, int64_t t, float gradient_scale, float weight_decay) {
    CHECK(master->Dtype() == DataType::kFLOAT32 && m->Dtype() == DataType::kFLOAT32
          && v->Dtype() == DataType::kFLOAT32);
    const size_t num_elements = grad->NumElements();
    CHECK_EQ(param->NumElements(), num_elements);
    CHECK_EQ(master->NumElements(), num_elements);
    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((num_elements + threads - 1) / threads);
    const auto device = grad->GetDevice();
    const auto stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                            infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                            ->cuda_stream();
    core::cuda::DispatchCudaFunc<DataTypeList<INFINI_ALL_FLOATING_TYPES>,
                                 DataTypeList<INFINI_ALL_FLOATING_TYPES>>(
        {grad->Dtype(), param->Dtype()},
        [=]<typename TGrad, typename TParam>() {
            AdamWFP32MasterStepKernel<<<blocks, threads, 0, stream>>>(
                static_cast<const TGrad *>(grad->DataPtr()), static_cast<TParam *>(param->DataPtr()),
                static_cast<float *>(master->DataPtr()), static_cast<float *>(m->DataPtr()),
                static_cast<float *>(v->DataPtr()), num_elements, learning_rate, beta1, beta2, eps,
                bias_correction_m, bias_correction_v, gradient_scale, weight_decay);
        },
        "CUDA AdamWFP32MasterStep");
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(L2NormAccumulate)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamWFP32MasterStep)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL

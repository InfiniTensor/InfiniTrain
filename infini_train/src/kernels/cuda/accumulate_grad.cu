#include <cmath>
#include <memory>

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

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

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
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

    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < num_elements) {
    //     m_data[idx] = common::cuda::Fma(common::cuda::Cast<T>(beta1), m_data[idx],
    //                                     common::cuda::Cast<T>(1 - beta1) * grad_data[idx]);
    //     v_data[idx] = common::cuda::Fma(common::cuda::Cast<T>(beta2), v_data[idx],
    //                                     common::cuda::Cast<T>(1 - beta2) * grad_data[idx] * grad_data[idx]);

    //     const float m_hat = common::cuda::Cast<float>(m_data[idx]) / bias_correction_m;
    //     const float v_hat = common::cuda::Cast<float>(v_data[idx]) / bias_correction_v;

    //     param_data[idx] = common::cuda::Sub(
    //         param_data[idx], common::cuda::Cast<T>(learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps)));
    // }

    //先搞向量化内存
    constexpr int VEC_SIZE = 16 / sizeof(T);
    size_t vec_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    size_t e_start = num_elements / VEC_SIZE * VEC_SIZE;
    if (vec_idx < e_start) {
        
        //能向量化搬运就向量化搬运
        T local_grad[VEC_SIZE], local_param[VEC_SIZE], local_m[VEC_SIZE], local_v[VEC_SIZE];
        
         // 开始搬运
        *reinterpret_cast<int4*>(local_grad)  = *reinterpret_cast<const int4*>(grad_data + vec_idx);
        *reinterpret_cast<int4*>(local_param) = *reinterpret_cast<int4*>(param_data + vec_idx);
        *reinterpret_cast<int4*>(local_m)     = *reinterpret_cast<int4*>(m_data + vec_idx);
        *reinterpret_cast<int4*>(local_v)     = *reinterpret_cast<int4*>(v_data + vec_idx); 
        

        # pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {

            //将存储的 int4* 转换为float
            float g = common::cuda::Cast<float>(local_grad[i]);
            float p = common::cuda::Cast<float>(local_param[i]); 
            float m = common::cuda::Cast<float>(local_m[i]);
            float v = common::cuda::Cast<float>(local_v[i]);


            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * g * g;

            float m_hat = m / bias_correction_m;
            float v_hat = v / bias_correction_v;

            // 使用内置的快速数学函数处理 float
            p -= learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps);
            
            // 计算完毕，转回 T 存储到 local 数组
            local_m[i] = common::cuda::Cast<T>(m);
            local_v[i] = common::cuda::Cast<T>(v);
            local_param[i] = common::cuda::Cast<T>(p);
        }

        // 写回原数组
        *reinterpret_cast<int4*>(param_data + vec_idx) = *reinterpret_cast<int4*>(local_param);
        *reinterpret_cast<int4*>(m_data + vec_idx)     = *reinterpret_cast<int4*>(local_m);
        *reinterpret_cast<int4*>(v_data + vec_idx)     = *reinterpret_cast<int4*>(local_v);
        
    }else if(vec_idx == e_start){
        
        # pragma unroll
        for(size_t idx = vec_idx; idx < num_elements; ++ idx){

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
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // size_t num_elements = grad->NumElements();

    // const float bias_correction_m = 1.0f - std::pow(beta1, t);
    // const float bias_correction_v = 1.0f - std::pow(beta2, t);

    // int threads_per_block = 256;
    // int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // auto device = grad->GetDevice();
    // const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
    //                               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
    //                               ->cuda_stream();

    // DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
    //     grad->Dtype(),
    //     [=]<typename T>() {
    //         AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
    //             static_cast<const T *>(grad->DataPtr()), static_cast<T *>(param->DataPtr()), num_elements,
    //             static_cast<T *>(m->DataPtr()), static_cast<T *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
    //             bias_correction_m, bias_correction_v);
    //     },
    //     "CUDA AdamAccumulateGrad");

    size_t num_elements = grad->NumElements();

    

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

   

    auto device = grad->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            int element_size = sizeof(T);
            int VEC_SIZE = 16 / element_size;
            int threads_per_block = 256;
            int total_threads = (num_elements + VEC_SIZE - 1) / VEC_SIZE;
            int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
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

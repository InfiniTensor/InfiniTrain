#include <cstddef>
#include <memory>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <typename T> __global__ void FillKernel(T *data, T value, size_t size) {
    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < size) {
    //     data[idx] = value;
    // }

    // 计算一个线程处理的向量步长（16字节 / 类型大小）
    constexpr int VEC_SIZE = 16 / sizeof(T);
    // 重新计算向量化后的全局索引
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (idx + VEC_SIZE <= size) {
        T local[VEC_SIZE];
        // 强制循环展开，在寄存器中准备好填充值
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            local[i] = value;
        }
        // 将寄存器数组转为 int4，单条指令完成 128-bit 写入，压榨显存带宽
        *reinterpret_cast<int4*>(data + idx) = *reinterpret_cast<int4*>(local);
    } else {
        // 处理末尾不足 VEC_SIZE 的非对齐部分
        for (size_t i = idx; i < size; ++i) {
            data[i] = value;
        }
    }
}

void Fill(std::shared_ptr<Tensor> tensor, void *value_ptr) {
    // const int num_tokens = tensor->NumElements();
    // const int threads_per_block = 256;
    // const int num_blocks = (num_tokens + threads_per_block - 1) / threads_per_block;
    // auto device = tensor->GetDevice();
    // const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
    //                               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
    //                               ->cuda_stream();
    // DispatchFunc<INFINI_ALL_TYPES>(
    //     tensor->Dtype(),
    //     [=]<typename T>() {
    //         FillKernel<T><<<num_blocks, threads_per_block, 0, cuda_stream>>>(
    //             static_cast<T *>(tensor->DataPtr()), *(static_cast<T *>(value_ptr)), tensor->NumElements());
    //     },
    //     "CUDA Fill");

    size_t num_elements = tensor->NumElements();
    auto device = tensor->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    DispatchFunc<INFINI_ALL_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            // 每一个 T 类型的大小
            int element_size = sizeof(T);
            // 计算向量化步长，通常 float 是 4 个，half 是 8 个
            int VEC_SIZE = 16 / element_size;
            int threads_per_block = 256;
            // 因为每个线程处理 VEC_SIZE 个元素，所以 Block 总数要除以步长
            int total_threads = (num_elements + VEC_SIZE - 1) / VEC_SIZE;
            int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
            
            FillKernel<T><<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<T *>(tensor->DataPtr()), *(static_cast<T *>(value_ptr)), num_elements);
        },
        "CUDA Fill");
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FILL_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FILL_KERNEL(Fill)

#undef REGISTER_CUDA_FILL_KERNEL
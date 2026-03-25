#include <memory>

#include "infini_train/include/common/common.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <typename Tdst, typename Tsrc>
__global__ void CastKernel(Tdst *dst, const Tsrc *src, size_t num_elements, size_t offset) {
    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    // if (idx < num_elements) {
    //     dst[idx] = common::cuda::Cast<Tdst>(src[idx]);
    // }

    // 统一每个线程处理 4 个元素
    constexpr int VEC_SIZE = 4;
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE + offset;

    if (idx + VEC_SIZE <= num_elements) {
        Tsrc s_vec[VEC_SIZE];
        Tdst d_vec[VEC_SIZE];

        // 根据 Tsrc 宽度决定加载指令 (如果是 2 字节读 8 字节, 如果是 4 字节读 16 字节)
        if constexpr (sizeof(Tsrc) == 2) {
            *reinterpret_cast<longlong1*>(s_vec) = *reinterpret_cast<const longlong1*>(src + idx);
        } else if constexpr (sizeof(Tsrc) == 4) {
            *reinterpret_cast<int4*>(s_vec) = *reinterpret_cast<const int4*>(src + idx);
        } else {
            for (int i = 0; i < VEC_SIZE; ++i) s_vec[i] = src[idx + i];
        }

        // 寄存器内完成类型转换
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            d_vec[i] = common::cuda::Cast<Tdst>(s_vec[i]);
        }

        // 根据 Tdst 宽度决定写回指令
        if constexpr (sizeof(Tdst) == 2) {
            *reinterpret_cast<longlong1*>(d_vec) = *reinterpret_cast<longlong1*>(d_vec);
            *reinterpret_cast<longlong1*>(dst + idx) = *reinterpret_cast<longlong1*>(d_vec);
        } else if constexpr (sizeof(Tdst) == 4) {
            *reinterpret_cast<int4*>(dst + idx) = *reinterpret_cast<int4*>(d_vec);
        } else {
            for (int i = 0; i < VEC_SIZE; ++i) dst[idx + i] = d_vec[i];
        }
    } else {
        // 处理末尾非对齐数据
        for (size_t i = idx; i < num_elements && i < idx + VEC_SIZE; ++i) {
            dst[i] = common::cuda::Cast<Tdst>(src[i]);
        }
    }
}

std::shared_ptr<Tensor> Cast(std::shared_ptr<Tensor> input, DataType dtype) {
    auto dst_tensor = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());
    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const size_t num_elements = input->NumElements();
    
    // const size_t num_elements = input->NumElements();
    // dim3 block_dims(256);
    // dim3 grid_dims(CEIL_DIV(num_elements, block_dims.x));
    // const size_t step = grid_dims.x * block_dims.x;

    // 这里的 VEC_SIZE 必须与 Kernel 内部保持一致
    int VEC_SIZE = 4;
    dim3 block_dims(256);
    // 每个线程干 4 个人的活，所以线程总数除以 4
    dim3 grid_dims(CEIL_DIV(num_elements, block_dims.x * VEC_SIZE));
    const size_t step = grid_dims.x * block_dims.x * VEC_SIZE;

    DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {dtype, input->Dtype()},
        [=]<typename Tdst, typename Tsrc>() {
            auto dst = static_cast<Tdst *>(dst_tensor->DataPtr());
            auto src = static_cast<const Tsrc *>(input->DataPtr());
            // 网格步进循环处理超大规模 Tensor
            for (size_t offset = 0; offset < num_elements; offset += step) {
                CastKernel<<<grid_dims, block_dims, 0, cuda_stream>>>(dst, src, num_elements, offset);
            }
        },
        "CUDA Cast");

    return {dst_tensor};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CAST_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CAST_KERNEL(Cast)

#undef REGISTER_CUDA_CAST_KERNEL
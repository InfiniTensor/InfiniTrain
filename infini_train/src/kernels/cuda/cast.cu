#include <memory>

#include "infini_train/include/common/common.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::kernels::cuda {

template <typename Tdst, typename Tsrc>
__global__ void CastKernel(Tdst *dst, const Tsrc *src, size_t num_elements, size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < num_elements) {
        dst[idx] = common::cuda::Cast<Tdst>(src[idx]);
    }
}

std::shared_ptr<Tensor> Cast(std::shared_ptr<Tensor> input, DataType dtype) {
    auto dst_tensor = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());
    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const size_t num_elements = input->NumElements();
    dim3 block_dims(256);
    dim3 grid_dims(CEIL_DIV(num_elements, block_dims.x));
    const size_t step = grid_dims.x * block_dims.x;

    DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {dtype, input->Dtype()},
        [=]<typename Tdst, typename Tsrc>() {
            auto dst = static_cast<Tdst *>(dst_tensor->DataPtr());
            auto src = static_cast<const Tsrc *>(input->DataPtr());
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

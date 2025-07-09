#include "infini_train/include/common/cuda/common_cuda.cuh"

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
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());

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
                CastKernel<<<grid_dims, block_dims, 0, cuda_device->Stream()>>>(dst, src, num_elements, offset);
            }
        },
        "CUDA Cast");

    return {dst_tensor};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CAST_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CAST_KERNEL(Cast)

#undef REGISTER_CUDA_CAST_KERNEL

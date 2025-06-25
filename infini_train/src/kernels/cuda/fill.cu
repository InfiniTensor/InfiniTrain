#include "glog/logging.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

template <typename T> __global__ void FillKernel(T *data, T value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

void Fill(std::shared_ptr<Tensor> tensor, void *value_ptr) {
    const int num_tokens = tensor->NumElements();
    const int threads_per_block = 256;
    const int num_blocks = (num_tokens + threads_per_block - 1) / threads_per_block;
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(tensor->GetDevice());

    switch (tensor->Dtype()) {
    case DataType::kFLOAT32: {
        FillKernel<float><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
            static_cast<float *>(tensor->DataPtr()), *(static_cast<float *>(value_ptr)), tensor->NumElements());
        break;
    }
    case DataType::kINT64: {
        FillKernel<int64_t><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
            static_cast<int64_t *>(tensor->DataPtr()), *(static_cast<int64_t *>(value_ptr)), tensor->NumElements());
        break;
    }
    default:
        LOG(FATAL) << "Unsupported data type: " << static_cast<int>(tensor->Dtype());
    }
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FILL_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FILL_KERNEL(Fill)

#undef REGISTER_CUDA_FILL_KERNEL

#include "glog/logging.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
void Fill(std::shared_ptr<Tensor> tensor, void *value_ptr) {
    // FIXME(zbl): support other data types
    thrust::device_ptr<float> dev_ptr(reinterpret_cast<float *>(tensor->DataPtr()));
    thrust::fill(thrust::cuda::par.on(0), dev_ptr, dev_ptr + tensor->NumElements(), *(static_cast<float *>(value_ptr)));
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FILL_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FILL_KERNEL(Fill)

#undef REGISTER_CUDA_FILL_KERNEL

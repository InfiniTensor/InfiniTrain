#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void TopKForwardKernel(const T *__restrict__ input, T *__restrict__ top_values,
                                  int64_t *__restrict__ top_indices, int64_t rows, int64_t dim_size, int64_t inner_size,
                                  int64_t topk, bool largest) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t outer = row / inner_size;
    const int64_t inner = row % inner_size;
    for (int64_t idx = 0; idx < dim_size; ++idx) {
        const float value = static_cast<float>(input[outer * dim_size * inner_size + idx * inner_size + inner]);
        int64_t rank = 0;
        for (int64_t other_idx = 0; other_idx < dim_size; ++other_idx) {
            const float other_value
                = static_cast<float>(input[outer * dim_size * inner_size + other_idx * inner_size + inner]);
            const bool ranks_before = largest ? (other_value > value || (other_value == value && other_idx < idx))
                                              : (other_value < value || (other_value == value && other_idx < idx));
            if (ranks_before) {
                ++rank;
            }
        }
        if (rank < topk) {
            const int64_t out_offset = outer * topk * inner_size + rank * inner_size + inner;
            top_values[out_offset] = input[outer * dim_size * inner_size + idx * inner_size + inner];
            top_indices[out_offset] = idx;
        }
    }
}

std::vector<std::shared_ptr<Tensor>> TopKForward(const std::shared_ptr<Tensor> &input, int64_t topk, int64_t dim,
                                                 bool largest, bool sorted) {
    CHECK_GE(input->Dims().size(), 1);
    CHECK(sorted) << "TopK currently only supports sorted=true";
    const auto &dims = input->Dims();
    if (dim < 0) {
        dim += static_cast<int64_t>(dims.size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(dims.size()));

    const int64_t dim_size = dims[dim];
    CHECK_GT(dim_size, 0);
    CHECK_GT(topk, 0);
    CHECK_LE(topk, dim_size);
    int64_t outer_size = 1;
    for (int64_t idx = 0; idx < dim; ++idx) { outer_size *= dims[idx]; }
    int64_t inner_size = 1;
    for (size_t idx = static_cast<size_t>(dim) + 1; idx < dims.size(); ++idx) { inner_size *= dims[idx]; }
    const int64_t rows = outer_size * inner_size;

    auto topk_dims = dims;
    topk_dims[dim] = topk;
    auto top_values = std::make_shared<Tensor>(topk_dims, input->Dtype(), input->GetDevice());
    auto top_indices = std::make_shared<Tensor>(topk_dims, DataType::kINT64, input->GetDevice());

    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            TopKForwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<T *>(top_values->DataPtr()),
                static_cast<int64_t *>(top_indices->DataPtr()), rows, dim_size, inner_size, topk, largest);
        },
        "CUDA TopKForward");

    return {top_values, top_indices};
}

template <typename T>
__global__ void TopKBackwardKernel(const T *__restrict__ grad_values, const int64_t *__restrict__ indices,
                                   T *__restrict__ grad_input, int64_t rows, int64_t dim_size, int64_t inner_size,
                                   int64_t topk) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t outer = row / inner_size;
    const int64_t inner = row % inner_size;
    for (int64_t selected = 0; selected < topk; ++selected) {
        const int64_t out_offset = outer * topk * inner_size + selected * inner_size + inner;
        const int64_t selected_idx = indices[out_offset];
        grad_input[outer * dim_size * inner_size + selected_idx * inner_size + inner] = grad_values[out_offset];
    }
}

std::shared_ptr<Tensor> TopKBackward(const std::shared_ptr<Tensor> &grad_values, const std::shared_ptr<Tensor> &indices,
                                     const std::vector<int64_t> &input_dims, int64_t dim) {
    CHECK(indices->Dtype() == DataType::kINT64) << "CUDA TopKBackward expects int64 indices";
    CHECK(grad_values->Dims() == indices->Dims());
    CHECK(!input_dims.empty());
    if (dim < 0) {
        dim += static_cast<int64_t>(input_dims.size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(input_dims.size()));

    const int64_t dim_size = input_dims[dim];
    const int64_t topk = indices->Dims()[dim];
    int64_t outer_size = 1;
    for (int64_t idx = 0; idx < dim; ++idx) { outer_size *= input_dims[idx]; }
    int64_t inner_size = 1;
    for (size_t idx = static_cast<size_t>(dim) + 1; idx < input_dims.size(); ++idx) { inner_size *= input_dims[idx]; }
    const int64_t rows = outer_size * inner_size;

    auto grad_input = std::make_shared<Tensor>(input_dims, grad_values->Dtype(), grad_values->GetDevice());
    auto device = grad_values->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    CUDA_CHECK(cudaMemsetAsync(grad_input->DataPtr(), 0, grad_input->SizeInBytes(), stream));

    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad_values->Dtype(),
        [=]<typename T>() {
            TopKBackwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(grad_values->DataPtr()), static_cast<const int64_t *>(indices->DataPtr()),
                static_cast<T *>(grad_input->DataPtr()), rows, dim_size, inner_size, topk);
        },
        "CUDA TopKBackward");

    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_TOPK_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_TOPK_KERNEL(TopKForward)
REGISTER_CUDA_TOPK_KERNEL(TopKBackward)

#undef REGISTER_CUDA_TOPK_KERNEL

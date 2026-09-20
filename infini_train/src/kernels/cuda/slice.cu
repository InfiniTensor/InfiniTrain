#include <algorithm>
#include <cstdint>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
constexpr int kMaxDims = 8;
struct SliceMeta {
    int64_t new_dims[kMaxDims];
    int64_t starts[kMaxDims];
    int64_t steps[kMaxDims];
    int64_t in_strides[kMaxDims];
    int64_t out_strides[kMaxDims];
};

template <typename T>
__global__ void SliceForwardKernel(const T *input, T *output, const SliceMeta meta, int num_dims,
                                   int64_t total_elements) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) {
        return;
    }

    int64_t in_index = 0;
    for (int i = 0; i < num_dims; ++i) {
        int64_t idx = (out_idx / meta.out_strides[i]) % meta.new_dims[i];
        in_index += (meta.starts[i] + idx * meta.steps[i]) * meta.in_strides[i];
    }

    output[out_idx] = input[in_index];
}

std::shared_ptr<Tensor> SliceForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                                     const std::vector<int64_t> &ends, const std::vector<int64_t> &steps) {
    CHECK_EQ(starts.size(), ends.size());
    CHECK_EQ(starts.size(), steps.size());
    auto &dims = input->Dims();
    CHECK_EQ(starts.size(), dims.size());
    const int64_t num_dims = dims.size();
    CHECK_LE(num_dims, kMaxDims);

    std::vector<int64_t> new_dims;
    for (int i = 0; i < starts.size(); ++i) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto dtype = input->Dtype();
    auto new_tensor = std::make_shared<Tensor>(new_dims, dtype, input->GetDevice());
    // SliceForwardKernel writes every output index in [0, total_elements); no Fill is needed.

    std::vector<int64_t> src_strides(dims.size(), 0), dst_strides(new_dims.size(), 0);
    int64_t stride = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
        src_strides[i] = stride;
        stride *= dims[i];
    }

    stride = 1;
    for (int i = new_dims.size() - 1; i >= 0; --i) {
        dst_strides[i] = stride;
        stride *= new_dims[i];
    }

    int64_t total_elements = stride;

    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();

    // Metadata (5 arrays x num_dims int64) is passed by value through kernel parameter space
    // (constant cache), so no device buffer / H2D memcpy is needed.
    SliceMeta meta{};
    std::copy(new_dims.begin(), new_dims.end(), meta.new_dims);
    std::copy(starts.begin(), starts.end(), meta.starts);
    std::copy(steps.begin(), steps.end(), meta.steps);
    std::copy(src_strides.begin(), src_strides.end(), meta.in_strides);
    std::copy(dst_strides.begin(), dst_strides.end(), meta.out_strides);

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    core::cuda::DispatchCudaFunc<INFINI_ALL_NUMERIC_TYPES>(
        dtype,
        [=]<typename T>() {
            SliceForwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T *>(input->DataPtr()),
                                                                             static_cast<T *>(new_tensor->DataPtr()),
                                                                             meta, num_dims, total_elements);
        },
        "CUDA SliceForward");

    return new_tensor;
}

template <typename T>
__global__ void SliceBackwardKernel(const T *grad_output, T *grad_input, const SliceMeta meta, int num_dims,
                                    int64_t total_elements) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) {
        return;
    }

    int64_t in_index = 0;
    for (int i = 0; i < num_dims; ++i) {
        int64_t idx = (out_idx / meta.out_strides[i]) % meta.new_dims[i];
        in_index += (meta.starts[i] + idx * meta.steps[i]) * meta.in_strides[i];
    }
    grad_input[in_index] = grad_output[out_idx];
}

std::shared_ptr<Tensor> SliceBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                      const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                      const std::vector<int64_t> &steps) {
    CHECK_EQ(starts.size(), ends.size());
    CHECK_EQ(starts.size(), steps.size());
    auto &dims = input->Dims();
    CHECK_EQ(starts.size(), dims.size());
    const int64_t num_dims = dims.size();
    CHECK_LE(num_dims, kMaxDims);

    std::vector<int64_t> new_dims;
    for (int i = 0; i < starts.size(); ++i) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto grad_output_dtype = grad_output->Dtype();
    auto grad_input = std::make_shared<Tensor>(input->Dims(), grad_output_dtype, grad_output->GetDevice());
    grad_input->Fill(0.0);

    std::vector<int64_t> src_strides(dims.size());
    int64_t stride = 1;
    for (int i = src_strides.size() - 1; i >= 0; --i) {
        src_strides[i] = stride;
        stride *= dims[i];
    }

    std::vector<int64_t> dst_strides(new_dims.size());
    stride = 1;
    for (int i = dst_strides.size() - 1; i >= 0; --i) {
        dst_strides[i] = stride;
        stride *= new_dims[i];
    }

    int64_t total_elements = stride;

    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();

    // Metadata is passed by value through kernel parameter space; no device buffer / H2D memcpy.
    SliceMeta meta{};
    std::copy(new_dims.begin(), new_dims.end(), meta.new_dims);
    std::copy(starts.begin(), starts.end(), meta.starts);
    std::copy(steps.begin(), steps.end(), meta.steps);
    std::copy(src_strides.begin(), src_strides.end(), meta.in_strides);
    std::copy(dst_strides.begin(), dst_strides.end(), meta.out_strides);

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    core::cuda::DispatchCudaFunc<INFINI_ALL_NUMERIC_TYPES>(
        grad_output_dtype,
        [=]<typename T>() {
            SliceBackwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_input->DataPtr()), meta, num_dims,
                total_elements);
        },
        "CUDA SliceBackward");

    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SLICE_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SLICE_KERNEL(SliceForward)
REGISTER_CUDA_SLICE_KERNEL(SliceBackward)

#undef REGISTER_CUDA_SLICE_KERNEL

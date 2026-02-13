#include <cstdint>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void SliceForwardKernel(const T *input, T *output, const int64_t *new_dims, const int64_t *starts,
                                   const int64_t *steps, const int64_t *in_strides, const int64_t *out_strides,
                                   int num_dims, int64_t total_elements) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) {
        return;
    }

    int64_t in_index = 0;
    for (int i = 0; i < num_dims; ++i) {
        int64_t idx = (out_idx / out_strides[i]) % new_dims[i];
        in_index += (starts[i] + idx * steps[i]) * in_strides[i];
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

    std::vector<int64_t> new_dims;
    for (int i = 0; i < starts.size(); ++i) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto dtype = input->Dtype();
    auto new_tensor = std::make_shared<Tensor>(new_dims, dtype, input->GetDevice());
    // NOTE(zbl): must initialize with 0
    DispatchFunc<INFINI_ALL_TYPES>(
        dtype, [=]<typename T>() { new_tensor->Fill<T>(0); }, "CUDA SliceForward");

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

    int64_t *new_dims_dev, *starts_dev, *steps_dev, *input_strides_dev, *output_strides_dev;

    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();

    CUDA_CHECK(cudaMallocAsync(
        &new_dims_dev, (ends.size() + starts.size() + steps.size() + dims.size() + new_dims.size()) * sizeof(int64_t),
        stream));
    starts_dev = new_dims_dev + ends.size();
    steps_dev = starts_dev + starts.size();
    input_strides_dev = steps_dev + steps.size();
    output_strides_dev = input_strides_dev + dims.size();

    CUDA_CHECK(
        cudaMemcpyAsync(new_dims_dev, new_dims.data(), ends.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(starts_dev, starts.data(), starts.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(steps_dev, steps.data(), steps.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(input_strides_dev, src_strides.data(), dims.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(output_strides_dev, dst_strides.data(), new_dims.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    DispatchFunc<INFINI_ALL_TYPES>(
        dtype,
        [=]<typename T>() {
            SliceForwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<T *>(new_tensor->DataPtr()), new_dims_dev,
                starts_dev, steps_dev, input_strides_dev, output_strides_dev, num_dims, total_elements);
        },
        "CUDA SliceForward");

    cudaFreeAsync(new_dims_dev, stream);

    // NOTE(dcj):
    // Synchronize the stream here to ensure all preceding H2D/D2H memcpy
    // operations have completed before the host buffers go out of scope.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return new_tensor;
}

template <typename T>
__global__ void SliceBackwardKernel(const T *grad_output, T *grad_input, const int64_t *new_dims, const int64_t *starts,
                                    const int64_t *steps, const int64_t *in_strides, const int64_t *out_strides,
                                    int num_dims, int64_t total_elements) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) {
        return;
    }

    int64_t in_index = 0;
    for (int i = 0; i < num_dims; ++i) {
        int64_t idx = (out_idx / out_strides[i]) % new_dims[i];
        in_index += (starts[i] + idx * steps[i]) * in_strides[i];
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

    std::vector<int64_t> new_dims;
    for (int i = 0; i < starts.size(); ++i) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto grad_output_dtype = grad_output->Dtype();
    auto grad_input = std::make_shared<Tensor>(input->Dims(), grad_output_dtype, grad_output->GetDevice());
    DispatchFunc<INFINI_ALL_TYPES>(
        grad_output_dtype, [=]<typename T>() { grad_input->Fill<T>(0); }, "CUDA SliceBackward");

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

    int dims_size = dims.size();
    int64_t *new_dims_dev, *starts_dev, *steps_dev, *input_strides_dev, *output_strides_dev;

    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    CUDA_CHECK(cudaMallocAsync(
        &new_dims_dev, (ends.size() + starts.size() + steps.size() + dims.size() + new_dims.size()) * sizeof(int64_t),
        stream));
    starts_dev = new_dims_dev + ends.size();
    steps_dev = starts_dev + starts.size();
    input_strides_dev = steps_dev + steps.size();
    output_strides_dev = input_strides_dev + dims.size();

    CUDA_CHECK(
        cudaMemcpyAsync(new_dims_dev, new_dims.data(), ends.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(starts_dev, starts.data(), starts.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(steps_dev, steps.data(), steps.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(input_strides_dev, src_strides.data(), dims.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(output_strides_dev, dst_strides.data(), new_dims.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    DispatchFunc<INFINI_ALL_TYPES>(
        grad_output_dtype,
        [=]<typename T>() {
            SliceBackwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_input->DataPtr()), new_dims_dev,
                starts_dev, steps_dev, input_strides_dev, output_strides_dev, num_dims, total_elements);
        },
        "CUDA SliceBackward");

    CUDA_CHECK(cudaFreeAsync(new_dims_dev, stream));

    // NOTE(dcj):
    // Synchronize the stream here to ensure all preceding H2D/D2H memcpy
    // operations have completed before the host buffers go out of scope.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SLICE_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SLICE_KERNEL(SliceForward)
REGISTER_CUDA_SLICE_KERNEL(SliceBackward)

#undef REGISTER_CUDA_SLICE_KERNEL

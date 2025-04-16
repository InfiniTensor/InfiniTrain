#include "infini_train/include/kernels/cuda/slice.h"

#include <cmath>
#include <functional>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
__global__ void SliceForwardKernel(const float *input, float *output, const int64_t *new_dims, const int64_t *starts,
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
    for (int i = 0; i < starts.size(); i++) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto new_tensor = std::make_shared<Tensor>(new_dims, input->Dtype(), input->GetDevice());

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
    cudaMalloc(&new_dims_dev, ends.size() * sizeof(int64_t));
    cudaMalloc(&starts_dev, starts.size() * sizeof(int64_t));
    cudaMalloc(&steps_dev, steps.size() * sizeof(int64_t));
    cudaMalloc(&input_strides_dev, dims.size() * sizeof(int64_t));
    cudaMalloc(&output_strides_dev, new_dims.size() * sizeof(int64_t));

    cudaMemcpy(new_dims_dev, new_dims.data(), ends.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(starts_dev, starts.data(), starts.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(steps_dev, steps.data(), steps.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(input_strides_dev, src_strides.data(), dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(output_strides_dev, dst_strides.data(), new_dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    SliceForwardKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<float *>(new_tensor->DataPtr()), new_dims_dev,
        starts_dev, steps_dev, input_strides_dev, output_strides_dev, num_dims, total_elements);

    cudaFree(new_dims_dev);
    cudaFree(starts_dev);
    cudaFree(steps_dev);
    cudaFree(input_strides_dev);
    cudaFree(output_strides_dev);

    return new_tensor;
}

__global__ void SliceBackwardKernel(const float *grad_output, float *grad_input, const int64_t *new_dims,
                                    const int64_t *starts, const int64_t *steps, const int64_t *in_strides,
                                    const int64_t *out_strides, int num_dims, int64_t total_elements) {
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
    for (int i = 0; i < starts.size(); i++) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    grad_input->Fill<float>(0.0);

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
    cudaMalloc(&new_dims_dev, ends.size() * sizeof(int64_t));
    cudaMalloc(&starts_dev, starts.size() * sizeof(int64_t));
    cudaMalloc(&steps_dev, steps.size() * sizeof(int64_t));
    cudaMalloc(&input_strides_dev, dims.size() * sizeof(int64_t));
    cudaMalloc(&output_strides_dev, new_dims.size() * sizeof(int64_t));

    cudaMemcpy(new_dims_dev, new_dims.data(), ends.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(starts_dev, starts.data(), starts.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(steps_dev, steps.data(), steps.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(input_strides_dev, src_strides.data(), dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(output_strides_dev, dst_strides.data(), new_dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    SliceBackwardKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<float *>(grad_input->DataPtr()), new_dims_dev,
        starts_dev, steps_dev, input_strides_dev, output_strides_dev, num_dims, total_elements);

    cudaFree(new_dims_dev);
    cudaFree(starts_dev);
    cudaFree(steps_dev);
    cudaFree(input_strides_dev);
    cudaFree(output_strides_dev);

    return grad_input;
}
} // namespace infini_train::kernels::cuda

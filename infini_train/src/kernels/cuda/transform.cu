#include "infini_train/include/kernels/cuda/elementwise.h"

#include <cmath>
#include <functional>
#include <memory>

#include "cuda_runtime.h"
#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void TrilForwardKernel(const float *input, float *output, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal >= 0) {
        output[idx] = input[idx];
    } else {
        output[idx] = 0.0f;
    }
}

std::shared_ptr<Tensor> TrilForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);
    int64_t rows = input->Dims()[0];
    int64_t cols = input->Dims()[1];

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    TrilForwardKernel<<<num_blocks, threads_per_block>>>(reinterpret_cast<float *>(input->DataPtr()),
                                                         reinterpret_cast<float *>(output->DataPtr()), rows, cols,
                                                         diagonal);
    return output;
}

__global__ void TrilBackwardKernel(const float *grad_output, float *grad_input, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal >= 0) {
        grad_input[idx] = grad_output[idx];
    } else {
        grad_input[idx] = 0.0f;
    }
}

std::shared_ptr<Tensor> TrilBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    int rows = grad_output->Dims()[0];
    int cols = grad_output->Dims()[1];

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    TrilBackwardKernel<<<num_blocks, threads_per_block>>>(reinterpret_cast<const float *>(grad_output->DataPtr()),
                                                          reinterpret_cast<float *>(grad_input->DataPtr()), rows, cols,
                                                          diagonal);

    return grad_input;
}

__global__ void TransposeForwardKernel(const float *input, float *output, const int64_t *in_dims,
                                       const int64_t *in_strides, const int64_t *out_strides, int64_t ndim,
                                       int64_t dim0, int64_t dim1, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    int64_t remaining = idx;
    int64_t in_flat_idx = 0;

    for (int i = 0; i < ndim; ++i) {
        // the i-th coords of output tensor
        int64_t coord = remaining / out_strides[i];
        remaining %= out_strides[i];

        // the corresponding dim in input after swapping dim0 & dim1
        int64_t mapped_dim = coord;
        if (i == dim0) {
            mapped_dim = remaining / out_strides[dim1]; // 注意 remaining 已变
        } else if (i == dim1) {
            mapped_dim = remaining / out_strides[dim0];
        }

        // calculate flat idx of input using mapped_dim
        in_flat_idx += mapped_dim * in_strides[i];
    }

    // copy to output
    output[idx] = input[in_flat_idx];
}

std::shared_ptr<Tensor> TransposeForward(const std::shared_ptr<Tensor> &input, int64_t dim0, int64_t dim1) {
    dim0 = dim0 < 0 ? dim0 + input->Dims().size() : dim0;
    dim1 = dim1 < 0 ? dim1 + input->Dims().size() : dim1;
    CHECK(dim0 >= 0 && dim0 < input->Dims().size() && dim1 >= 0 && dim1 < input->Dims().size());

    auto in_dims = input->Dims();
    std::vector<int64_t> out_dims = in_dims;
    std::swap(out_dims[dim0], out_dims[dim1]);

    auto output = std::make_shared<Tensor>(out_dims, input->Dtype(), input->GetDevice());

    int64_t ndim = in_dims.size();
    int64_t num_elements = output->NumElements();

    // compute strides of in_dims and out_dims
    std::vector<int64_t> in_strides(ndim, 1);
    std::vector<int64_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    // Allocate device memory for dims and strides
    // TODO(zbl): avoid using cudaMalloc?
    int64_t *in_dims_dev, *in_strides_dev, *out_strides_dev;
    cudaMalloc(&in_dims_dev, sizeof(int64_t) * ndim);
    cudaMalloc(&in_strides_dev, sizeof(int64_t) * ndim);
    cudaMalloc(&out_strides_dev, sizeof(int64_t) * ndim);
    cudaMemcpy(in_dims_dev, in_dims.data(), sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
    cudaMemcpy(in_strides_dev, in_strides.data(), sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
    cudaMemcpy(out_strides_dev, out_strides.data(), sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    TransposeForwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const float *>(input->DataPtr()), reinterpret_cast<float *>(output->DataPtr()), in_dims_dev,
        in_strides_dev, out_strides_dev, ndim, dim0, dim1, threads_per_block);

    cudaFree(in_dims_dev);
    cudaFree(in_strides_dev);
    cudaFree(out_strides_dev);

    return output;
}

std::shared_ptr<Tensor> TransposeBackward(const std::shared_ptr<Tensor> &grad_output, int64_t dim0, int64_t dim1) {
    return TransposeForward(grad_output, dim1, dim0);
}

__global__ void MaskForwardKernel(const float *input, const float *mask, float *output, float value, int batch_size,
                                  int mask_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * mask_size) {
        output[i] = (mask[i % mask_size] == 1.0f) ? value : input[i];
    }
}

std::shared_ptr<Tensor> MaskForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &mask,
                                    float value) {
    auto input_shape = input->Dims();
    auto mask_shape = mask->Dims();
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(mask->Dtype()));

    int64_t input_dims = input_shape.size();
    int64_t mask_dims = mask_shape.size();
    for (int i = 0; i < mask_dims; ++i) {
        int input_dim = input_shape[input_dims - mask_dims + i];
        int mask_dim = mask_shape[i];
        CHECK(input_dim == mask_dim || mask_dim == 1);
    }

    int64_t mask_size = mask->NumElements();
    int64_t batch_size = input->NumElements() / mask_size;

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (input->NumElements() + threads_per_block - 1) / threads_per_block;

    MaskForwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const float *>(input->DataPtr()), reinterpret_cast<const float *>(mask->DataPtr()),
        reinterpret_cast<float *>(output->DataPtr()), value, batch_size, mask_size);
    return output;
}

__global__ void MaskBackwardKernel(const float *grad_output, const float *mask, float *grad_input, int batch_size,
                                   int mask_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * mask_size) {
        grad_input[i] = (mask[i % mask_size] == 1.0f) ? 0.0f : grad_output[i];
    }
}

std::shared_ptr<Tensor> MaskBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask) {
    auto output_shape = grad_output->Dims();
    auto mask_shape = mask->Dims();
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(mask->Dtype()));

    int64_t output_dims = output_shape.size();
    int64_t mask_dims = mask_shape.size();
    for (int i = 0; i < mask_dims; ++i) {
        int out_dim = output_shape[output_dims - mask_dims + i];
        int mask_dim = mask_shape[i];
        CHECK(out_dim == mask_dim || mask_dim == 1);
    }

    int64_t mask_size = mask->NumElements();
    int64_t batch_size = grad_output->NumElements() / mask_size;

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (grad_output->NumElements() + threads_per_block - 1) / threads_per_block;

    MaskBackwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const float *>(grad_output->DataPtr()), reinterpret_cast<const float *>(mask->DataPtr()),
        reinterpret_cast<float *>(grad_input->DataPtr()), batch_size, mask_size);
    return grad_input;
}
} // namespace infini_train::kernels::cuda

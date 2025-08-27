#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

template <typename T>
__global__ void TrilForwardKernel(const T *input, T *output, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal >= 0) {
        output[idx] = input[idx];
    } else {
        output[idx] = T(0);
    }
}

std::shared_ptr<Tensor> TrilForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);
    int64_t rows = input->Dims()[0];
    int64_t cols = input->Dims()[1];

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());

    DispatchFunc<INFINI_ALL_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            TrilForwardKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), rows, cols, diagonal);
        },
        "CUDA TrilForward");

    return output;
}

template <typename T>
__global__ void TrilBackwardKernel(const T *grad_output, T *grad_input, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal >= 0) {
        grad_input[idx] = grad_output[idx];
    } else {
        grad_input[idx] = T(0);
    }
}

std::shared_ptr<Tensor> TrilBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    int rows = grad_output->Dims()[0];
    int cols = grad_output->Dims()[1];

    auto dtype = grad_output->Dtype();
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), dtype, grad_output->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());

    DispatchFunc<INFINI_ALL_TYPES>(
        dtype,
        [=]<typename T>() {
            grad_input->Fill<T>(0);
            TrilBackwardKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_input->DataPtr()), rows, cols,
                diagonal);
        },
        "CUDA TrilBackward");

    return grad_input;
}

template <typename T>
__global__ void TriuForwardKernel(const T *input, T *output, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal <= 0) {
        output[idx] = input[idx];
    } else {
        output[idx] = T(0);
    }
}

std::shared_ptr<Tensor> TriuForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);
    int64_t rows = input->Dims()[0];
    int64_t cols = input->Dims()[1];

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());

    DispatchFunc<INFINI_ALL_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            TriuForwardKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), rows, cols, diagonal);
        },
        "CUDA TriuForward");

    return output;
}

template <typename T>
__global__ void TriuBackwardKernel(const T *grad_output, T *grad_input, int rows, int cols, int64_t diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    int row = idx / cols;
    int col = idx % cols;

    if (row - col + diagonal <= 0) {
        grad_input[idx] = grad_output[idx];
    } else {
        grad_input[idx] = T(0);
    }
}

std::shared_ptr<Tensor> TriuBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    int rows = grad_output->Dims()[0];
    int cols = grad_output->Dims()[1];

    auto dtype = grad_output->Dtype();
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), dtype, grad_output->GetDevice());

    int threads_per_block = 256;
    int num_blocks = (rows * cols + threads_per_block - 1) / threads_per_block;
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());

    DispatchFunc<INFINI_ALL_TYPES>(
        dtype,
        [=]<typename T>() {
            grad_input->Fill<T>(0);
            TriuBackwardKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_input->DataPtr()), rows, cols,
                diagonal);
        },
        "CUDA TriuBackward");

    return grad_input;
}

template <typename T>
__global__ void TransposeForwardKernel(const T *input, T *output, const int64_t *in_dims, const int64_t *in_strides,
                                       const int64_t *out_strides, int64_t ndim, int64_t dim0, int64_t dim1,
                                       int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    int64_t remaining = idx;
    // TODO(zbl): assume ndim <= 8 here
    int64_t coords[8];

    // 1. decode coord from output index
    for (int i = 0; i < ndim; ++i) {
        coords[i] = remaining / out_strides[i];
        remaining %= out_strides[i];
    }

    // 2. swap the coordinates
    int64_t tmp = coords[dim0];
    coords[dim0] = coords[dim1];
    coords[dim1] = tmp;

    // 3. compute input flat index
    int64_t in_flat_idx = 0;
    for (int i = 0; i < ndim; ++i) { in_flat_idx += coords[i] * in_strides[i]; }

    output[idx] = input[in_flat_idx];
}

std::shared_ptr<Tensor> TransposeForward(const std::shared_ptr<Tensor> &input, int64_t dim0, int64_t dim1) {
    // TODO(zbl): assume ndim <= 8 here
    CHECK_LE(input->Dims().size(), 8);
    dim0 = dim0 < 0 ? dim0 + input->Dims().size() : dim0;
    dim1 = dim1 < 0 ? dim1 + input->Dims().size() : dim1;
    CHECK(dim0 >= 0 && dim0 < input->Dims().size() && dim1 >= 0 && dim1 < input->Dims().size());

    auto in_dims = input->Dims();
    std::vector<int64_t> out_dims = in_dims;
    std::swap(out_dims[dim0], out_dims[dim1]);

    auto dtype = input->Dtype();
    auto output = std::make_shared<Tensor>(out_dims, dtype, input->GetDevice());
    int64_t ndim = in_dims.size();
    int64_t num_elements = output->NumElements();

    // compute strides of in_dims and out_dims
    std::vector<int64_t> in_strides(ndim, 1);
    std::vector<int64_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    const auto &stream = cuda_device->Stream();

    // Allocate device memory for dims and strides
    // TODO(zbl): avoid using cudaMalloc?
    int64_t *device_buffer;
    cudaMallocAsync(&device_buffer, 3 * ndim * sizeof(int64_t), stream);

    int64_t *in_dims_dev = device_buffer;
    int64_t *in_strides_dev = device_buffer + ndim;
    int64_t *out_strides_dev = device_buffer + 2 * ndim;

    std::vector<int64_t> host_buffer;
    host_buffer.insert(host_buffer.end(), in_dims.begin(), in_dims.end());
    host_buffer.insert(host_buffer.end(), in_strides.begin(), in_strides.end());
    host_buffer.insert(host_buffer.end(), out_strides.begin(), out_strides.end());

    cudaMemcpyAsync(device_buffer, host_buffer.data(), 3 * ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    DispatchFunc<INFINI_ALL_TYPES>(
        dtype,
        [=]<typename T>() {
            output->Fill<T>(0);
            TransposeForwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), in_dims_dev,
                in_strides_dev, out_strides_dev, ndim, dim0, dim1, num_elements);
        },
        "CUDA TransposeForward");

    cudaFreeAsync(device_buffer, stream);

    return output;
}

std::shared_ptr<Tensor> TransposeBackward(const std::shared_ptr<Tensor> &grad_output, int64_t dim0, int64_t dim1) {
    return TransposeForward(grad_output, dim1, dim0);
}

namespace {
enum class MaskMode { kLead, kTail };

static bool IsLeadMaskShape(const std::vector<int64_t> &in, const std::vector<int64_t> &mk) {
    if (mk.empty() || in.empty()) {
        return false;
    }
    if (mk.size() > in.size()) {
        return false;
    }
    for (size_t d = 0; d < mk.size(); ++d) {
        if (!(mk[d] == in[d] || mk[d] == 1)) {
            return false;
        }
    }
    return true;
}

static bool IsTailMaskShape(const std::vector<int64_t> &in, const std::vector<int64_t> &mk) {
    if (mk.size() > in.size()) {
        return false;
    }
    size_t k = mk.size();
    for (size_t i = 0; i < k; ++i) {
        int64_t in_dim = in[in.size() - k + i];
        int64_t mk_dim = mk[i];
        if (!(mk_dim == in_dim || mk_dim == 1)) {
            return false;
        }
    }
    return true;
}

static MaskMode DecideMaskMode(const std::vector<int64_t> &in, const std::vector<int64_t> &mk) {
    bool lead = IsLeadMaskShape(in, mk);
    bool tail = IsTailMaskShape(in, mk);
    CHECK(lead || tail) << "Mask must align/broadcast to either leading or trailing axes.";
    // By default mask along tailing dims
    return tail ? MaskMode::kTail : MaskMode::kLead;
}
} // namespace

template <typename T>
__global__ void MaskForwardKernel(const T *input, const T *mask, T *output, T value, int batch_size, int mask_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * mask_size) {
        output[i] = (mask[i % mask_size] == T(1)) ? value : input[i];
    }
}

template <typename T>
__global__ void MaskLeadsForwardKernel(const T *input, const T *mask, T *output, T value, int rows, int inner) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * inner) {
        output[i] = (mask[i / inner] == T(1)) ? value : input[i];
    }
}

std::shared_ptr<Tensor> MaskForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &mask,
                                    float value) {
    auto input_shape = input->Dims();
    auto mask_shape = mask->Dims();
    auto dtype = input->Dtype();
    // TODO(zbl): support bool mask
    CHECK_EQ(static_cast<int>(dtype), static_cast<int>(mask->Dtype())) << "For now, input/mask dtypes must match.";

    MaskMode mode = DecideMaskMode(input_shape, mask_shape);

    auto output = std::make_shared<Tensor>(input_shape, dtype, input->GetDevice());
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(output->GetDevice());
    int threads_per_block = 256;

    if (mode == MaskMode::kLead) {
        int64_t rows = mask->NumElements();
        int64_t inner = input->NumElements() / rows;
        int num_blocks = static_cast<int>((input->NumElements() + threads_per_block - 1) / threads_per_block);

        DispatchFunc<INFINI_ALL_TYPES>(
            dtype,
            [=]<typename T>() {
                MaskLeadsForwardKernel<T><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                    static_cast<const T *>(input->DataPtr()), static_cast<const T *>(mask->DataPtr()),
                    static_cast<T *>(output->DataPtr()), common::cuda::Cast<T>(value), rows, inner);
            },
            "CUDA MaskForward(rows)");
    } else { // kTail
        int64_t mask_size = mask->NumElements();
        int64_t batch_size = input->NumElements() / mask_size;
        int num_blocks = static_cast<int>((input->NumElements() + threads_per_block - 1) / threads_per_block);

        DispatchFunc<INFINI_ALL_TYPES>(
            dtype,
            [=]<typename T>() {
                MaskForwardKernel<T><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                    static_cast<const T *>(input->DataPtr()), static_cast<const T *>(mask->DataPtr()),
                    static_cast<T *>(output->DataPtr()), common::cuda::Cast<T>(value), static_cast<int>(batch_size),
                    static_cast<int>(mask_size));
            },
            "CUDA MaskForward(tail)");
    }

    return output;
}

template <typename T>
__global__ void MaskBackwardKernel(const T *grad_output, const T *mask, T *grad_input, int batch_size, int mask_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * mask_size) {
        grad_input[i] = (mask[i % mask_size] == T(1)) ? T(0) : grad_output[i];
    }
}

template <typename T>
__global__ void MaskLeadsBackwardKernel(const T *grad_output, const T *mask, T *grad_input, int rows, int inner) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * inner) {
        grad_input[i] = (mask[i / inner] == T(1)) ? T(0) : grad_output[i];
    }
}

std::shared_ptr<Tensor> MaskBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask) {
    auto output_shape = grad_output->Dims();
    auto mask_shape = mask->Dims();
    auto dtype = grad_output->Dtype();
    auto grad_output_ = std::make_shared(grad_output->To(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(grad_output_->Dtype()), static_cast<int>(mask->Dtype()))
        << "For now, grad_output/mask dtypes must match.";

    MaskMode mode = DecideMaskMode(output_shape, mask_shape);

    auto grad_input = std::make_shared<Tensor>(output_shape, dtype, grad_output->GetDevice());
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());
    int threads_per_block = 256;

    if (mode == MaskMode::kLead) {
        int64_t rows = mask->NumElements();
        int64_t inner = grad_output->NumElements() / rows;
        int num_blocks = static_cast<int>((grad_output->NumElements() + threads_per_block - 1) / threads_per_block);

        DispatchFunc<INFINI_ALL_TYPES>(
            dtype,
            [=]<typename T>() {
                grad_input->Fill<T>(0);
                MaskLeadsBackwardKernel<T><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                    static_cast<const T *>(grad_output_->DataPtr()), static_cast<const T *>(mask->DataPtr()),
                    static_cast<T *>(grad_input->DataPtr()), rows, inner);
            },
            "CUDA MaskBackward(rows)");
    } else { // kTail
        int64_t mask_size = mask->NumElements();
        int64_t batch_size = grad_output->NumElements() / mask_size;
        int num_blocks = static_cast<int>((grad_output->NumElements() + threads_per_block - 1) / threads_per_block);

        DispatchFunc<INFINI_ALL_TYPES>(
            dtype,
            [=]<typename T>() {
                grad_input->Fill<T>(0);
                MaskBackwardKernel<T><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                    static_cast<const T *>(grad_output_->DataPtr()), static_cast<const T *>(mask->DataPtr()),
                    static_cast<T *>(grad_input->DataPtr()), static_cast<int>(batch_size), static_cast<int>(mask_size));
            },
            "CUDA MaskBackward(tail)");
    }

    return grad_input;
}

template <typename T>
__global__ void RepeatInterleaveForwardKernel(const T *input, T *output, int64_t outer, int64_t dim_size, int64_t inner,
                                              int64_t repeat) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * dim_size * repeat * inner;
    if (idx >= total) {
        return;
    }

    int64_t i = idx / inner;
    int64_t j = idx % inner;

    int64_t o = i / (dim_size * repeat);
    int64_t di = (i / repeat) % dim_size;

    output[idx] = input[(o * dim_size + di) * inner + j];
}

std::shared_ptr<Tensor> RepeatInterleaveForward(const std::shared_ptr<Tensor> &input, int64_t repeat, int64_t dim) {
    CHECK_GT(repeat, 0);
    CHECK_GE(dim, 0);
    CHECK_LT(dim, input->Dims().size());

    const auto &input_dims = input->Dims();
    const int64_t outer = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t inner
        = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t dim_size = input_dims[dim];

    std::vector<int64_t> output_dims = input_dims;
    output_dims[dim] = dim_size * repeat;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    int64_t total_elements = outer * dim_size * repeat * inner;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());

    DispatchFunc<INFINI_ALL_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            RepeatInterleaveForwardKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), outer, dim_size, inner,
                repeat);
        },
        "CUDA RepeatInterleaveForward");

    return output;
}

template <typename T>
__global__ void RepeatInterleaveBackwardKernel(const T *grad_output, T *grad_input, int64_t outer, int64_t dim_size,
                                               int64_t inner, int64_t repeat) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer * dim_size * inner;
    if (idx >= total) {
        return;
    }

    int64_t i = idx / inner;
    int64_t j = idx % inner;

    int64_t o = i / dim_size;
    int64_t di = i % dim_size;

    T sum = T(0);
    for (int64_t r = 0; r < repeat; ++r) {
        int64_t out_idx = ((o * dim_size * repeat + di * repeat + r) * inner) + j;
        sum += grad_output[out_idx];
    }
    grad_input[idx] = sum;
}

std::shared_ptr<Tensor> RepeatInterleaveBackward(const std::shared_ptr<Tensor> &grad_output,
                                                 const std::vector<int64_t> &input_dims, int64_t dim) {
    CHECK_GE(dim, 0);
    CHECK_LT(dim, input_dims.size());

    const int64_t outer = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t inner
        = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t dim_size = input_dims[dim];

    int64_t repeat = grad_output->Dims()[dim] / dim_size;
    CHECK_EQ(grad_output->Dims()[dim], dim_size * repeat);

    auto grad_input = std::make_shared<Tensor>(input_dims, grad_output->Dtype(), grad_output->GetDevice());

    int64_t total_elements = outer * dim_size * inner;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());

    DispatchFunc<INFINI_ALL_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            grad_input->Fill<T>(0);
            RepeatInterleaveBackwardKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_input->DataPtr()), outer,
                dim_size, inner, repeat);
        },
        "CUDA RepeatInterleaveBackward");

    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_TRANSFORM_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_TRANSFORM_KERNEL(TrilForward)
REGISTER_CUDA_TRANSFORM_KERNEL(TrilBackward)
REGISTER_CUDA_TRANSFORM_KERNEL(TriuForward)
REGISTER_CUDA_TRANSFORM_KERNEL(TriuBackward)
REGISTER_CUDA_TRANSFORM_KERNEL(TransposeForward)
REGISTER_CUDA_TRANSFORM_KERNEL(TransposeBackward)
REGISTER_CUDA_TRANSFORM_KERNEL(MaskForward)
REGISTER_CUDA_TRANSFORM_KERNEL(MaskBackward)
REGISTER_CUDA_TRANSFORM_KERNEL(RepeatInterleaveForward)
REGISTER_CUDA_TRANSFORM_KERNEL(RepeatInterleaveBackward)

#undef REGISTER_CUDA_TRANSFORM_KERNEL

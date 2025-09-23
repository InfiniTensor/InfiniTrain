#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {
static inline std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &dims) {
    if (dims.empty()) {
        return {};
    }
    std::vector<int64_t> strides(dims.size());
    int64_t s = 1;
    for (int i = (int)dims.size() - 1; i >= 0; --i) {
        strides[i] = s;
        s *= dims[i];
    }
    return strides;
}
} // namespace

template <typename T>
__global__ void IndexGatherForwardKernel(const T *__restrict__ input, const int64_t *__restrict__ norm_index,
                                         T *__restrict__ output, const int64_t *__restrict__ out_dims,
                                         const int64_t *__restrict__ in_strides,
                                         const int64_t *__restrict__ out_strides, int num_dims, int gather_dim,
                                         int64_t dim_size_gather, int64_t total_elements) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) {
        return;
    }

    const int64_t gather_j = norm_index[out_idx];

    int64_t in_linear = 0, tmp = out_idx;
#pragma unroll
    for (int d = 0; d < num_dims; ++d) {
        int64_t coord = tmp / out_strides[d];
        tmp -= coord * out_strides[d];
        in_linear += ((d == gather_dim) ? gather_j : coord) * in_strides[d];
    }
    output[out_idx] = input[in_linear];
}

std::shared_ptr<Tensor> IndexGatherForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &index,
                                           int64_t dim) {
    const auto &in_dims = input->Dims();
    const auto &idx_dims = index->Dims();
    CHECK_EQ(in_dims.size(), idx_dims.size());
    CHECK(input->GetDevice()->Type() == index->GetDevice()->Type());
    CHECK(input->GetDevice()->Index() == index->GetDevice()->Index());

    const int64_t num_dims = in_dims.size();
    if (dim < 0) {
        dim += num_dims;
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, num_dims);

    // NOTE(zbl): Assume index to be int64 Tensors
    CHECK(index->Dtype() == DataType::kINT64);

    for (int d = 0; d < num_dims; ++d) {
        if (d == dim) {
            continue;
        }
        CHECK_EQ(in_dims[d], idx_dims[d]) << "index shape must match input on non-gather dims";
    }

    const auto *cuda_dev = dynamic_cast<const CudaDevice *>(input->GetDevice());
    const auto &stream = cuda_dev->Stream();

    auto dtype = input->Dtype();
    auto out = std::make_shared<Tensor>(idx_dims, dtype, cuda_dev);

    auto in_strides = ComputeStrides(in_dims);
    auto out_strides = ComputeStrides(idx_dims);
    const int64_t total_elements = index->NumElements();

    const int64_t gather_dim_size = in_dims[dim];

    int64_t *dev_buf = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dev_buf, (3 * num_dims) * sizeof(int64_t), stream));
    int64_t *out_dims_dev = dev_buf + 0 * num_dims;
    int64_t *in_strides_dev = dev_buf + 1 * num_dims;
    int64_t *out_strides_dev = dev_buf + 2 * num_dims;

    CUDA_CHECK(
        cudaMemcpyAsync(out_dims_dev, idx_dims.data(), num_dims * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(in_strides_dev, in_strides.data(), num_dims * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(out_strides_dev, out_strides.data(), num_dims * sizeof(int64_t), cudaMemcpyHostToDevice,
                               stream));

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            IndexGatherForwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<const int64_t *>(index->DataPtr()),
                static_cast<T *>(out->DataPtr()), out_dims_dev, in_strides_dev, out_strides_dev, (int)num_dims,
                (int)dim, gather_dim_size, total_elements);
        },
        "CUDA IndexGatherForward");

    CUDA_CHECK(cudaFreeAsync(dev_buf, stream));
    return out;
}

template <typename T>
__global__ void IndexGatherBackwardKernel(const T *__restrict__ grad_output, const int64_t *__restrict__ index,
                                          T *__restrict__ grad_input, const int64_t *__restrict__ out_dims,
                                          const int64_t *__restrict__ in_strides,
                                          const int64_t *__restrict__ out_strides, int num_dims, int gather_dim,
                                          int64_t dim_size_gather, int64_t total_elements) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) {
        return;
    }

    int64_t gather_j = index[out_idx];
    gather_j = (gather_j < 0) ? (gather_j + dim_size_gather) : gather_j;
    if (gather_j < 0) {
        gather_j = 0;
    }
    if (gather_j >= dim_size_gather) {
        gather_j = dim_size_gather - 1;
    }

    int64_t in_linear = 0;
    int64_t tmp = out_idx;
#pragma unroll
    for (int d = 0; d < num_dims; ++d) {
        int64_t coord = tmp / out_strides[d];
        tmp -= coord * out_strides[d];
        if (d == gather_dim) {
            in_linear += gather_j * in_strides[d];
        } else {
            in_linear += coord * in_strides[d];
        }
    }
    atomicAdd(&grad_input[in_linear], grad_output[out_idx]);
}

std::shared_ptr<Tensor> IndexGatherBackward(const std::shared_ptr<Tensor> &grad_output,
                                            const std::shared_ptr<Tensor> &index, int64_t dim,
                                            const std::vector<int64_t> &input_dims) {
    const auto &in_dims = input_dims;
    const auto &idx_dims = index->Dims();
    CHECK_EQ(in_dims.size(), idx_dims.size());
    const int64_t num_dims = in_dims.size();
    if (dim < 0) {
        dim += num_dims;
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, num_dims);

    // NOTE(zbl): Assume index to be int64 Tensors
    CHECK(index->Dtype() == DataType::kINT64);

    for (int d = 0; d < num_dims; ++d) {
        if (d == dim) {
            continue;
        }
        CHECK_EQ(in_dims[d], idx_dims[d]);
    }

    auto dtype = grad_output->Dtype();
    auto grad_input = std::make_shared<Tensor>(in_dims, dtype, grad_output->GetDevice());
    DispatchFunc<INFINI_ALL_TYPES>(
        dtype, [=]<typename T>() { grad_input->Fill<T>(0); }, "CUDA IndexGatherBackwardZero");

    auto in_strides = ComputeStrides(in_dims);
    auto out_strides = ComputeStrides(idx_dims);
    const int64_t total_elements
        = std::accumulate(idx_dims.begin(), idx_dims.end(), (int64_t)1, std::multiplies<int64_t>{});
    const int64_t gather_dim_size = in_dims[dim];

    int64_t *dev_buf = nullptr;
    const size_t n_out = idx_dims.size();
    const size_t n_in_strides = in_dims.size();
    const size_t n_out_strides = idx_dims.size();
    const size_t total_i64 = n_out + n_in_strides + n_out_strides;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());
    const auto &stream = cuda_device->Stream();
    CUDA_CHECK(cudaMallocAsync(&dev_buf, total_i64 * sizeof(int64_t), stream));
    int64_t *out_dims_dev = dev_buf;
    int64_t *in_strides_dev = out_dims_dev + n_out;
    int64_t *out_strides_dev = in_strides_dev + n_in_strides;

    CUDA_CHECK(cudaMemcpyAsync(out_dims_dev, idx_dims.data(), n_out * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(in_strides_dev, in_strides.data(), n_in_strides * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(out_strides_dev, out_strides.data(), n_out_strides * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));

    const int threads = 256;
    const int blocks = (int)((total_elements + threads - 1) / threads);

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            IndexGatherBackwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<const int64_t *>(index->DataPtr()),
                static_cast<T *>(grad_input->DataPtr()), out_dims_dev, in_strides_dev, out_strides_dev, (int)num_dims,
                (int)dim, gather_dim_size, total_elements);
        },
        "CUDA IndexGatherBackward");

    CUDA_CHECK(cudaFreeAsync(dev_buf, stream));
    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_GATHER_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_GATHER_KERNEL(IndexGatherForward)
REGISTER_CUDA_GATHER_KERNEL(IndexGatherBackward)

#undef REGISTER_CUDA_GATHER_KERNEL

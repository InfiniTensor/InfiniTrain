#include <cub/cub.cuh>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.cuh"

namespace infini_train::kernels::cuda {
namespace {
__host__ __device__ constexpr float kInfinity = std::numeric_limits<float>::infinity();
} // namespace

namespace {
// Reduction operators
template <typename T, typename ReduceFunc> struct CubOp;

template <typename T> struct CubOp<T, cub::Sum> {
    __device__ static T Init() { return common::cuda::Cast<T>(0); }
    __device__ static T Reduce(T a, T b) { return common::cuda::Add<T>(a, b); }
    __device__ static cub::Sum Op() { return cub::Sum(); }
};

template <typename T> struct CubOp<T, cub::Max> {
    __device__ static T Init() { return common::cuda::Cast<T>(-kInfinity); }
    __device__ static T Reduce(T a, T b) { return common::cuda::Max<T>(a, b); }
    __device__ static cub::Max Op() { return cub::Max(); }
};

template <typename T> struct CubOp<T, cub::Min> {
    __device__ static T Init() { return common::cuda::Cast<T>(kInfinity); }
    __device__ static T Reduce(T a, T b) { return common::cuda::Min<T>(a, b); }
    __device__ static cub::Min Op() { return cub::Min(); }
};

// Finalization strategies
template <typename T> struct MeanFinalize {
    __device__ __forceinline__ T operator()(T sum, int64_t count) const {
        return common::cuda::Div(sum, common::cuda::Cast<T>(count));
    }
};

template <typename T> struct IdentityFinalize {
    __device__ __forceinline__ T operator()(T val, int64_t) const { return val; }
};

// Generic reduction kernel
template <typename T, typename ReduceFunc, typename FinalizeOp, int BLOCK_SIZE>
__global__ void GenericReduceKernel(const T *input, T *output, int64_t N, int64_t H, int64_t W,
                                    FinalizeOp finalize_op) {
    using BlockReduce = cub::BlockReduce<T, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x;
    if (idx >= N * W) {
        return;
    }

    int n = idx / W;
    int w = idx % W;

    T acc = CubOp<T, ReduceFunc>::Init();
    for (int64_t h = threadIdx.x; h < H; h += blockDim.x) {
        int input_idx = (n * H + h) * W + w;
        acc = CubOp<T, ReduceFunc>::Reduce(acc, input[input_idx]);
    }

    T reduced = BlockReduce(temp_storage).Reduce(acc, CubOp<T, ReduceFunc>::Op());

    if (threadIdx.x == 0) {
        output[idx] = finalize_op(reduced, H);
    }
}

// Unified backward kernel for Mean, Sum, Max, and Min
template <typename T>
__global__ void GenericReduceBackwardKernel(T *grad_input, const T *grad_output, const T *input, const T *reduced,
                                            int64_t N, int64_t H, int64_t W, bool is_mean, bool is_masked) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * W) {
        return;
    }

    int n = idx / (H * W);
    int hw = idx % (H * W);
    int w = hw % W;

    int reduced_idx = n * W + w;

    if (is_masked) {
        T selected = reduced[reduced_idx];
        T value = input[idx];
        grad_input[idx] = (value == selected) ? grad_output[reduced_idx] : T(0);
    } else {
        grad_input[idx] = grad_output[reduced_idx];
        if (is_mean) {
            T H_casted;
            if constexpr (std::is_same_v<T, half>) {
                H_casted = __float2half(static_cast<float>(H));
            } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
                H_casted = __float2bfloat16(static_cast<float>(H));
            } else {
                H_casted = T(H);
            }
            grad_input[idx] /= H_casted;
        }
    }
}
} // namespace

// Common forward implementation for reduce ops
template <typename ReduceFunc, template <typename> class FinalizeOp>
std::shared_ptr<Tensor> ReduceOpForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    const auto &input_dims = input->Dims();
    int64_t actual_dim = dim < 0 ? dim + input_dims.size() : dim;
    CHECK_GE(actual_dim, 0);
    CHECK_LT(actual_dim, input_dims.size());

    std::vector<int64_t> output_dims = input_dims;
    if (keep_dim) {
        output_dims[actual_dim] = 1;
    } else {
        output_dims.erase(output_dims.begin() + actual_dim);
    }

    auto dtype = input->Dtype();
    auto output = std::make_shared<Tensor>(output_dims, dtype, input->GetDevice());

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + actual_dim, 1, std::multiplies<int64_t>());
    int64_t H = input_dims[actual_dim];
    int64_t W = std::accumulate(input_dims.begin() + actual_dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());

    constexpr int BLOCK_SIZE = 256;
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = N * W;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            GenericReduceKernel<T, ReduceFunc, FinalizeOp<T>, BLOCK_SIZE>
                <<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(static_cast<const T *>(input->DataPtr()),
                                                                              static_cast<T *>(output->DataPtr()), N, H,
                                                                              W, FinalizeOp<T>{});
        },
        "CUDA ReductionForward");
    return output;
}

// Common backward implementation for reduce ops
std::shared_ptr<Tensor> ReduceOpBackward(const std::shared_ptr<Tensor> &grad_output,
                                         const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &reduced,
                                         const std::vector<int64_t> &input_dims, const int64_t dim, bool keep_dim,
                                         bool is_mean, bool is_masked) {
    int64_t actual_dim = dim < 0 ? dim + input_dims.size() : dim;
    CHECK_GE(actual_dim, 0);
    CHECK_LT(actual_dim, input_dims.size());

    auto dtype = grad_output->Dtype();
    auto grad_input = std::make_shared<Tensor>(input_dims, dtype, grad_output->GetDevice());

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + actual_dim, 1, std::multiplies<int64_t>());
    int64_t H = input_dims[actual_dim];
    int64_t W = std::accumulate(input_dims.begin() + actual_dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());

    int threads_per_block = 256;
    int num_blocks = (N * H * W + threads_per_block - 1) / threads_per_block;

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());
    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            grad_input->Fill<T>(0);
            GenericReduceBackwardKernel<<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<T *>(grad_input->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
                input ? static_cast<const T *>(input->DataPtr()) : nullptr,
                reduced ? static_cast<const T *>(reduced->DataPtr()) : nullptr, N, H, W, is_mean, is_masked);
        },
        "CUDA ReductionBackward");
    return grad_input;
}

std::shared_ptr<Tensor> MeanForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward<cub::Sum, MeanFinalize>(input, dim, keep_dim);
}

std::shared_ptr<Tensor> SumForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward<cub::Sum, IdentityFinalize>(input, dim, keep_dim);
}

std::shared_ptr<Tensor> MaxForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward<cub::Max, IdentityFinalize>(input, dim, keep_dim);
}

std::shared_ptr<Tensor> MinForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward<cub::Min, IdentityFinalize>(input, dim, keep_dim);
}

std::shared_ptr<Tensor> MeanBackward(const std::shared_ptr<Tensor> &grad_output, const std::vector<int64_t> &input_dims,
                                     const int64_t dim, bool keep_dim) {
    return ReduceOpBackward(grad_output, nullptr, nullptr, input_dims, dim, keep_dim, true, false);
}

std::shared_ptr<Tensor> SumBackward(const std::shared_ptr<Tensor> &grad_output, const std::vector<int64_t> &input_dims,
                                    const int64_t dim, bool keep_dim) {
    return ReduceOpBackward(grad_output, nullptr, nullptr, input_dims, dim, keep_dim, false, false);
}

std::shared_ptr<Tensor> MaxBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &reduced, const int64_t dim, bool keep_dim) {
    return ReduceOpBackward(grad_output, input, reduced, input->Dims(), dim, keep_dim, false, true);
}

std::shared_ptr<Tensor> MinBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &reduced, const int64_t dim, bool keep_dim) {
    return ReduceOpBackward(grad_output, input, reduced, input->Dims(), dim, keep_dim, false, true);
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_REDUCTION_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_REDUCTION_KERNEL(MeanForward)
REGISTER_CUDA_REDUCTION_KERNEL(SumForward)
REGISTER_CUDA_REDUCTION_KERNEL(MaxForward)
REGISTER_CUDA_REDUCTION_KERNEL(MinForward)
REGISTER_CUDA_REDUCTION_KERNEL(MeanBackward)
REGISTER_CUDA_REDUCTION_KERNEL(SumBackward)
REGISTER_CUDA_REDUCTION_KERNEL(MaxBackward)
REGISTER_CUDA_REDUCTION_KERNEL(MinBackward)

#undef REGISTER_CUDA_REDUCTION_KERNEL

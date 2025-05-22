#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {
constexpr float kInfinity = std::numeric_limits<float>::infinity();
}

namespace {

// Generic reduction kernel
template <typename ReduceOp, typename FinalizeOp>
__global__ void GenericReduceKernel(const float *input, float *output, int64_t N, int64_t H, int64_t W,
                                    ReduceOp reduce_op, FinalizeOp finalize_op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * W) {
        return;
    }

    int n = idx / W;
    int w = idx % W;

    float acc = reduce_op.init();
    for (int64_t h = 0; h < H; ++h) {
        int input_idx = (n * H + h) * W + w;
        acc = reduce_op(acc, input[input_idx]);
    }
    output[n * W + w] = finalize_op(acc, H);
}

// Unified backward kernel for Mean, Sum, Max, and Min
__global__ void GenericReduceBackwardKernel(float *grad_input, const float *grad_output, const float *input,
                                            const float *reduced, int64_t N, int64_t H, int64_t W, bool is_mean,
                                            bool is_masked) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * W) {
        return;
    }

    int n = idx / (H * W);
    int hw = idx % (H * W);
    int w = hw % W;

    int reduced_idx = n * W + w;

    if (is_masked) {
        float selected = reduced[reduced_idx];
        float value = input[idx];
        grad_input[idx] = (value == selected) ? grad_output[reduced_idx] : 0.0f;
    } else {
        grad_input[idx] = grad_output[reduced_idx];
        if (is_mean) {
            grad_input[idx] /= static_cast<float>(H);
        }
    }
}

// Reduction operators
struct SumOp {
    __device__ __forceinline__ float init() const { return 0.0f; }
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

struct MaxOp {
    __device__ __forceinline__ float init() const { return -kInfinity; }
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct MinOp {
    __device__ __forceinline__ float init() const { return kInfinity; }
    __device__ __forceinline__ float operator()(float a, float b) const { return fminf(a, b); }
};

// Finalization strategies
struct MeanFinalize {
    __device__ __forceinline__ float operator()(float sum, int64_t count) const {
        return sum / static_cast<float>(count);
    }
};

struct IdentityFinalize {
    __device__ __forceinline__ float operator()(float val, int64_t) const { return val; }
};

} // namespace

// Common forward implementation for reduce ops
template <typename ReduceOp, typename FinalizeOp>
std::shared_ptr<Tensor> ReduceOpForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim,
                                        ReduceOp reduce_op, FinalizeOp finalize_op) {
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

    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + actual_dim, 1, std::multiplies<int64_t>());
    int64_t H = input_dims[actual_dim];
    int64_t W = std::accumulate(input_dims.begin() + actual_dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (N * W + threads_per_block - 1) / threads_per_block;

    GenericReduceKernel<<<num_blocks, threads_per_block>>>(input_ptr, output_ptr, N, H, W, reduce_op, finalize_op);

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

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + actual_dim, 1, std::multiplies<int64_t>());
    int64_t H = input_dims[actual_dim];
    int64_t W = std::accumulate(input_dims.begin() + actual_dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());

    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    const float *input_ptr = input ? static_cast<const float *>(input->DataPtr()) : nullptr;
    const float *reduced_ptr = reduced ? static_cast<const float *>(reduced->DataPtr()) : nullptr;

    int threads_per_block = 256;
    int num_blocks = (N * H * W + threads_per_block - 1) / threads_per_block;

    GenericReduceBackwardKernel<<<num_blocks, threads_per_block>>>(grad_input_ptr, grad_output_ptr, input_ptr,
                                                                   reduced_ptr, N, H, W, is_mean, is_masked);

    return grad_input;
}

std::shared_ptr<Tensor> MeanForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward(input, dim, keep_dim, SumOp{}, MeanFinalize{});
}

std::shared_ptr<Tensor> SumForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward(input, dim, keep_dim, SumOp{}, IdentityFinalize{});
}

std::shared_ptr<Tensor> MaxForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward(input, dim, keep_dim, MaxOp{}, IdentityFinalize{});
}

std::shared_ptr<Tensor> MinForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward(input, dim, keep_dim, MinOp{}, IdentityFinalize{});
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

// REGISTER_CUDA_REDUCTION_KERNEL(ReduceOpForward)
REGISTER_CUDA_REDUCTION_KERNEL(MeanForward)
REGISTER_CUDA_REDUCTION_KERNEL(SumForward)
REGISTER_CUDA_REDUCTION_KERNEL(MaxForward)
REGISTER_CUDA_REDUCTION_KERNEL(MinForward)
REGISTER_CUDA_REDUCTION_KERNEL(MeanBackward)
REGISTER_CUDA_REDUCTION_KERNEL(SumBackward)
REGISTER_CUDA_REDUCTION_KERNEL(MaxBackward)
REGISTER_CUDA_REDUCTION_KERNEL(MinBackward)

#undef REGISTER_CUDA_REDUCTION_KERNEL

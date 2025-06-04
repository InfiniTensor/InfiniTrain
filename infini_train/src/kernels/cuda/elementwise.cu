#include <cstdint>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

namespace {

template <typename T, typename Func>
__global__ void UnaryForwardKernel(T *output, Func fn, size_t num_elements, size_t offset, const T *input) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < num_elements) {
        output[idx] = fn(input[idx]);
    }
}

// Helper for broadcast indexing
__device__ inline int64_t CalcOffset(int64_t idx, int ndim, const int64_t *strides, const int64_t *shape,
                                     const int64_t *out_strides) {
    int64_t offset = 0;
    for (int i = 0; i < ndim; ++i) {
        int64_t out_index = (idx / out_strides[i]) % shape[i];
        int64_t index = shape[i] == 1 ? 0 : out_index;
        offset += index * strides[i];
    }
    return offset;
}

template <typename T, typename Func>
__global__ void BinaryForwardKernel(T *output, Func fn, int ndim, const int64_t *a_strides, const int64_t *a_shape,
                                    const int64_t *b_strides, const int64_t *b_shape, const int64_t *out_strides,
                                    const T *a, const T *b, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    int64_t a_offset = CalcOffset(idx, ndim, a_strides, a_shape, out_strides);
    int64_t b_offset = CalcOffset(idx, ndim, b_strides, b_shape, out_strides);

    output[idx] = fn(a[a_offset], b[b_offset]);
}

// launch the given kernel function with the given output and inputs
template <size_t BLOCK_SIZE, typename T, typename Kernel, typename... Inputs>
void LaunchKernel(Kernel &&kernel, const std::shared_ptr<Tensor> &output, const Inputs &...inputs) {
    auto extract_ptrs
        = [](const auto &...ts) { return std::make_tuple(static_cast<T *>(ts ? ts->DataPtr() : nullptr)...); };
    auto input_ptrs = extract_ptrs(inputs...);

    const size_t num_elements = output->NumElements();
    dim3 block_dims(std::min(BLOCK_SIZE, static_cast<size_t>(1024)));
    dim3 grid_dims(CEIL_DIV(num_elements, block_dims.x));
    const size_t step = grid_dims.x * block_dims.x;

    for (size_t offset = 0; offset < num_elements; offset += step) {
        std::apply([&](auto... ptrs) { kernel(grid_dims, block_dims, offset, ptrs...); }, input_ptrs);
    }
}

// Helper for stride calculation
std::vector<int64_t> ComputeStride(const std::vector<int64_t> &dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i) { strides[i] = strides[i + 1] * dims[i + 1]; }
    return strides;
}

// launch a forward elementwise operation given the calculation function, output, and the inputs
// Note: currently only support unary and binary operations
template <size_t BLOCK_SIZE, typename T, typename Func, typename... Inputs>
void LaunchForward(Func func, const std::shared_ptr<Tensor> &output, const Inputs &...inputs) {
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(output->GetDevice());
    const auto &stream = cuda_device->Stream();
    T *output_ptr = static_cast<T *>(output->DataPtr());

    if constexpr (sizeof...(inputs) == 1) {
        // Unary case
        LaunchKernel<BLOCK_SIZE, T>(
            [&](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
                UnaryForwardKernel<<<grid, block, 0, stream>>>(output_ptr, func, output->NumElements(), offset,
                                                               ptrs...);
            },
            output, inputs...);
    } else if constexpr (sizeof...(inputs) == 2) {
        // Binary case
        auto input_tuple = std::make_tuple(inputs...);
        const auto &input_a = std::get<0>(input_tuple);
        const auto &input_b = std::get<1>(input_tuple);

        const auto &a_dims = input_a->Dims();
        const auto &b_dims = input_b->Dims();
        const auto &out_dims = output->Dims();
        int ndim = out_dims.size();

        std::vector<int64_t> a_shape(ndim, 1), b_shape(ndim, 1), out_shape(ndim, 1);
        std::copy_backward(a_dims.begin(), a_dims.end(), a_shape.end());
        std::copy_backward(b_dims.begin(), b_dims.end(), b_shape.end());
        std::copy_backward(out_dims.begin(), out_dims.end(), out_shape.end());

        auto a_stride_host = ComputeStride(a_shape);
        auto b_stride_host = ComputeStride(b_shape);
        auto out_stride_host = ComputeStride(out_shape);

        int64_t *device_buffer;
        cudaMallocAsync(&device_buffer, 5 * ndim * sizeof(int64_t), stream);

        int64_t *device_a_strides, *device_b_strides, *device_out_strides, *device_a_shape, *device_b_shape;
        device_a_strides = device_buffer + ndim * 0;
        device_b_strides = device_buffer + ndim * 1;
        device_out_strides = device_buffer + ndim * 2;
        device_a_shape = device_buffer + ndim * 3;
        device_b_shape = device_buffer + ndim * 4;

        std::vector<int64_t> host_buffer;
        host_buffer.insert(host_buffer.end(), a_stride_host.begin(), a_stride_host.end());
        host_buffer.insert(host_buffer.end(), b_stride_host.begin(), b_stride_host.end());
        host_buffer.insert(host_buffer.end(), out_stride_host.begin(), out_stride_host.end());
        host_buffer.insert(host_buffer.end(), a_shape.begin(), a_shape.end());
        host_buffer.insert(host_buffer.end(), b_shape.begin(), b_shape.end());

        cudaMemcpyAsync(device_buffer, host_buffer.data(), 5 * ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        LaunchKernel<BLOCK_SIZE, T>(
            [&](dim3 grid, dim3 block, size_t offset, const T *a_ptr, const T *b_ptr) {
                BinaryForwardKernel<<<grid, block, 0, stream>>>(
                    output_ptr, func, ndim, device_a_strides, device_a_shape, device_b_strides, device_b_shape,
                    device_out_strides, a_ptr, b_ptr, output->NumElements());
            },
            output, inputs...);

        cudaFreeAsync(device_buffer, stream);
    } else {
        static_assert(sizeof...(inputs) == 1 || sizeof...(inputs) == 2,
                      "LaunchForward currently only supports unary and binary operations.");
    }
}

// Backward kernel for unary operators
template <typename T, typename Func>
__global__ void UnaryBackwardKernel(T *output, Func fn, size_t num_elements, size_t offset, const T *grad_output,
                                    const T *input) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < num_elements) {
        output[idx] = grad_output[idx] * fn(input ? input[idx] : T(0));
    }
}

// Backward kernel for binary operators
template <typename T, typename FuncA, typename FuncB>
__global__ void BinaryBackwardKernel(T *output_a, T *output_b, FuncA fn_a, FuncB fn_b, int ndim, size_t num_elements,
                                     const int64_t *a_strides, const int64_t *a_shape, const int64_t *b_strides,
                                     const int64_t *b_shape, const int64_t *out_strides, const T *grad_output,
                                     const T *input_a, const T *input_b) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    int64_t a_offset = CalcOffset(idx, ndim, a_strides, a_shape, out_strides);
    int64_t b_offset = CalcOffset(idx, ndim, b_strides, b_shape, out_strides);

    const T a_val = input_a ? input_a[a_offset] : T(0);
    const T b_val = input_b ? input_b[b_offset] : T(0);

    output_a[a_offset] = grad_output[idx] * fn_a(a_val, b_val);
    atomicAdd(&output_b[b_offset], grad_output[idx] * fn_b(a_val, b_val));
}

// launch unary operator's backward kernel
template <size_t BLOCK_SIZE, typename T, typename Func, typename... Inputs>
void LaunchBackward(Func func, const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &grad_output,
                    const Inputs &...inputs) {
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(output->GetDevice());
    T *output_ptr = static_cast<T *>(output->DataPtr());
    const T *grad_ptr = static_cast<const T *>(grad_output->DataPtr());

    LaunchKernel<BLOCK_SIZE, T>(
        [=](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
            UnaryBackwardKernel<<<grid, block, 0, cuda_device->Stream()>>>(output_ptr, func, output->NumElements(),
                                                                           offset, grad_ptr, ptrs...);
        },
        output, inputs...);
}

// launch binary operator's backward kernel
template <size_t BLOCK_SIZE, typename T, typename FuncA, typename FuncB, typename... Inputs>
void LaunchBackward(FuncA fun_a, FuncB fun_b, const std::shared_ptr<Tensor> &output_a,
                    const std::shared_ptr<Tensor> &output_b, const std::vector<int64_t> &a_dims,
                    const std::vector<int64_t> &b_dims, const std::shared_ptr<Tensor> &grad_output,
                    const Inputs &...inputs) {
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(output_a->GetDevice());
    const auto &stream = cuda_device->Stream();
    T *output_a_ptr = static_cast<T *>(output_a->DataPtr());
    T *output_b_ptr = static_cast<T *>(output_b->DataPtr());
    const T *grad_output_ptr = static_cast<const T *>(grad_output->DataPtr());

    const auto &out_dims = grad_output->Dims();
    int ndim = out_dims.size();

    std::vector<int64_t> a_shape(ndim, 1), b_shape(ndim, 1), out_shape(ndim, 1);
    std::copy_backward(a_dims.begin(), a_dims.end(), a_shape.end());
    std::copy_backward(b_dims.begin(), b_dims.end(), b_shape.end());
    std::copy_backward(out_dims.begin(), out_dims.end(), out_shape.end());

    auto a_stride_host = ComputeStride(a_shape);
    auto b_stride_host = ComputeStride(b_shape);
    auto out_stride_host = ComputeStride(out_shape);

    int64_t *device_buffer;
    cudaMallocAsync(&device_buffer, 5 * ndim * sizeof(int64_t), stream);

    int64_t *device_a_strides, *device_b_strides, *device_out_strides, *device_a_shape, *device_b_shape;
    device_a_strides = device_buffer + ndim * 0;
    device_b_strides = device_buffer + ndim * 1;
    device_out_strides = device_buffer + ndim * 2;
    device_a_shape = device_buffer + ndim * 3;
    device_b_shape = device_buffer + ndim * 4;

    std::vector<int64_t> host_buffer;
    host_buffer.insert(host_buffer.end(), a_stride_host.begin(), a_stride_host.end());
    host_buffer.insert(host_buffer.end(), b_stride_host.begin(), b_stride_host.end());
    host_buffer.insert(host_buffer.end(), out_stride_host.begin(), out_stride_host.end());
    host_buffer.insert(host_buffer.end(), a_shape.begin(), a_shape.end());
    host_buffer.insert(host_buffer.end(), b_shape.begin(), b_shape.end());

    cudaMemcpyAsync(device_buffer, host_buffer.data(), 5 * ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    const size_t num_elements = grad_output->NumElements();
    LaunchKernel<BLOCK_SIZE, T>(
        [=](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
            BinaryBackwardKernel<<<grid, block, 0, stream>>>(
                output_a_ptr, output_b_ptr, fun_a, fun_b, ndim, num_elements, device_a_strides, device_a_shape,
                device_b_strides, device_b_shape, device_out_strides, grad_output_ptr, ptrs...);
        },
        output_a, inputs...);

    cudaFreeAsync(device_buffer, stream);
}

template <typename Func> std::shared_ptr<Tensor> UnaryForward(const std::shared_ptr<Tensor> &input, Func unary_fn) {
    auto dtype = input->Dtype();
    auto output = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());

    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchForward<256, float>(unary_fn, output, input);
        break;
    default:
        LOG(FATAL) << "CUDA unary forward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    return output;
}

template <typename Func>
std::shared_ptr<Tensor> UnaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
                                      Func unary_fn) {
    auto dtype = grad_output->Dtype();
    auto output = std::make_shared<Tensor>(grad_output->Dims(), dtype, grad_output->GetDevice());
    output->Fill<float>(0.0f);
    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchBackward<256, float>(unary_fn, output, grad_output, a);
        break;
    default:
        LOG(FATAL) << "CUDA unary backward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    return output;
}

template <typename Func>
std::shared_ptr<Tensor> BinaryForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b,
                                      Func binary_fn) {
    auto dtype = a->Dtype();
    // Currently a and b should have the same data type and only one-way broadcasting from b to a is assumed by default
    CHECK(dtype == b->Dtype() && a->NumElements() >= b->NumElements() && a->NumElements() % b->NumElements() == 0);

    auto output = std::make_shared<Tensor>(a->Dims(), dtype, a->GetDevice());

    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchForward<256, float>(binary_fn, output, a, b);
        break;
    default:
        LOG(FATAL) << "CUDA binary forward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    return output;
}

template <typename FuncA, typename FuncB>
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
BinaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
               const std::shared_ptr<Tensor> &b, const std::vector<int64_t> &a_dims, const std::vector<int64_t> &b_dims,
               FuncA fn_a, FuncB fn_b) {
    const auto a_num_elements = std::accumulate(a_dims.begin(), a_dims.end(), 1, std::multiplies<int64_t>());
    const auto b_num_elements = std::accumulate(b_dims.begin(), b_dims.end(), 1, std::multiplies<int64_t>());

    CHECK(a_num_elements >= b_num_elements && a_num_elements % b_num_elements == 0);
    if (a) {
        CHECK(a_num_elements == a->NumElements());
    }
    if (b) {
        CHECK(b_num_elements == b->NumElements());
    }
    auto dtype = grad_output->Dtype();
    auto device = grad_output->GetDevice();

    // Currently a and b should have the same data type
    if (a && b) {
        CHECK(a->Dtype() == b->Dtype());
    }
    auto grad_a = std::make_shared<Tensor>(a_dims, dtype, device);
    auto grad_b = std::make_shared<Tensor>(b_dims, dtype, device);
    grad_a->Fill<float>(0.0f);
    grad_b->Fill<float>(0.0f);
    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchBackward<256, float>(fn_a, fn_b, grad_a, grad_b, a_dims, b_dims, grad_output, a, b);
        break;
    default:
        LOG(FATAL) << "CUDA binary backward: 'Unsupported data type' at " << __FILE__ << ":" << __LINE__;
    }

    return {grad_a, grad_b};
}
} // namespace

std::shared_ptr<Tensor> NegForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [] __device__(float x) { return -x; });
}

std::shared_ptr<Tensor> NegBackward(const std::shared_ptr<Tensor> &grad_output) {
    return UnaryBackward(grad_output, nullptr, [] __device__(float) { return -1.0f; });
}

std::shared_ptr<Tensor> ReciprocalForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [] __device__(float x) { return 1.0f / x; });
}

std::shared_ptr<Tensor> ReciprocalBackward(const std::shared_ptr<Tensor> &grad_output,
                                           const std::shared_ptr<Tensor> &input) {
    return UnaryBackward(grad_output, input, [] __device__(float x) { return -1.0f / (x * x); });
}

std::shared_ptr<Tensor> SinForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [] __device__(float x) { return sinf(x); });
}

std::shared_ptr<Tensor> SinBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    return UnaryBackward(grad_output, input, [] __device__(float x) { return cosf(x); });
}

std::shared_ptr<Tensor> CosForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [] __device__(float x) { return cosf(x); });
}

std::shared_ptr<Tensor> CosBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    return UnaryBackward(grad_output, input, [] __device__(float x) { return -sinf(x); });
}

std::shared_ptr<Tensor> TanhForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [] __device__(float x) { return tanhf(x); });
}

std::shared_ptr<Tensor> TanhBackward(const std::shared_ptr<Tensor> &grad_output,
                                     const std::shared_ptr<Tensor> &output) {
    return UnaryBackward(grad_output, output, [] __device__(float x) { return 1.0 - x * x; });
}

std::shared_ptr<Tensor> PowForward(const std::shared_ptr<Tensor> &input, float scalar, bool scalar_is_base) {
    if (scalar_is_base) {
        return UnaryForward(input, [scalar] __device__(float x) { return powf(scalar, x); });
    } else {
        return UnaryForward(input, [scalar] __device__(float x) { return powf(x, scalar); });
    }
}

std::shared_ptr<Tensor> PowBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    float scalar, bool scalar_is_base) {
    if (scalar_is_base) {
        return UnaryBackward(grad_output, input,
                             [scalar] __device__(float x) { return logf(scalar) * powf(scalar, x); });
    } else {
        return UnaryBackward(grad_output, input,
                             [scalar] __device__(float x) { return scalar * powf(x, scalar - 1.0f); });
    }
}

std::shared_ptr<Tensor> RsqrtForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [] __device__(float x) { return 1.0f / sqrtf(x); });
}

std::shared_ptr<Tensor> RsqrtBackward(const std::shared_ptr<Tensor> &grad_output,
                                      const std::shared_ptr<Tensor> &input) {
    return UnaryBackward(grad_output, input, [] __device__(float x) { return -0.5f / (x * sqrtf(x)); });
}

std::shared_ptr<Tensor> EqualsScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar] __device__(float x) { return x == scalar ? 1.0f : 0.0f; });
}

std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [] __device__(float x, float y) { return x + y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims) {
    return BinaryBackward(
        grad_output, nullptr, nullptr, a_dims, b_dims, [] __device__(float, float) { return 1.f; },
        [] __device__(float, float) { return 1.f; });
}

std::shared_ptr<Tensor> AddScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar] __device__(float x) { return x + scalar; });
}

std::shared_ptr<Tensor> AddScalarBackward(const std::shared_ptr<Tensor> &grad_output) {
    return UnaryBackward(grad_output, nullptr, [] __device__(float) { return 1.0f; });
}

std::shared_ptr<Tensor> SubForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [] __device__(float x, float y) { return x - y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> SubBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims) {
    return BinaryBackward(
        grad_output, nullptr, nullptr, a_dims, b_dims, [] __device__(float, float) { return 1.f; },
        [] __device__(float, float) { return -1.f; });
}

std::shared_ptr<Tensor> MulForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [] __device__(float x, float y) { return x * y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MulBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b) {
    return BinaryBackward(
        grad_output, a, b, a->Dims(), b->Dims(), [] __device__(float, float y) { return y; },
        [] __device__(float x, float) { return x; });
}

std::shared_ptr<Tensor> MulScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar] __device__(float x) { return x * scalar; });
}

std::shared_ptr<Tensor> MulScalarBackward(const std::shared_ptr<Tensor> &grad_output, float scalar) {
    return UnaryBackward(grad_output, nullptr, [scalar] __device__(float) { return scalar; });
}

std::shared_ptr<Tensor> DivForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [] __device__(float x, float y) { return x / y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DivBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b) {
    return BinaryBackward(
        grad_output, a, b, a->Dims(), b->Dims(), [] __device__(float, float y) { return 1 / y; },
        [] __device__(float x, float y) { return -x / (y * y); });
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ELEMENTWISE_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ELEMENTWISE_KERNEL(NegForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(NegBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(ReciprocalForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(ReciprocalBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(SinForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(SinBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(CosForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(CosBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(TanhForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(TanhBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(PowForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(PowBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(RsqrtForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(RsqrtBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(EqualsScalarForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(AddForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(AddBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(AddScalarForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(AddScalarBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(SubForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(SubBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(MulForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(MulBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(MulScalarForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(MulScalarBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(DivForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(DivBackward)

#undef REGISTER_CUDA_ELEMENTWISE_KERNEL

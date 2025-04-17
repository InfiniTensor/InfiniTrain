#include "infini_train/include/kernels/cuda/elementwise.h"

#include <cmath>
#include <functional>
#include <memory>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

namespace {

template <typename T, typename Func, typename... Inputs>
__global__ void ElementwiseForwardKernel(T *output, Func fn, size_t num_elements, size_t offset, Inputs... inputs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < num_elements) {
        output[idx] = fn(inputs[idx]...);
    }
}

template <size_t BLOCK_SIZE, typename T, typename Kernel, typename... Inputs>
void LaunchKernel(Kernel &&kernel, const std::shared_ptr<Tensor> &output, const Inputs &...inputs) {
    auto extract_ptrs = [](const auto &...ts) { return std::make_tuple(static_cast<T *>(ts->DataPtr())...); };
    auto input_ptrs = extract_ptrs(inputs...);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, output->GetDevice().Index());

    const size_t num_elements = output->NumElements();
    dim3 block_dims(std::min(BLOCK_SIZE, static_cast<size_t>(prop.maxThreadsPerBlock)));
    dim3 grid_dims(std::min(CEIL_DIV(num_elements, block_dims.x), static_cast<size_t>(prop.maxGridSize[0])));
    const size_t step = grid_dims.x * block_dims.x;

    for (size_t offset = 0; offset < num_elements; offset += step) {
        std::apply([&](auto... ptrs) { kernel(grid_dims, block_dims, offset, ptrs...); }, input_ptrs);
    }
}

// launch the kernel function given the output, inputs, and the operation function
template <size_t BLOCK_SIZE, typename T, typename Func, typename... Inputs>
void LaunchForward(Func func, const std::shared_ptr<Tensor> &output, const Inputs &...inputs) {
    T *output_ptr = static_cast<T *>(output->DataPtr());

    LaunchKernel<BLOCK_SIZE, T>(
        [&](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
            ElementwiseForwardKernel<<<grid, block>>>(output_ptr, func, output->NumElements(), offset, ptrs...);
        },
        output, inputs...);
}

// Backward kernel for unary operators
template <typename T, typename Func, typename... Inputs>
__global__ void ElementwiseBackwardKernel(T *output, Func fn, size_t num_elements, size_t offset, const T *grad_output,
                                          Inputs... inputs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < num_elements) {
        output[idx] = grad_output[idx] * fn(inputs[idx]...);
    }
}

// Backward kernel for binary operators
template <typename T, typename FuncA, typename FuncB, typename... Inputs>
__global__ void ElementwiseBackwardKernel(T *output_a, T *output_b, FuncA fun_a, FuncB fun_b, size_t num_elements,
                                          size_t offset, const T *grad_output, Inputs... inputs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < num_elements) {
        output_a[idx] = grad_output[idx] * fun_a(inputs[idx]...);
        output_b[idx] = grad_output[idx] * fun_b(inputs[idx]...);
    }
}

template <size_t BLOCK_SIZE, typename T, typename Func, typename... Inputs>
void LaunchBackward(Func func, const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &grad_output,
                    const Inputs &...inputs) {
    T *output_ptr = static_cast<T *>(output->DataPtr());
    const T *grad_ptr = static_cast<const T *>(grad_output->DataPtr());

    LaunchKernel<BLOCK_SIZE, T>(
        [=](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
            ElementwiseBackwardKernel<<<grid, block>>>(output_ptr, func, output->NumElements(), offset, grad_ptr,
                                                       ptrs...);
        },
        output, inputs...);
}

template <size_t BLOCK_SIZE, typename T, typename FuncA, typename FuncB, typename... Inputs>
void LaunchBackward(FuncA fun_a, FuncB fun_b, const std::shared_ptr<Tensor> &output_a,
                    const std::shared_ptr<Tensor> &output_b, const std::shared_ptr<Tensor> &grad_output,
                    const Inputs &...inputs) {
    size_t num_elements = output_a->NumElements();
    T *output_a_ptr = static_cast<T *>(output_a->DataPtr());
    T *output_b_ptr = static_cast<T *>(output_b->DataPtr());
    const T *grad_output_ptr = static_cast<const T *>(grad_output->DataPtr());

    LaunchKernel<BLOCK_SIZE, T>(
        [=](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
            ElementwiseBackwardKernel<<<grid, block>>>(output_a_ptr, output_b_ptr, fun_a, fun_b,
                                                       output_a->NumElements(), offset, grad_output_ptr, ptrs...);
        },
        output_a, inputs...);
}

template <typename Func> std::shared_ptr<Tensor> UnaryForward(const std::shared_ptr<Tensor> &input, Func unary_fn) {
    auto dtype = input->Dtype();
    auto output = std::make_shared<Tensor>(input->Dims(), dtype, Device(DeviceType::kCUDA, 0));

    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchForward<256, float>(unary_fn, output, input);
        break;
    default:
        return nullptr;
    }

    return output;
}

template <typename Func>
std::shared_ptr<Tensor> UnaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
                                      Func unary_fn) {
    auto dtype = grad_output->Dtype();
    auto output = std::make_shared<Tensor>(grad_output->Dims(), dtype, Device(DeviceType::kCUDA, 0));

    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchBackward<256, float>(unary_fn, output, grad_output, a);
        break;
    default:
        return nullptr;
    }

    return output;
}

template <typename Func>
std::shared_ptr<Tensor> BinaryForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b,
                                      Func binary_fn) {
    auto dtype = a->Dtype();
    // Currently a and b should have the same data type and the same total number of elements
    CHECK(dtype == b->Dtype() && a->NumElements() == b->NumElements());

    auto output = std::make_shared<Tensor>(a->Dims(), dtype, Device(DeviceType::kCUDA, 0));

    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchForward<256, float>(binary_fn, output, a, b);
        break;
    default:
        return nullptr;
    }

    return output;
}

template <typename FuncA, typename FuncB>
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
BinaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
               const std::shared_ptr<Tensor> &b, FuncA fn_a, FuncB fn_b) {
    auto dtype = grad_output->Dtype();
    // Currently a and b should have the same data type and the same total number of elements
    CHECK(dtype == b->Dtype() && a->NumElements() == b->NumElements());
    auto grad_a = std::make_shared<Tensor>(a->Dims(), dtype, Device(DeviceType::kCUDA, 0));
    auto grad_b = std::make_shared<Tensor>(b->Dims(), dtype, Device(DeviceType::kCUDA, 0));

    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchBackward<256, float>(fn_a, fn_b, grad_a, grad_b, grad_output, a, b);
        break;
    default:
        return {nullptr, nullptr};
    }

    return {grad_a, grad_b};
}
} // namespace

std::shared_ptr<Tensor> TanhForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [] __device__(float x) { return tanhf(x); });
}

std::shared_ptr<Tensor> TanhBackward(const std::shared_ptr<Tensor> &grad_output,
                                     const std::shared_ptr<Tensor> &output) {
    return UnaryBackward(grad_output, output, [] __device__(float x) { return 1.0 - x * x; });
}

std::shared_ptr<Tensor> PowForward(const std::shared_ptr<Tensor> &input, float exponent) {
    return UnaryForward(input, [exponent] __device__(float x) { return powf(x, exponent); });
}

std::shared_ptr<Tensor> PowBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    float exponent) {
    return UnaryBackward(grad_output, input,
                         [exponent] __device__(float x) { return exponent * powf(x, exponent - 1.0f); });
}

std::shared_ptr<Tensor> EqualsScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar] __device__(float x) { return x == scalar ? 1.0f : 0.0f; });
}

std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [] __device__(float x, float y) { return x + y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(const std::shared_ptr<Tensor> &grad_output) {
    return BinaryBackward(
        grad_output, nullptr, nullptr, [] __device__(float, float) { return 1.0f; },
        [] __device__(float, float) { return 1.0f; });
}

std::shared_ptr<Tensor> AddScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar] __device__(float x) { return x + scalar; });
}

std::shared_ptr<Tensor> AddScalarBackward(const std::shared_ptr<Tensor> &grad_output) {
    return UnaryBackward(grad_output, nullptr, [] __device__(float) { return 1.0f; });
}

std::shared_ptr<Tensor> MulForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [] __device__(float x, float y) { return x * y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MulBackward(const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b,
                                                                        const std::shared_ptr<Tensor> &grad_output) {
    return BinaryBackward(
        grad_output, a, b, [] __device__(float, float y) { return y; }, [] __device__(float x, float) { return x; });
}

std::shared_ptr<Tensor> MulScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar] __device__(float x) { return x * scalar; });
}

std::shared_ptr<Tensor> MulScalarBackward(const std::shared_ptr<Tensor> &grad_output, float scalar) {
    return UnaryBackward(grad_output, nullptr, [scalar] __device__(float) { return scalar; });
}
} // namespace infini_train::kernels::cuda

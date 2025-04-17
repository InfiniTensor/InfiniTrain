#include "infini_train/include/kernels/cuda/elementwise.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "glog/logging.h"

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

template <typename T, typename Func>
__global__ void BinaryForwardKernel(T *output, Func fn, size_t num_elements_a, size_t num_elements_b, size_t offset,
                                    const T *a, const T *b) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < num_elements_a) {
        output[idx] = fn(a[idx], b[idx % num_elements_b]);
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

    if constexpr (sizeof...(inputs) == 1) {
        // Unary case
        LaunchKernel<BLOCK_SIZE, T>(
            [&](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
                UnaryForwardKernel<<<grid, block>>>(output_ptr, func, output->NumElements(), offset, ptrs...);
            },
            output, inputs...);
    } else if constexpr (sizeof...(inputs) == 2) {
        // Binary case
        auto input_tuple = std::make_tuple(inputs...);
        const auto &input_a = std::get<0>(input_tuple);
        const auto &input_b = std::get<1>(input_tuple);

        LaunchKernel<BLOCK_SIZE, T>(
            [&](dim3 grid, dim3 block, size_t offset, const T *a_ptr, const T *b_ptr) {
                BinaryForwardKernel<<<grid, block>>>(output_ptr, func, input_a->NumElements(), input_b->NumElements(),
                                                     offset, a_ptr, b_ptr);
            },
            output, inputs...);
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
__global__ void BinaryBackwardKernel(T *output_a, T *output_b, FuncA fun_a, FuncB fun_b, int64_t a_num_elements,
                                     int64_t b_num_elements, size_t offset, const T *grad_output, const T *input_a,
                                     const T *input_b) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < a_num_elements) {
        const T &a = input_a ? input_a[idx] : T(0);
        const T &b = input_b ? input_b[idx % b_num_elements] : T(0);
        output_a[idx] = grad_output[idx] * fun_a(a, b);
        output_b[idx % b_num_elements] = grad_output[idx] * fun_b(a, b);
    }
}

// launch unary operator's backward kernel
template <size_t BLOCK_SIZE, typename T, typename Func, typename... Inputs>
void LaunchBackward(Func func, const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &grad_output,
                    const Inputs &...inputs) {
    T *output_ptr = static_cast<T *>(output->DataPtr());
    const T *grad_ptr = static_cast<const T *>(grad_output->DataPtr());

    LaunchKernel<BLOCK_SIZE, T>(
        [=](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
            UnaryBackwardKernel<<<grid, block>>>(output_ptr, func, output->NumElements(), offset, grad_ptr, ptrs...);
        },
        output, inputs...);
}

// launch binary operator's backward kernel
template <size_t BLOCK_SIZE, typename T, typename FuncA, typename FuncB, typename... Inputs>
void LaunchBackward(FuncA fun_a, FuncB fun_b, const std::shared_ptr<Tensor> &output_a,
                    const std::shared_ptr<Tensor> &output_b, int64_t a_num_elements, int64_t b_num_elements,
                    const std::shared_ptr<Tensor> &grad_output, const Inputs &...inputs) {
    T *output_a_ptr = static_cast<T *>(output_a->DataPtr());
    T *output_b_ptr = static_cast<T *>(output_b->DataPtr());
    const T *grad_output_ptr = static_cast<const T *>(grad_output->DataPtr());

    LaunchKernel<BLOCK_SIZE, T>(
        [=](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
            BinaryBackwardKernel<<<grid, block>>>(output_a_ptr, output_b_ptr, fun_a, fun_b, a_num_elements,
                                                  b_num_elements, offset, grad_output_ptr, ptrs...);
        },
        output_a, inputs...);
}

template <typename Func> std::shared_ptr<Tensor> UnaryForward(const std::shared_ptr<Tensor> &input, Func unary_fn) {
    auto dtype = input->Dtype();
    auto output = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());

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
    auto output = std::make_shared<Tensor>(grad_output->Dims(), dtype, grad_output->GetDevice());

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
    // Currently a and b should have the same data type and only one-way broadcasting from b to a is assumed by default
    CHECK(dtype == b->Dtype() && a->NumElements() >= b->NumElements() && a->NumElements() % b->NumElements() == 0);

    auto output = std::make_shared<Tensor>(a->Dims(), dtype, a->GetDevice());

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
    auto device = a->GetDevice();
    // Currently a and b should have the same data type
    CHECK(dtype == b->Dtype());
    auto grad_a = std::make_shared<Tensor>(a->Dims(), dtype, device);
    auto grad_b = std::make_shared<Tensor>(b->Dims(), dtype, device);

    switch (dtype) {
    case DataType::kFLOAT32:
        LaunchBackward<256, float>(fn_a, fn_b, grad_a, grad_b, a_num_elements, b_num_elements, grad_output, a, b);
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

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims) {
    return BinaryBackward(
        grad_output, nullptr, nullptr, a_dims, b_dims, [] __device__(float, float) { return 1.0f; },
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
        grad_output, a, b, a->Dims(), b->Dims(), [] __device__(float, float y) { return y; },
        [] __device__(float x, float) { return x; });
}

std::shared_ptr<Tensor> MulScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar] __device__(float x) { return x * scalar; });
}

std::shared_ptr<Tensor> MulScalarBackward(const std::shared_ptr<Tensor> &grad_output, float scalar) {
    return UnaryBackward(grad_output, nullptr, [scalar] __device__(float) { return scalar; });
}
} // namespace infini_train::kernels::cuda

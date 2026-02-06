#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::kernels::cuda {
__device__ __forceinline__ int64_t UpperBoundI64(const int64_t *offsets, int64_t n_plus_1, int64_t x) {
    // Return the largest s so that offsets[s] <= x
    // offsets[0] = 0, offsets is monotonically increasing
    // len(offsets) = num_inputs + 1
    int64_t l = 0, r = n_plus_1; // start search in [0, n+1)
    while (l < r) {
        int64_t m = l + ((r - l) >> 1);
        if (offsets[m] <= x) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l - 1;
}

template <typename T>
__global__ void ConcatForwardKernel(const T **inputs, T *output, const int64_t *offsets, int64_t N, int64_t D,
                                    int64_t num_inputs, int64_t K_total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * K_total * D;
    if (idx >= total) {
        return;
    }

    int64_t d = idx % D;
    int64_t k = (idx / D) % K_total;
    int64_t n = idx / (D * K_total);

    // find the largest s so that offsets[s] <= k < offsets[s+1]
    int64_t s = UpperBoundI64(offsets, num_inputs + 1, k);
    int64_t k_local = k - offsets[s];
    int64_t Ki = offsets[s + 1] - offsets[s];

    const T *input = inputs[s];
    output[idx] = input[n * (Ki * D) + k_local * D + d];
}

std::shared_ptr<Tensor> ConcatForward(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    CHECK(!inputs.empty());

    const auto &base_dims = inputs[0]->Dims();
    auto dtype = inputs[0]->Dtype();
    auto device = inputs[0]->GetDevice();

    if (dim < 0) {
        dim += static_cast<int64_t>(base_dims.size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(base_dims.size()));

    // Check shape requirements and save length along dim
    std::vector<int64_t> Ks;
    Ks.reserve(inputs.size());
    for (const auto &t : inputs) {
        CHECK(t->Dtype() == dtype);
        CHECK_EQ(t->Dims().size(), base_dims.size());
        for (size_t ax = 0; ax < base_dims.size(); ++ax) {
            if (static_cast<int64_t>(ax) == dim) {
                continue;
            }
            CHECK_EQ(t->Dims()[ax], base_dims[ax]) << "All non-concat dims must match";
        }
        Ks.push_back(t->Dims()[dim]);
    }

    std::vector<int64_t> out_dims = base_dims;
    out_dims[dim] = std::accumulate(Ks.begin(), Ks.end(), int64_t{0});
    auto output = std::make_shared<Tensor>(out_dims, dtype, device);

    const int64_t N = std::accumulate(base_dims.begin(), base_dims.begin() + dim, 1LL, std::multiplies<int64_t>());
    const int64_t D = std::accumulate(base_dims.begin() + dim + 1, base_dims.end(), 1LL, std::multiplies<int64_t>());
    const int64_t num_inputs = static_cast<int64_t>(inputs.size());
    const int64_t K_total = out_dims[dim];

    // offsets records the sum of Ks
    // offsets[i] = sum_{j < i} K_j
    std::vector<int64_t> host_offsets(num_inputs + 1, 0);
    for (int64_t i = 0; i < num_inputs; ++i) { host_offsets[i + 1] = host_offsets[i] + Ks[i]; }

    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();

    int64_t total = N * K_total * D;
    int threads_per_block = 256;
    int num_blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);

    DispatchFunc<INFINI_ALL_TYPES>(
        dtype,
        [=, &inputs, &host_offsets]<typename T>() {
            std::vector<const T *> host_input_ptrs;
            host_input_ptrs.reserve(inputs.size());
            for (const auto &t : inputs) { host_input_ptrs.push_back(static_cast<const T *>(t->DataPtr())); }

            const T **device_input_ptrs = nullptr;
            int64_t *device_offsets = nullptr;

            CUDA_CHECK(cudaMallocAsync(&device_input_ptrs, sizeof(T *) * num_inputs, stream));
            CUDA_CHECK(cudaMemcpyAsync(device_input_ptrs, host_input_ptrs.data(), sizeof(T *) * num_inputs,
                                       cudaMemcpyHostToDevice, stream));

            CUDA_CHECK(cudaMallocAsync(&device_offsets, sizeof(int64_t) * (num_inputs + 1), stream));
            CUDA_CHECK(cudaMemcpyAsync(device_offsets, host_offsets.data(), sizeof(int64_t) * (num_inputs + 1),
                                       cudaMemcpyHostToDevice, stream));

            ConcatForwardKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
                device_input_ptrs, static_cast<T *>(output->DataPtr()), device_offsets, N, D, num_inputs, K_total);

            CUDA_CHECK(cudaFreeAsync(device_input_ptrs, stream));
            CUDA_CHECK(cudaFreeAsync(device_offsets, stream));
        },
        "CUDA ConcatForward");

    return output;
}

template <typename T>
__global__ void ConcatBackwardKernel(const T *grad_output, T **grad_inputs, const int64_t *offsets, int64_t N,
                                     int64_t D, int64_t num_inputs, int64_t K_total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * K_total * D;
    if (idx >= total) {
        return;
    }

    int64_t d = idx % D;
    int64_t k = (idx / D) % K_total;
    int64_t n = idx / (D * K_total);

    int64_t s = UpperBoundI64(offsets, num_inputs + 1, k);
    int64_t k_local = k - offsets[s];
    int64_t Ki = offsets[s + 1] - offsets[s];

    T *gi = grad_inputs[s];
    gi[n * (Ki * D) + k_local * D + d] = grad_output[idx];
}

std::vector<std::shared_ptr<Tensor>> ConcatBackward(const std::shared_ptr<Tensor> &grad_output,
                                                    const std::vector<std::vector<int64_t>> &input_dims_list,
                                                    int64_t dim) {
    CHECK(!input_dims_list.empty());

    auto dtype = grad_output->Dtype();
    const auto &output_dims = grad_output->Dims();
    if (dim < 0) {
        dim += static_cast<int64_t>(output_dims.size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(output_dims.size()));

    const auto &base_rank = input_dims_list[0].size();
    std::vector<int64_t> Ks;
    Ks.reserve(input_dims_list.size());
    for (const auto &dvec : input_dims_list) {
        CHECK_EQ(dvec.size(), base_rank);
        for (size_t ax = 0; ax < dvec.size(); ++ax) {
            if (static_cast<int64_t>(ax) == dim) {
                continue;
            }
            CHECK_EQ(dvec[ax], input_dims_list[0][ax]);
        }
        Ks.push_back(dvec[dim]);
    }

    auto device = grad_output->GetDevice();

    std::vector<std::shared_ptr<Tensor>> grads;
    grads.reserve(input_dims_list.size());
    for (const auto &dvec : input_dims_list) {
        auto t = std::make_shared<Tensor>(dvec, dtype, device);
        DispatchFunc<INFINI_ALL_TYPES>(
            dtype, [=]<typename T>() { t->Fill<T>(0); }, "CUDA ConcatBackward");
        grads.push_back(t);
    }

    const int64_t N = std::accumulate(input_dims_list[0].begin(), input_dims_list[0].begin() + dim, 1LL,
                                      std::multiplies<int64_t>());
    const int64_t D = std::accumulate(input_dims_list[0].begin() + dim + 1, input_dims_list[0].end(), 1LL,
                                      std::multiplies<int64_t>());
    const int64_t num_inputs = static_cast<int64_t>(input_dims_list.size());
    const int64_t K_total = std::accumulate(Ks.begin(), Ks.end(), int64_t{0});

    std::vector<int64_t> host_offsets(num_inputs + 1, 0);
    for (int64_t i = 0; i < num_inputs; ++i) { host_offsets[i + 1] = host_offsets[i] + Ks[i]; }

    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();

    int64_t total = N * K_total * D;
    int threads_per_block = 256;
    int num_blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);

    DispatchFunc<INFINI_ALL_TYPES>(
        dtype,
        [=, &grads, &host_offsets]<typename T>() {
            std::vector<T *> host_ptrs;
            host_ptrs.reserve(grads.size());
            for (auto &t : grads) { host_ptrs.push_back(static_cast<T *>(t->DataPtr())); }

            T **device_ptrs = nullptr;
            int64_t *device_offsets = nullptr;

            CUDA_CHECK(cudaMallocAsync(&device_ptrs, sizeof(T *) * num_inputs, stream));
            CUDA_CHECK(cudaMemcpyAsync(device_ptrs, host_ptrs.data(), sizeof(T *) * num_inputs, cudaMemcpyHostToDevice,
                                       stream));

            CUDA_CHECK(cudaMallocAsync(&device_offsets, sizeof(int64_t) * (num_inputs + 1), stream));
            CUDA_CHECK(cudaMemcpyAsync(device_offsets, host_offsets.data(), sizeof(int64_t) * (num_inputs + 1),
                                       cudaMemcpyHostToDevice, stream));

            ConcatBackwardKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), device_ptrs, device_offsets, N, D, num_inputs, K_total);

            CUDA_CHECK(cudaFreeAsync(device_ptrs, stream));
            CUDA_CHECK(cudaFreeAsync(device_offsets, stream));
        },
        "CUDA ConcatBackward");

    return grads;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CONCAT_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CONCAT_KERNEL(ConcatForward)
REGISTER_CUDA_CONCAT_KERNEL(ConcatBackward)

#undef REGISTER_CUDA_CONCAT_KERNEL

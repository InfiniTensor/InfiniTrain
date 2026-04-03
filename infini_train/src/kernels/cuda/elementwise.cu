#include <cstddef>

#include <cub/warp/warp_reduce.cuh>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {
using namespace infini_train::common::cuda;
constexpr int kWarpSize = 32;

// Aligned vector type for vectorized loads/stores (128-bit).
template <typename T, int N> struct __align__(sizeof(T) * N) aligned_vector { T val[N]; };

// Elements per vectorized load/store: 128-bit / sizeof(T).
// float → 4, bf16/half → 8, double → 2.
template <typename T> constexpr int kVecSize = 16 / sizeof(T);

// Maximum number of dimensions supported by the broadcast metadata.
// Real-world tensors in this codebase top out at 4-5 dims, so 8 leaves comfortable headroom
// while keeping the struct under the 4 KB CUDA kernel parameter limit.
constexpr int kMaxBroadcastDims = 8;

// POD metadata for broadcast kernels. Passed by value into __global__ kernels so the data
// lives in CUDA kernel parameter memory (constant cache) instead of being uploaded via a
// per-call cudaMallocAsync + cudaMemcpyAsync into global memory.
struct BroadcastMeta {
    int ndim;
    int64_t a_strides[kMaxBroadcastDims];
    int64_t b_strides[kMaxBroadcastDims];
    int64_t out_strides[kMaxBroadcastDims];
    int64_t a_shape[kMaxBroadcastDims];
    int64_t b_shape[kMaxBroadcastDims];
};

// Build a BroadcastMeta on the host from input/output dim vectors. Right-aligns a_dims/b_dims
// to out_dims's rank (the broadcasting convention) and computes contiguous strides for each.
inline BroadcastMeta MakeBroadcastMeta(const std::vector<int64_t> &a_dims, const std::vector<int64_t> &b_dims,
                                       const std::vector<int64_t> &out_dims) {
    BroadcastMeta m{};
    const int ndim = static_cast<int>(out_dims.size());
    CHECK_LE(ndim, kMaxBroadcastDims) << "Broadcast ndim exceeds kMaxBroadcastDims (" << kMaxBroadcastDims << ")";
    m.ndim = ndim;

    std::vector<int64_t> a_shape(ndim, 1), b_shape(ndim, 1);
    std::copy_backward(a_dims.begin(), a_dims.end(), a_shape.end());
    std::copy_backward(b_dims.begin(), b_dims.end(), b_shape.end());

    auto a_str = ComputeStrides(a_shape);
    auto b_str = ComputeStrides(b_shape);
    auto out_str = ComputeStrides(out_dims);

    for (int i = 0; i < ndim; ++i) {
        m.a_strides[i] = a_str[i];
        m.b_strides[i] = b_str[i];
        m.out_strides[i] = out_str[i];
        m.a_shape[i] = a_shape[i];
        m.b_shape[i] = b_shape[i];
    }
    return m;
}

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

inline bool ShapesEqual(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

template <typename T, typename Func>
__global__ void BinaryForwardKernel(T *output, Func fn, BroadcastMeta meta, const T *a, const T *b,
                                    size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    int64_t a_offset = CalcOffset(idx, meta.ndim, meta.a_strides, meta.a_shape, meta.out_strides);
    int64_t b_offset = CalcOffset(idx, meta.ndim, meta.b_strides, meta.b_shape, meta.out_strides);

    output[idx] = fn(a[a_offset], b[b_offset]);
}

// Fast path: no broadcast, contiguous tensors — skip CalcOffset entirely
template <typename T, typename Func>
__global__ void BinaryForwardKernelNoBroadcast(T *__restrict__ output, Func fn, const T *__restrict__ a,
                                               const T *__restrict__ b, size_t num_elements) {
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < num_elements;
         idx += grid_stride) {
        output[idx] = fn(a[idx], b[idx]);
    }
}

// Fast path backward: no broadcast, contiguous — skip CalcOffset entirely
template <typename T, typename FuncA, typename FuncB>
__global__ void BinaryBackwardKernelNoBroadcastFast(T *__restrict__ outA, T *__restrict__ outB, FuncA fn_a, FuncB fn_b,
                                                    size_t numel, const T *__restrict__ grad_out,
                                                    const T *__restrict__ inA, const T *__restrict__ inB) {
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += grid_stride) {
        const T a = inA ? inA[idx] : T(0);
        const T b = inB ? inB[idx] : T(0);
        outA[idx] = Mul<T>(grad_out[idx], fn_a(a, b));
        outB[idx] = Mul<T>(grad_out[idx], fn_b(a, b));
    }
}

// Vectorized fast path backward: no broadcast, contiguous.
// Each thread processes VecSize elements using 128-bit loads/stores.
template <typename T, int VecSize, typename FuncA, typename FuncB>
__global__ void BinaryBackwardKernelNoBroadcastVectorized(T *__restrict__ outA, T *__restrict__ outB, FuncA fn_a,
                                                          FuncB fn_b, size_t numel, const T *__restrict__ grad_out,
                                                          const T *__restrict__ inA, const T *__restrict__ inB) {
    using VecT = aligned_vector<T, VecSize>;
    const size_t num_vecs = numel / VecSize;
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t vid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; vid < num_vecs; vid += grid_stride) {
        const size_t base = vid * VecSize;

        // 128-bit vectorized loads
        VecT g_vec = *reinterpret_cast<const VecT *>(&grad_out[base]);
        VecT a_vec, b_vec;
        if (inA) {
            a_vec = *reinterpret_cast<const VecT *>(&inA[base]);
        } else {
#pragma unroll
            for (int i = 0; i < VecSize; ++i) { a_vec.val[i] = T(0); }
        }
        if (inB) {
            b_vec = *reinterpret_cast<const VecT *>(&inB[base]);
        } else {
#pragma unroll
            for (int i = 0; i < VecSize; ++i) { b_vec.val[i] = T(0); }
        }

        // Element-wise computation
        VecT outA_vec, outB_vec;
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            outA_vec.val[i] = Mul<T>(g_vec.val[i], fn_a(a_vec.val[i], b_vec.val[i]));
            outB_vec.val[i] = Mul<T>(g_vec.val[i], fn_b(a_vec.val[i], b_vec.val[i]));
        }

        // 128-bit vectorized stores
        *reinterpret_cast<VecT *>(&outA[base]) = outA_vec;
        *reinterpret_cast<VecT *>(&outB[base]) = outB_vec;
    }

    // Handle tail elements (numel % VecSize != 0)
    const size_t tail_start = num_vecs * VecSize;
    for (size_t idx = tail_start + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel;
         idx += grid_stride) {
        const T a = inA ? inA[idx] : T(0);
        const T b = inB ? inB[idx] : T(0);
        outA[idx] = Mul<T>(grad_out[idx], fn_a(a, b));
        outB[idx] = Mul<T>(grad_out[idx], fn_b(a, b));
    }
}

// Helper to choose optimal block size based on tensor size
inline size_t ChooseBlockSize(size_t num_elements) {
    if (num_elements < 1024) {
        return 64;
    }
    if (num_elements < 65536) {
        return 128;
    }
    if (num_elements < 1048576) {
        return 256;
    }
    return 512;
}

// launch the given kernel function with the given output and inputs
template <size_t BLOCK_SIZE, typename T, typename Kernel, typename... Inputs>
void LaunchKernel(Kernel &&kernel, const std::shared_ptr<Tensor> &output, const Inputs &...inputs) {
    auto extract_ptrs
        = [](const auto &...ts) { return std::make_tuple(static_cast<T *>(ts ? ts->DataPtr() : nullptr)...); };
    auto input_ptrs = extract_ptrs(inputs...);

    const size_t num_elements = output->NumElements();
    // Use dynamic block size based on tensor size for better occupancy
    size_t block_size = std::min(ChooseBlockSize(num_elements), static_cast<size_t>(1024));
    dim3 block_dims(block_size);
    dim3 grid_dims(CEIL_DIV(num_elements, block_dims.x));
    const size_t step = grid_dims.x * block_dims.x;

    for (size_t offset = 0; offset < num_elements; offset += step) {
        std::apply([&](auto... ptrs) { kernel(grid_dims, block_dims, offset, ptrs...); }, input_ptrs);
    }
}

// launch a forward elementwise operation given the calculation function, output, and the inputs
// Note: currently only support unary and binary operations
template <size_t BLOCK_SIZE, typename T, typename Func, typename... Inputs>
void LaunchForward(Func func, const std::shared_ptr<Tensor> &output, const Inputs &...inputs) {
    auto device = output->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();
    T *output_ptr = static_cast<T *>(output->DataPtr());

    if constexpr (sizeof...(inputs) == 1) {
        // Unary case
        LaunchKernel<BLOCK_SIZE, T>(
            [&](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
                UnaryForwardKernel<<<grid, block, 0, cuda_stream>>>(output_ptr, func, output->NumElements(), offset,
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

        // Fast path: no broadcast, contiguous — skip cudaMalloc/Memcpy/CalcOffset.
        // The IsContiguous() guards ensure non-contiguous tensors fall back to the broadcast
        // path, keeping the fast path correct when non-contiguous support is added later.
        if (ShapesEqual(a_dims, out_dims) && ShapesEqual(b_dims, out_dims) && input_a->IsContiguous()
            && input_b->IsContiguous()) {
            const size_t num_elements = output->NumElements();
            const T *a_ptr = static_cast<const T *>(input_a->DataPtr());
            const T *b_ptr = static_cast<const T *>(input_b->DataPtr());
            dim3 block_dims(std::min(BLOCK_SIZE, static_cast<size_t>(1024)));
            dim3 grid_dims(std::min(CEIL_DIV(num_elements, block_dims.x), static_cast<size_t>(65535)));
            BinaryForwardKernelNoBroadcast<<<grid_dims, block_dims, 0, cuda_stream>>>(output_ptr, func, a_ptr, b_ptr,
                                                                                      num_elements);
        } else {
            // Broadcast path: pass strides/shapes by value via kernel parameter memory.
            // This avoids the per-call cudaMallocAsync/cudaMemcpyAsync/cudaFreeAsync that previously
            // dominated the host-side jitter floor (especially under LoRA training).
            BroadcastMeta meta = MakeBroadcastMeta(a_dims, b_dims, out_dims);

            LaunchKernel<BLOCK_SIZE, T>(
                [&](dim3 grid, dim3 block, size_t /*offset*/, const T *a_ptr, const T *b_ptr) {
                    BinaryForwardKernel<<<grid, block, 0, cuda_stream>>>(output_ptr, func, meta, a_ptr, b_ptr,
                                                                         output->NumElements());
                },
                output, inputs...);
        }
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
        output[idx] = Mul<T>(grad_output[idx], fn(input ? input[idx] : T(0)));
    }
}

enum class BF16Path { NoBroadcast, TwoPassHist, BlockReduce };

// Lightweight and stable selector for bf16/half execution paths.
inline BF16Path DecideBF16Path(const std::vector<int64_t> &b_shape, const std::vector<int64_t> &out_shape,
                               size_t b_num_elements) {
    if (ShapesEqual(b_shape, out_shape)) {
        return BF16Path::NoBroadcast;
    }
    const bool varies_last = (b_shape.back() > 1);
    if (varies_last) {
        if (b_num_elements <= 4096) {
            return BF16Path::TwoPassHist; // shared histogram two-pass path
        }
    }
    return BF16Path::BlockReduce; // fallback to block reduction kernel otherwise
}

// Each B element is used exactly once, so gradients can be written directly without reduction.
template <typename T, typename FuncA, typename FuncB>
__global__ void BinaryBackwardKernelNoBroadcast(T *__restrict__ outA, T *__restrict__ outB, FuncA fn_a, FuncB fn_b,
                                                BroadcastMeta meta, size_t numel, const T *__restrict__ grad_out,
                                                const T *__restrict__ inA, const T *__restrict__ inB) {
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += grid_stride) {
        const int64_t a_off = CalcOffset(idx, meta.ndim, meta.a_strides, meta.a_shape, meta.out_strides);
        const int64_t b_off = CalcOffset(idx, meta.ndim, meta.b_strides, meta.b_shape, meta.out_strides);

        const T a = inA ? inA[a_off] : T(0);
        const T b = inB ? inB[b_off] : T(0);

        // Gradient for A has a one-to-one mapping, so we write directly.
        outA[a_off] = Mul<T>(grad_out[idx], fn_a(a, b));

        // Gradient for B also maps one-to-one; no atomics or reductions are required.
        outB[b_off] = common::cuda::Cast<T>(Mul<T>(grad_out[idx], fn_b(a, b)));
    }
}

// First pass of histogram two-pass strategy: per-block accumulation in shared memory.
template <typename T, typename FuncA, typename FuncB>
__global__ void BinaryBackwardBhistPass1Kernel(T *__restrict__ outA, float *__restrict__ work, FuncA fn_a, FuncB fn_b,
                                               BroadcastMeta meta, size_t numel, int K, const T *__restrict__ grad_out,
                                               const T *__restrict__ inA, const T *__restrict__ inB) {
    extern __shared__ float s_hist[]; // dynamic shared memory: K bins plus padding for every 32 buckets
    const int pad = K >> 5;           // insert one padding slot for every 32 buckets
    const int hist_len = K + pad;

    // Zero the shared histogram buffer.
    for (int t = threadIdx.x; t < hist_len; t += blockDim.x) { s_hist[t] = 0.0f; }
    __syncthreads();

    const size_t total_threads = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < numel; idx += total_threads) {
        // Linearized offset for B under general broadcasting.
        const int64_t b_off = CalcOffset(idx, meta.ndim, meta.b_strides, meta.b_shape, meta.out_strides);
        const int bin = static_cast<int>(b_off); // assume K fits in a 32-bit int
        const int pbin = bin + (bin >> 5);       // apply padding mapping

        // Compute the offset for A under broadcasting.
        const int64_t a_off = CalcOffset(idx, meta.ndim, meta.a_strides, meta.a_shape, meta.out_strides);

        const T a = inA ? inA[a_off] : T(0);
        const T b = inB ? inB[bin] : T(0); // B is indexed via the flattened bin

        // A is not broadcast, so gradients can be written directly.
        outA[a_off] = Mul<T>(grad_out[idx], fn_a(a, b));

        // Accumulate B's contribution into the shared histogram using float precision.
        const float g = common::cuda::Cast<float>(Mul<T>(grad_out[idx], fn_b(a, b)));
        atomicAdd(&s_hist[pbin], g);
    }
    __syncthreads();

    // Write this block's histogram back to the global workspace: work[block, :].
    float *dst = work + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(K);
    for (int bin = threadIdx.x; bin < K; bin += blockDim.x) {
        const int pbin = bin + (bin >> 5);
        dst[bin] = s_hist[pbin];
    }
}

// Second pass for histogram path: tile the workspace along CTA dimension and atomically add into float buffer.
template <typename T>
__global__ void BinaryBackwardBhistPass2Reduce2D(const float *__restrict__ work, float *__restrict__ outB_accum,
                                                 size_t numBlocks, int K, int tile_height) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) {
        return;
    }

    const size_t begin_row = static_cast<size_t>(blockIdx.y) * static_cast<size_t>(tile_height);
    const size_t end_row = min(begin_row + static_cast<size_t>(tile_height), numBlocks);

    float acc = 0.0f;
    for (size_t row = begin_row; row < end_row; ++row) { acc += work[row * static_cast<size_t>(K) + k]; }

    atomicAdd(outB_accum + k, acc);
}

// Convert the accumulated float buffer back to the target type (bf16/half/float).
template <typename T> __global__ void CastFloatToTBhist(const float *__restrict__ src, T *__restrict__ dst, int K) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        dst[k] = common::cuda::Cast<T>(src[k]);
    }
}

// Legacy single-dimensional reduction fallback for small grids where atomic tiling is unnecessary.
template <typename T>
__global__ void BinaryBackwardBhistPass2Reduce1D(const float *__restrict__ work, T *__restrict__ outB, size_t numBlocks,
                                                 int K) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= static_cast<size_t>(K)) {
        return;
    }

    float acc = 0.0f;
    for (size_t b = 0; b < numBlocks; ++b) { acc += work[b * static_cast<size_t>(K) + k]; }
    outB[k] = common::cuda::Cast<T>(acc);
}

// Helper that materializes the two-pass histogram path for bf16/half B gradients.
template <typename T, typename FuncA, typename FuncB>
void BinaryBackwardBhistLaunch(FuncA fn_a, FuncB fn_b, T *outA, T *outB, const T *grad_out, const BroadcastMeta &meta,
                               size_t numel, int K, const T *inA, const T *inB, cudaStream_t stream) {
    const int kBlockSize = 256;
    int grid = static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
    if (grid < 1) {
        grid = 1;
    }

    // Workspace layout: [grid, K] floats.
    float *work = nullptr;
    CUDA_CHECK(cudaMallocAsync(&work, static_cast<size_t>(grid) * static_cast<size_t>(K) * sizeof(float), stream));

    // Pass 1: per-block histogram accumulation.
    const size_t smem_bytes = static_cast<size_t>(K + (K >> 5)) * sizeof(float);
    BinaryBackwardBhistPass1Kernel<T, FuncA, FuncB>
        <<<grid, kBlockSize, smem_bytes, stream>>>(outA, work, fn_a, fn_b, meta, numel, K, grad_out, inA, inB);
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: choose between 1D and 2D reductions depending on workload shape.
    int dev = 0;
    int sm_count = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

    const int RED_THREADS = 256;
    const int oneD_blocks = (K + RED_THREADS - 1) / RED_THREADS;

    // Use the 2D path when the 1D kernel underutilizes the SMs and there are many partial histograms to merge.
    const bool use2D = (oneD_blocks < sm_count) && (grid > 4 * sm_count);

    if (!use2D) {
        // Fallback: reuse the legacy 1D kernel without atomics.
        const dim3 rgrid(oneD_blocks);
        const dim3 rblock(RED_THREADS);
        BinaryBackwardBhistPass2Reduce1D<T><<<rgrid, rblock, 0, stream>>>(work, outB, static_cast<size_t>(grid), K);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // 2D tiling path: slice the workspace and accumulate using float atomics.
        constexpr int kTileHeight = 128; // rows per CTA; tune between 128 and 256 if needed
        float *outB_accum = nullptr;
        CUDA_CHECK(cudaMallocAsync(&outB_accum, static_cast<size_t>(K) * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(outB_accum, 0, static_cast<size_t>(K) * sizeof(float), stream));

        const dim3 rblock(RED_THREADS, 1, 1);
        const dim3 rgrid2((K + RED_THREADS - 1) / RED_THREADS, (grid + kTileHeight - 1) / kTileHeight, 1);

        BinaryBackwardBhistPass2Reduce2D<T>
            <<<rgrid2, rblock, 0, stream>>>(work, outB_accum, static_cast<size_t>(grid), K, kTileHeight);
        CUDA_CHECK(cudaGetLastError());

        // Convert accumulated floats back to the target dtype.
        const dim3 cgrid((K + RED_THREADS - 1) / RED_THREADS);
        CastFloatToTBhist<T><<<cgrid, RED_THREADS, 0, stream>>>(outB_accum, outB, K);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFreeAsync(outB_accum, stream));
    }

    CUDA_CHECK(cudaFreeAsync(work, stream));
}

// Backward kernel for binary operators
// TODO(lzm): determining and passing b_is_broadcasted from the caller; optimize further
template <typename T, typename FuncA, typename FuncB>
__global__ void BinaryBackwardKernel(T *output_a, T *output_b, FuncA fn_a, FuncB fn_b, BroadcastMeta meta,
                                     size_t num_elements, const T *grad_output, const T *input_a, const T *input_b) {
    extern __shared__ char shared_memory[];
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    using WarpReduce = cub::WarpReduce<float>;
    WarpReduce::TempStorage *temp_storage = reinterpret_cast<WarpReduce::TempStorage *>(shared_memory);

    size_t idx = blockIdx.x * blockDim.x + tid;
    bool in_bounds = (idx < num_elements);

    int64_t a_offset = 0, b_offset = 0;
    T a_val = T(0), b_val = T(0);
    float grad_val = 0.0f;

    if (in_bounds) {
        a_offset = CalcOffset(idx, meta.ndim, meta.a_strides, meta.a_shape, meta.out_strides);
        b_offset = CalcOffset(idx, meta.ndim, meta.b_strides, meta.b_shape, meta.out_strides);
        a_val = input_a ? input_a[a_offset] : T(0);
        b_val = input_b ? input_b[b_offset] : T(0);
        output_a[a_offset] = Mul<T>(grad_output[idx], fn_a(a_val, b_val));
        grad_val = common::cuda::Cast<float>(Mul<T>(grad_output[idx], fn_b(a_val, b_val)));
    }

    unsigned active_mask = __ballot_sync(0xFFFFFFFF, in_bounds);
    if (!active_mask) {
        return;
    }

    int leader = __ffs(active_mask) - 1;
    int64_t common_offset = __shfl_sync(active_mask, b_offset, leader);

    // Check if all active threads share common b_offset
    bool warp_uniform = true;
    for (int i = 0; i < 32; ++i) {
        if (!(active_mask & (1 << i))) {
            continue;
        }
        int64_t offset_i = __shfl_sync(active_mask, b_offset, i);
        if (offset_i != common_offset) {
            warp_uniform = false;
            break;
        }
    }

    if (warp_uniform) {
        float reduced = WarpReduce(temp_storage[warp_id]).Sum(grad_val);
        if (lane_id == leader) {
            // FIXME(lzm): atomicAdd is much slower for bf16 and half compared to float, needs further optimization
            atomicAdd(&output_b[common_offset], common::cuda::Cast<T>(reduced));
        }
    } else if (in_bounds) {
        // FIXME(lzm): atomicAdd is much slower for bf16 and half compared to float, needs further optimization
        atomicAdd(&output_b[b_offset], common::cuda::Cast<T>(grad_val));
    }
}

// NOTE(dcj): Specialized BinaryBackwardKernel for low-precision types (__half / bfloat16)
template <typename T, typename FuncA, typename FuncB>
__global__ void BinaryBackwardKernel(T *output_a, T *output_b, FuncA fn_a, FuncB fn_b, BroadcastMeta meta,
                                     size_t num_elements, size_t b_num_elements, const T *grad_output, const T *input_a,
                                     const T *input_b, bool fast_atomics) {

    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;
    const int global_idx = blockIdx.x * blockDim.x + tid;
    bool in_bounds = (global_idx < num_elements);

    // Dynamic shared memory layout: split offsets and gradients into parallel arrays.
    extern __shared__ char shared_memory[];
    int64_t *s_offset = reinterpret_cast<int64_t *>(shared_memory);
    float *s_grad = reinterpret_cast<float *>(s_offset + block_threads + block_threads / kWarpSize);

    // Padding: insert one slot per 32 threads to avoid bank conflicts.
    const int padded_tid = tid + (tid >> 5);

    // Each thread calculates its own a_offset and b_offset
    int64_t a_offset = 0, b_offset = 0;
    float grad_val = 0.0f;
    T a_val = T(0), b_val = T(0);

    if (in_bounds) {
        a_offset = CalcOffset(global_idx, meta.ndim, meta.a_strides, meta.a_shape, meta.out_strides);
        b_offset = CalcOffset(global_idx, meta.ndim, meta.b_strides, meta.b_shape, meta.out_strides);

        a_val = input_a ? input_a[a_offset] : T(0);
        b_val = input_b ? input_b[b_offset] : T(0);

        // Compute gradient contribution for output_a
        output_a[a_offset] = Mul<T>(grad_output[global_idx], fn_a(a_val, b_val));
        // Store gradient contribution for output_b in float for accumulation
        grad_val = common::cuda::Cast<float>(Mul<T>(grad_output[global_idx], fn_b(a_val, b_val)));
    }

    // Store partial results in shared memory.
    s_offset[padded_tid] = in_bounds ? b_offset : -1;
    s_grad[padded_tid] = grad_val;

    __syncthreads();

    // Perform block-wide reduction with padded indices.
    for (int stride = 1; stride < block_threads; stride *= 2) {
        __syncthreads();
        if ((tid % (2 * stride)) == 0 && (tid + stride) < block_threads) {
            const int p1 = tid + (tid >> 5);
            const int p2 = (tid + stride) + ((tid + stride) >> 5);

            if (s_offset[p1] == s_offset[p2] && s_offset[p1] != -1) {
                s_grad[p1] += s_grad[p2];
                s_offset[p2] = -1;
            }
        }
    }
    __syncthreads();

    // Write final result back to global memory
    if (in_bounds) {
        const int shared_idx = tid + (tid >> 5);
        if (s_offset[shared_idx] != -1) {
            fastAtomicAdd<T, size_t>(output_b, s_offset[shared_idx], b_num_elements,
                                     common::cuda::Cast<T>(s_grad[shared_idx]), fast_atomics);
        }
    }
}

// launch unary operator's backward kernel
template <size_t BLOCK_SIZE, typename T, typename Func, typename... Inputs>
void LaunchBackward(Func func, const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &grad_output,
                    const Inputs &...inputs) {
    auto device = output->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    T *output_ptr = static_cast<T *>(output->DataPtr());
    const T *grad_ptr = static_cast<const T *>(grad_output->DataPtr());

    LaunchKernel<BLOCK_SIZE, T>(
        [=](dim3 grid, dim3 block, size_t offset, auto... ptrs) {
            UnaryBackwardKernel<<<grid, block, 0, cuda_stream>>>(output_ptr, func, output->NumElements(), offset,
                                                                 grad_ptr, ptrs...);
        },
        output, inputs...);
}

// launch binary operator's backward kernel
template <size_t BLOCK_SIZE, typename T, typename FuncA, typename FuncB, typename... Inputs>
void LaunchBackward(FuncA fun_a, FuncB fun_b, const std::shared_ptr<Tensor> &output_a,
                    const std::shared_ptr<Tensor> &output_b, const std::vector<int64_t> &a_dims,
                    const std::vector<int64_t> &b_dims, const std::shared_ptr<Tensor> &grad_output,
                    const Inputs &...inputs) {
    auto device = output_a->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();

    T *output_a_ptr = static_cast<T *>(output_a->DataPtr());
    T *output_b_ptr = static_cast<T *>(output_b->DataPtr());
    const T *grad_output_ptr = static_cast<const T *>(grad_output->DataPtr());

    const auto &out_dims = grad_output->Dims();
    const size_t num_elements = grad_output->NumElements();

    // Fast path: no broadcast, contiguous — skip cudaMalloc/Memcpy/CalcOffset.
    // The IsContiguous() guard ensures non-contiguous grad_output falls back to the broadcast
    // path, keeping the fast path correct when non-contiguous support is added later.
    if (ShapesEqual(a_dims, b_dims) && ShapesEqual(a_dims, out_dims) && grad_output->IsContiguous()) {
        auto extract_ptrs = [](const auto &...ts) {
            return std::make_tuple(static_cast<const T *>(ts ? ts->DataPtr() : nullptr)...);
        };
        auto [input_a_ptr, input_b_ptr] = extract_ptrs(inputs...);

        constexpr int VecSize = kVecSize<T>;
        // Use vectorized kernel if all pointers are 16-byte aligned and numel is large enough
        const bool can_vectorize
            = (num_elements >= static_cast<size_t>(VecSize))
           && (reinterpret_cast<uintptr_t>(output_a_ptr) % (sizeof(T) * VecSize) == 0)
           && (reinterpret_cast<uintptr_t>(output_b_ptr) % (sizeof(T) * VecSize) == 0)
           && (reinterpret_cast<uintptr_t>(grad_output_ptr) % (sizeof(T) * VecSize) == 0)
           && (!input_a_ptr || reinterpret_cast<uintptr_t>(input_a_ptr) % (sizeof(T) * VecSize) == 0)
           && (!input_b_ptr || reinterpret_cast<uintptr_t>(input_b_ptr) % (sizeof(T) * VecSize) == 0);

        if (can_vectorize) {
            const size_t num_vecs = num_elements / VecSize;
            dim3 block_dims(std::min(static_cast<size_t>(256), std::min(num_vecs, static_cast<size_t>(1024))));
            dim3 grid_dims(std::min(CEIL_DIV(num_vecs, block_dims.x), static_cast<size_t>(65535)));
            BinaryBackwardKernelNoBroadcastVectorized<T, VecSize><<<grid_dims, block_dims, 0, stream>>>(
                output_a_ptr, output_b_ptr, fun_a, fun_b, num_elements, grad_output_ptr, input_a_ptr, input_b_ptr);
        } else {
            dim3 block_dims(std::min(BLOCK_SIZE, static_cast<size_t>(1024)));
            dim3 grid_dims(std::min(CEIL_DIV(num_elements, block_dims.x), static_cast<size_t>(65535)));
            BinaryBackwardKernelNoBroadcastFast<<<grid_dims, block_dims, 0, stream>>>(
                output_a_ptr, output_b_ptr, fun_a, fun_b, num_elements, grad_output_ptr, input_a_ptr, input_b_ptr);
        }
        return;
    }

    // Broadcast path: pass strides/shapes by value via kernel parameter memory.
    // This avoids the per-call cudaMallocAsync/cudaMemcpyAsync/cudaFreeAsync that previously
    // dominated the host-side jitter floor (especially under LoRA training).
    BroadcastMeta meta = MakeBroadcastMeta(a_dims, b_dims, out_dims);

    if constexpr (std::is_same_v<T, float>) {
        LaunchKernel<BLOCK_SIZE, T>(
            [=](dim3 grid, dim3 block, size_t /*offset*/, auto... ptrs) {
                const int num_warps = BLOCK_SIZE / kWarpSize;
                const size_t smem_size = num_warps * sizeof(cub::WarpReduce<float>::TempStorage);
                BinaryBackwardKernel<<<grid, block, smem_size, stream>>>(output_a_ptr, output_b_ptr, fun_a, fun_b, meta,
                                                                         num_elements, grad_output_ptr, ptrs...);
            },
            output_a, inputs...);
    } else if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        // Dynamically choose the most efficient bf16/half strategy based on broadcast pattern.
        // Reconstruct right-aligned b_shape (stack-only, no device allocations) for
        // DecideBF16Path which still operates on std::vector.
        const int ndim = meta.ndim;
        std::vector<int64_t> b_shape(meta.b_shape, meta.b_shape + ndim);
        const std::vector<int64_t> &out_shape = out_dims;

        size_t b_num_elements = 1;
        for (auto v : b_shape) { b_num_elements *= static_cast<size_t>(v); }
        const int K_linear = static_cast<int>(b_num_elements);

        // Select the execution path.
        const BF16Path path = DecideBF16Path(b_shape, out_shape, b_num_elements);

        if (path == BF16Path::NoBroadcast) {
            // No broadcast: write gradients directly without shared memory or atomics.
            LaunchKernel<BLOCK_SIZE, T>(
                [=](dim3 grid, dim3 block, size_t /*offset*/, auto... ptrs) {
                    BinaryBackwardKernelNoBroadcast<T, FuncA, FuncB><<<grid, block, 0, stream>>>(
                        output_a_ptr, output_b_ptr, fun_a, fun_b, meta, num_elements, grad_output_ptr, ptrs...);
                },
                output_a, inputs...);
            return;
        }

        if (path == BF16Path::TwoPassHist) {
            // Small K with variation in the innermost dimension: use two-pass histogram strategy.
            LaunchKernel<BLOCK_SIZE, T>(
                [=](dim3 /*grid*/, dim3 /*block*/, size_t /*offset*/, const T *input_a_ptr, const T *input_b_ptr) {
                    BinaryBackwardBhistLaunch<T, FuncA, FuncB>(fun_a, fun_b, output_a_ptr, output_b_ptr,
                                                               grad_output_ptr, meta, num_elements, K_linear,
                                                               input_a_ptr, input_b_ptr, stream);
                },
                output_a, inputs...);

            return;
        }

        // Otherwise fall back to the block-reduction kernel with SoA layout and fast atomics.
        LaunchKernel<BLOCK_SIZE, T>(
            [=](dim3 grid, dim3 block, size_t /*offset*/, auto... ptrs) {
                const int padded_block = BLOCK_SIZE + BLOCK_SIZE / kWarpSize;
                const size_t smem_size = static_cast<size_t>(padded_block) * (sizeof(int64_t) + sizeof(float));
                BinaryBackwardKernel<<<grid, block, smem_size, stream>>>(
                    output_a_ptr, output_b_ptr, fun_a, fun_b, meta, num_elements, output_b->NumElements(),
                    grad_output_ptr, ptrs..., /*fast_atomics=*/true);
            },
            output_a, inputs...);
    }
}

template <typename Func> std::shared_ptr<Tensor> UnaryForward(const std::shared_ptr<Tensor> &input, Func unary_fn) {
    auto dtype = input->Dtype();
    auto output = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchForward<256, float>(unary_fn, output, input);), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchForward<256, nv_bfloat16>(unary_fn, output, input);), DataType::kBFLOAT16)
        DISPATCH_CASE(WRAP(LaunchForward<256, int64_t>(unary_fn, output, input);), DataType::kINT64)
    default:
        LOG_LOC(FATAL, "CUDA unary forward: 'Unsupported data type'");
    }

    return output;
}

template <typename Func>
std::shared_ptr<Tensor> UnaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
                                      Func unary_fn) {
    auto dtype = grad_output->Dtype();
    auto a_dtype = a ? a->Dtype() : dtype;
    DataType promoted_type = PromoteDataTypes(dtype, a_dtype);

    auto grad_output_promoted
        = dtype == promoted_type ? grad_output : std::make_shared<Tensor>(grad_output->To(promoted_type));
    auto a_promoted = a_dtype == promoted_type ? a : std::make_shared<Tensor>(a->To(promoted_type));
    auto output = std::make_shared<Tensor>(grad_output->Dims(), promoted_type, grad_output->GetDevice());

    switch (promoted_type) {
        DISPATCH_CASE(WRAP({ LaunchBackward<256, float>(unary_fn, output, grad_output_promoted, a_promoted); }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({ LaunchBackward<256, nv_bfloat16>(unary_fn, output, grad_output_promoted, a_promoted); }),
                      DataType::kBFLOAT16)
        DISPATCH_CASE(WRAP({ LaunchBackward<256, int64_t>(unary_fn, output, grad_output_promoted, a_promoted); }),
                      DataType::kINT64)
    default:
        LOG_LOC(FATAL, "CUDA unary backward: 'Unsupported data type'");
    }

    return output;
}

template <typename Func>
std::shared_ptr<Tensor> BinaryForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b,
                                      Func binary_fn) {
    auto a_dtype = a->Dtype();
    auto b_dtype = b->Dtype();

    DataType promoted_type = PromoteDataTypes(a_dtype, b_dtype);

    auto a_promoted = a_dtype == promoted_type ? a : std::make_shared<Tensor>(a->To(promoted_type));
    auto b_promoted = b_dtype == promoted_type ? b : std::make_shared<Tensor>(b->To(promoted_type));
    // Currently a and b should have the same data type and only one-way broadcasting from b to a is assumed by
    // default
    CHECK(a->NumElements() >= b->NumElements() && a->NumElements() % b->NumElements() == 0);

    auto output = std::make_shared<Tensor>(a->Dims(), promoted_type, a->GetDevice());

    switch (promoted_type) {
        DISPATCH_CASE(WRAP(LaunchForward<256, float>(binary_fn, output, a_promoted, b_promoted);), DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchForward<256, nv_bfloat16>(binary_fn, output, a_promoted, b_promoted);),
                      DataType::kBFLOAT16)
        DISPATCH_CASE(WRAP(LaunchForward<256, int64_t>(binary_fn, output, a_promoted, b_promoted);), DataType::kINT64)
    default:
        LOG_LOC(FATAL, "CUDA binary forward: 'Unsupported data type'");
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

    std::shared_ptr<Tensor> a_promoted = a;
    std::shared_ptr<Tensor> b_promoted = b;
    std::shared_ptr<Tensor> grad_output_promoted = grad_output;

    auto dtype = grad_output_promoted->Dtype();
    auto device = grad_output->GetDevice();

    auto a_dtype = a_promoted ? a_promoted->Dtype() : dtype;
    auto b_dtype = b_promoted ? b_promoted->Dtype() : dtype;
    // Compute dtype determined by saved tensors (forward compute dtype), not grad_output
    DataType promoted_type = PromoteDataTypes(a_dtype, b_dtype);

    CHECK(a_num_elements >= b_num_elements && a_num_elements % b_num_elements == 0);

    auto promote_if_needed = [&](std::shared_ptr<Tensor> &t, size_t expected_numel, DataType promoted_type) {
        if (t) {
            CHECK(expected_numel == t->NumElements());
            if (t->Dtype() != promoted_type) {
                t = std::make_shared<Tensor>(t->To(promoted_type));
            }
        }
    };
    promote_if_needed(a_promoted, a_num_elements, promoted_type);
    promote_if_needed(b_promoted, b_num_elements, promoted_type);
    if (dtype != promoted_type) {
        grad_output_promoted = std::make_shared<Tensor>(grad_output_promoted->To(promoted_type));
    }

    auto grad_a = std::make_shared<Tensor>(a_dims, promoted_type, device);
    auto grad_b = std::make_shared<Tensor>(b_dims, promoted_type, device);

    // Only Fill(0) when broadcast is needed (atomicAdd requires zero-init).
    // The no-broadcast fast path writes every element directly.
    const bool needs_broadcast = !ShapesEqual(a_dims, b_dims) || !ShapesEqual(a_dims, grad_output->Dims());

    switch (promoted_type) {
        DISPATCH_CASE(WRAP({
                          if (needs_broadcast) {
                              grad_a->Fill<float>(0.0f);
                              grad_b->Fill<float>(0.0f);
                          }
                          LaunchBackward<256, float>(fn_a, fn_b, grad_a, grad_b, a_dims, b_dims, grad_output_promoted,
                                                     a_promoted, b_promoted);
                      }),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP({
                          if (needs_broadcast) {
                              grad_a->Fill<nv_bfloat16>(0);
                              grad_b->Fill<nv_bfloat16>(0);
                          }
                          LaunchBackward<256, nv_bfloat16>(fn_a, fn_b, grad_a, grad_b, a_dims, b_dims,
                                                           grad_output_promoted, a_promoted, b_promoted);
                      }),
                      DataType::kBFLOAT16)
        // FIXME(zbl): AtomicAdd does not support int64_t
        // DISPATCH_CASE(WRAP({
        //                   grad_a->Fill(0.0);
        //                   grad_b->Fill(0.0);
        //                   LaunchBackward<256, int64_t>(fn_a, fn_b, grad_a, grad_b, a_dims, b_dims, grad_output, a,
        //                   b);
        //               }),
        //               DataType::kINT64)
    default:
        LOG_LOC(FATAL, "CUDA binary backward: 'Unsupported data type'");
    }

    return {grad_a, grad_b};
}
} // namespace

std::shared_ptr<Tensor> NegForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Neg(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> NegBackward(const std::shared_ptr<Tensor> &grad_output) {
    DISPATCH(grad_output->Dtype(),
             return UnaryBackward(grad_output, nullptr, [] __device__(auto x) { return decltype(x){-1}; });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> ReciprocalForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Reciprocal(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> ReciprocalBackward(const std::shared_ptr<Tensor> &grad_output,
                                           const std::shared_ptr<Tensor> &input) {
    DISPATCH(
        grad_output->Dtype(),
        return UnaryBackward(grad_output, input, [] __device__(auto x) { return Div(decltype(x){-1}, Mul(x, x)); });
        , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> SinForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Sin(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> SinBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    DISPATCH(grad_output->Dtype(), return UnaryBackward(grad_output, input, [] __device__(auto x) { return Cos(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> CosForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Cos(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> CosBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    DISPATCH(grad_output->Dtype(),
             return UnaryBackward(grad_output, input, [] __device__(auto x) { return Neg(Sin(x)); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> TanhForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Tanh(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> TanhBackward(const std::shared_ptr<Tensor> &grad_output,
                                     const std::shared_ptr<Tensor> &output) {
    DISPATCH(grad_output->Dtype(),
             return UnaryBackward(grad_output, output, [] __device__(auto x) { return decltype(x){1} - Mul(x, x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> PowForward(const std::shared_ptr<Tensor> &input, float scalar, bool scalar_is_base) {
    DISPATCH(input->Dtype(), WRAP({
                 if (scalar_is_base) {
                     return UnaryForward(
                         input, [scalar] __device__(auto x) { return Pow(static_cast<decltype(x)>(scalar), x); });
                 } else {
                     return UnaryForward(
                         input, [scalar] __device__(auto x) { return Pow(x, static_cast<decltype(x)>(scalar)); });
                 }
             }),
             INFINI_ALL_FLOATING_TYPES);
}

std::shared_ptr<Tensor> PowBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    float scalar, bool scalar_is_base) {
    DISPATCH(grad_output->Dtype(),
             return UnaryBackward(grad_output, input,
                                  [scalar, scalar_is_base] __device__(auto x) {
                                      auto casted_scalar = common::cuda::Cast<decltype(x)>(scalar);
                                      if (scalar_is_base) {
                                          return Mul(Log(casted_scalar), Pow(casted_scalar, x));
                                      } else {
                                          return Mul(casted_scalar, Pow(x, casted_scalar - decltype(x){1}));
                                      }
                                  });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> RsqrtForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Rsqrt(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> RsqrtBackward(const std::shared_ptr<Tensor> &grad_output,
                                      const std::shared_ptr<Tensor> &input) {
    DISPATCH(grad_output->Dtype(),
             return UnaryBackward(
                 grad_output, input,
                 [] __device__(auto x) { return Mul(static_cast<decltype(x)>(-0.5), Mul(Reciprocal(x), Rsqrt(x))); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> ExpForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Exp(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> ExpBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &output) {
    DISPATCH(grad_output->Dtype(), return UnaryBackward(grad_output, output, [] __device__(auto y) { return y; });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> LogForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Log(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> LogBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    DISPATCH(grad_output->Dtype(),
             return UnaryBackward(grad_output, input, [] __device__(auto x) { return Reciprocal(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> EqualsForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(),
             return BinaryForward(a, b,
                                  [] __device__(auto x, auto y) { return (x == y) ? decltype(x){1} : decltype(x){0}; });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> EqualsScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    DISPATCH(a->Dtype(), return UnaryForward(a,
                                             [scalar] __device__(auto x) {
                                                 return x == static_cast<decltype(x)>(scalar) ? decltype(x){1}
                                                                                              : decltype(x){0};
                                             });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> LtForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(), return BinaryForward(
                             a, b, [] __device__(auto x, auto y) { return x < y ? decltype(x){1} : decltype(x){0}; });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> LtScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    DISPATCH(a->Dtype(), return UnaryForward(a,
                                             [scalar] __device__(auto x) {
                                                 return (x < static_cast<decltype(x)>(scalar)) ? decltype(x){1}
                                                                                               : decltype(x){0};
                                             });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> LeForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(),
             return BinaryForward(a, b,
                                  [] __device__(auto x, auto y) { return (x <= y) ? decltype(x){1} : decltype(x){0}; });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> LeScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    DISPATCH(a->Dtype(), return UnaryForward(a,
                                             [scalar] __device__(auto x) {
                                                 return (x <= static_cast<decltype(x)>(scalar)) ? decltype(x){1}
                                                                                                : decltype(x){0};
                                             });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> GtForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(), return BinaryForward(
                             a, b, [] __device__(auto x, auto y) { return x > y ? decltype(x){1} : decltype(x){0}; });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> GtScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    DISPATCH(a->Dtype(), return UnaryForward(a,
                                             [scalar] __device__(auto x) {
                                                 return (x > static_cast<decltype(x)>(scalar)) ? decltype(x){1}
                                                                                               : decltype(x){0};
                                             });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> GeForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(),
             return BinaryForward(a, b,
                                  [] __device__(auto x, auto y) { return (x >= y) ? decltype(x){1} : decltype(x){0}; });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> GeScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    DISPATCH(a->Dtype(), return UnaryForward(a,
                                             [scalar] __device__(auto x) {
                                                 return (x >= static_cast<decltype(x)>(scalar)) ? decltype(x){1}
                                                                                                : decltype(x){0};
                                             });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> OrForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(), return BinaryForward(a, b,
                                              [] __device__(auto x, auto y) {
                                                  return (x != decltype(x){0} || y != decltype(y){0}) ? decltype(x){1}
                                                                                                      : decltype(x){0};
                                              });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> AndForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(), return BinaryForward(a, b,
                                              [] __device__(auto x, auto y) {
                                                  return (x != decltype(x){0} && y != decltype(y){0}) ? decltype(x){1}
                                                                                                      : decltype(x){0};
                                              });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(), return BinaryForward(a, b, [] __device__(auto x, auto y) { return Add(x, y); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims) {
    auto fn = [] __device__(auto x, auto y) { return decltype(x){1}; };
    return BinaryBackward(grad_output, nullptr, nullptr, a_dims, b_dims, fn, fn);
}

std::shared_ptr<Tensor> AddScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    DISPATCH(a->Dtype(),
             return UnaryForward(a, [scalar] __device__(auto x) { return Add(x, static_cast<decltype(x)>(scalar)); });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> AddScalarBackward(const std::shared_ptr<Tensor> &grad_output) {
    DISPATCH(grad_output->Dtype(),
             return UnaryBackward(grad_output, nullptr,
                                  [] __device__(auto x) { return common::cuda::Cast<decltype(x)>(1); });
             , INFINI_ALL_TYPES)
}

std::shared_ptr<Tensor> SubForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(), return BinaryForward(a, b, [] __device__(auto x, auto y) { return Sub(x, y); });
             , INFINI_ALL_TYPES)
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> SubBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims) {
    auto fn_a = [] __device__(auto x, auto y) { return decltype(x){1}; };
    auto fn_b = [] __device__(auto x, auto y) { return decltype(x){-1}; };
    return BinaryBackward(grad_output, nullptr, nullptr, a_dims, b_dims, fn_a, fn_b);
}

std::shared_ptr<Tensor> MulForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(), return BinaryForward(a, b, [] __device__(auto x, auto y) { return Mul(x, y); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MulBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b) {
    DISPATCH_WITH_DEFAULT(grad_output->Dtype(),
                          return BinaryBackward(
                              grad_output, a, b, a->Dims(), b->Dims(), [] __device__(auto, auto y) { return y; },
                              [] __device__(auto x, auto) { return x; });
                          , WRAP({
                              LOG_LOC(FATAL, "CUDA MulBackward: 'Unsupported data type'");
                              return {nullptr, nullptr};
                          }),
                          INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> MulScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    DISPATCH(a->Dtype(),
             return UnaryForward(a, [scalar] __device__(auto x) { return Mul(x, static_cast<decltype(x)>(scalar)); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> MulScalarBackward(const std::shared_ptr<Tensor> &grad_output, float scalar) {
    DISPATCH(grad_output->Dtype(),
             return UnaryBackward(grad_output, nullptr,
                                  [scalar] __device__(auto x) { return static_cast<decltype(x)>(scalar); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> DivForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    DISPATCH(a->Dtype(), return BinaryForward(a, b, [] __device__(auto x, auto y) { return Div(x, y); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DivBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b) {
    DISPATCH_WITH_DEFAULT(grad_output->Dtype(), return BinaryBackward(
                                                    grad_output, a, b, a->Dims(), b->Dims(),
                                                    [] __device__(auto, auto y) { return Reciprocal(y); },
                                                    [] __device__(auto x, auto y) { return Div(Neg(x), Mul(y, y)); });
                          , WRAP({
                              LOG_LOC(FATAL, "CUDA DivBackward: 'Unsupported data type'");
                              return {nullptr, nullptr};
                          }),
                          INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> SigmoidForward(const std::shared_ptr<Tensor> &input) {
    DISPATCH(input->Dtype(), return UnaryForward(input, [] __device__(auto x) { return Sigmoid(x); });
             , INFINI_ALL_FLOATING_TYPES)
}

std::shared_ptr<Tensor> SigmoidBackward(const std::shared_ptr<Tensor> &output,
                                        const std::shared_ptr<Tensor> &grad_output) {
    DISPATCH(
        grad_output->Dtype(),
        return UnaryBackward(grad_output, output, [] __device__(auto x) { return Mul(x, Sub(decltype(x){1}, x)); });
        , INFINI_ALL_FLOATING_TYPES)
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ELEMENTWISE_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

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
REGISTER_CUDA_ELEMENTWISE_KERNEL(ExpForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(ExpBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(LogForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(LogBackward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(EqualsForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(EqualsScalarForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(LtForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(LtScalarForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(LeForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(LeScalarForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(GtForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(GtScalarForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(GeForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(GeScalarForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(OrForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(AndForward)
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
REGISTER_CUDA_ELEMENTWISE_KERNEL(SigmoidForward)
REGISTER_CUDA_ELEMENTWISE_KERNEL(SigmoidBackward)

#undef REGISTER_CUDA_ELEMENTWISE_KERNEL

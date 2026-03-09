// FlashAttention v2 CUDA kernel implementation for InfiniTrain.
//
// Implements IO-aware fused attention with online softmax, supporting:
//   - Forward and backward passes (full recomputation-based backward)
//   - Causal masking
//   - Configurable scaling factor
//   - GQA (Grouped Query Attention): Q may have more heads than K/V
//   - Dropout with deterministic Philox RNG
//
// Reference: FlashAttention-2 (Dao, 2023), arXiv:2307.08691
//
// Data layout: Q, K, V in [B, H, N, d] (batch, head, sequence, head_dim)
// All intermediate computations are in float32 for numerical stability.

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

namespace {

// Get the CUDA stream for the given device.
cudaStream_t GetCudaStream(const Device &device) {
    return dynamic_cast<infini_train::core::cuda::CudaStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

// Philox-based deterministic RNG for dropout reproducibility.
// Given a 64-bit counter and a seed, produces a pseudo-random float in [0, 1).
__device__ __forceinline__ float philox_uniform(unsigned long long counter, unsigned long long seed) {
    unsigned long long x = counter ^ seed;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (x & 0xFFFFFFFF) * 2.3283064365386963e-10f;
}

// ============================================================================
// FlashAttention Forward Kernel
// ============================================================================
//
// Each thread block processes one (batch, q_head, q_tile) combination.
// It iterates over all K/V tiles, computing attention using online softmax.
//
// Shared memory layout (all float):
//   sQ  [Br * d]      - query tile
//   sKV [Bc * d]      - key or value tile (reused: loads K first, then V)
//   sS  [Br * Bc]     - attention scores / probabilities
//   row_m [Br]         - running row max
//   row_l [Br]         - running row sum
//   sO  [Br * d]       - output accumulator
template <int Br, int Bc, typename T>
__global__ void FlashAttnFwdKernel(const T *__restrict__ Q,  // [B, H_q, N, d]
                                   const T *__restrict__ K,  // [B, H_kv, N, d]
                                   const T *__restrict__ V,  // [B, H_kv, N, d]
                                   T *__restrict__ O,        // [B, H_q, N, d]
                                   float *__restrict__ L,    // [B, H_q, N]
                                   int N, int d, int H_q, int H_kv, float scale, bool is_causal, float dropout_p,
                                   unsigned long long rng_seed) {
    const int q_tile_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / H_q;
    const int head_idx = bh_idx % H_q;
    const int kv_head_idx = H_kv == H_q ? head_idx : head_idx / (H_q / H_kv);
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int q_start = q_tile_idx * Br;
    if (q_start >= N) {
        return;
    }
    const int q_len = min(Br, N - q_start);

    // Global memory pointers
    const int64_t q_head_offset = ((int64_t)batch_idx * H_q + head_idx) * N * d;
    const int64_t kv_head_offset = ((int64_t)batch_idx * H_kv + kv_head_idx) * N * d;
    const T *Q_ptr = Q + q_head_offset + (int64_t)q_start * d;
    T *O_ptr = O + q_head_offset + (int64_t)q_start * d;
    float *L_ptr = L + ((int64_t)batch_idx * H_q + head_idx) * N + q_start;
    const T *K_base = K + kv_head_offset;
    const T *V_base = V + kv_head_offset;

    // Shared memory
    extern __shared__ float smem[];
    float *sQ = smem;                       // [Br * d]
    float *sKV = sQ + Br * d;              // [Bc * d] (holds K then V)
    float *sS = sKV + Bc * d;              // [Br * Bc]
    float *row_m = sS + Br * Bc;           // [Br]
    float *row_l = row_m + Br;             // [Br]
    float *sO = row_l + Br;               // [Br * d]

    // Load Q tile (convert to float)
    for (int idx = tid; idx < q_len * d; idx += num_threads) {
        int r = idx / d;
        int c = idx % d;
        sQ[r * d + c] = common::cuda::Cast<float>(Q_ptr[r * d + c]);
    }
    // Initialize accumulators
    for (int idx = tid; idx < Br; idx += num_threads) {
        row_m[idx] = -INFINITY;
        row_l[idx] = 0.0f;
    }
    for (int idx = tid; idx < Br * d; idx += num_threads) {
        sO[idx] = 0.0f;
    }
    __syncthreads();

    // Iterate over KV tiles
    const int num_kv_tiles = (N + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * Bc;
        const int kv_len = min(Bc, N - kv_start);

        // Early exit for causal: skip if all KV positions are after all Q positions
        if (is_causal && kv_start > q_start + q_len - 1) {
            break;
        }

        // --- Phase 1: Load K, compute S = Q @ K^T * scale ---
        for (int idx = tid; idx < kv_len * d; idx += num_threads) {
            int r = idx / d;
            int c = idx % d;
            sKV[r * d + c] = common::cuda::Cast<float>(K_base[(kv_start + r) * d + c]);
        }
        __syncthreads();

        // Compute S[qi][ki] = sum_c Q[qi][c] * K[ki][c] * scale
        // Use consistent stride Bc for sS indexing
        for (int idx = tid; idx < q_len * Bc; idx += num_threads) {
            int qi = idx / Bc;
            int ki = idx % Bc;
            if (ki < kv_len) {
                float dot = 0.0f;
                for (int c = 0; c < d; ++c) {
                    dot += sQ[qi * d + c] * sKV[ki * d + c];
                }
                dot *= scale;
                // Apply causal mask
                if (is_causal && (kv_start + ki) > (q_start + qi)) {
                    dot = -INFINITY;
                }
                sS[qi * Bc + ki] = dot;
            } else {
                sS[qi * Bc + ki] = -INFINITY;
            }
        }
        __syncthreads();

        // --- Phase 2: Online softmax per row ---
        // Each thread handles one row: compute max, exp(S-max), row_sum, rescale
        for (int qi = tid; qi < q_len; qi += num_threads) {
            float m_old = row_m[qi];
            float l_old = row_l[qi];

            // Find row max
            float m_new = m_old;
            for (int ki = 0; ki < kv_len; ++ki) {
                m_new = fmaxf(m_new, sS[qi * Bc + ki]);
            }

            // Compute P = exp(S - m_new) and row sum
            float l_sum = 0.0f;
            for (int ki = 0; ki < kv_len; ++ki) {
                float s_val = sS[qi * Bc + ki];
                float p = (s_val > -INFINITY) ? expf(s_val - m_new) : 0.0f;
                // Apply dropout
                if (dropout_p > 0.0f && p > 0.0f) {
                    unsigned long long counter = (unsigned long long)(batch_idx * H_q + head_idx) * N * N
                                               + (unsigned long long)(q_start + qi) * N + (kv_start + ki);
                    float r = philox_uniform(counter, rng_seed);
                    p = (r < dropout_p) ? 0.0f : p / (1.0f - dropout_p);
                }
                sS[qi * Bc + ki] = p;
                l_sum += p;
            }
            // Zero out padding positions in P (already 0 from exp(-inf) but be explicit)
            for (int ki = kv_len; ki < Bc; ++ki) {
                sS[qi * Bc + ki] = 0.0f;
            }

            // Rescale old output accumulator
            float rescale = (m_old > -INFINITY) ? expf(m_old - m_new) : 0.0f;
            for (int c = 0; c < d; ++c) {
                sO[qi * d + c] *= rescale;
            }

            row_m[qi] = m_new;
            row_l[qi] = rescale * l_old + l_sum;
        }
        __syncthreads();

        // --- Phase 3: Load V, accumulate P @ V ---
        for (int idx = tid; idx < kv_len * d; idx += num_threads) {
            int r = idx / d;
            int c = idx % d;
            sKV[r * d + c] = common::cuda::Cast<float>(V_base[(kv_start + r) * d + c]);
        }
        __syncthreads();

        // O[qi][c] += sum_ki P[qi][ki] * V[ki][c]
        for (int idx = tid; idx < q_len * d; idx += num_threads) {
            int qi = idx / d;
            int c = idx % d;
            float acc = 0.0f;
            for (int ki = 0; ki < kv_len; ++ki) {
                acc += sS[qi * Bc + ki] * sKV[ki * d + c];
            }
            sO[qi * d + c] += acc;
        }
        __syncthreads();
    }

    // --- Phase 4: Normalize output and write ---
    for (int idx = tid; idx < q_len * d; idx += num_threads) {
        int qi = idx / d;
        int c = idx % d;
        float l_val = row_l[qi];
        float o_val = (l_val > 0.0f) ? sO[qi * d + c] / l_val : 0.0f;
        O_ptr[qi * d + c] = common::cuda::Cast<T>(o_val);
    }
    // Write logsumexp L = m + log(l)
    for (int qi = tid; qi < q_len; qi += num_threads) {
        L_ptr[qi] = (row_l[qi] > 0.0f) ? row_m[qi] + logf(row_l[qi]) : -INFINITY;
    }
}

// ============================================================================
// FlashAttention Backward Kernel
// ============================================================================
//
// Recomputation-based backward: recomputes attention weights from Q, K, V, L
// to avoid storing the N x N attention matrix.
//
// Uses float accumulators for dK, dV (written to float buffers).
// This avoids atomicAdd issues with bf16 and ensures numerical precision.
//
// Shared memory layout (all float):
//   sQ  [Br * d]       - query tile
//   sdO [Br * d]       - dO tile
//   sKV [Bc * d]       - key or value tile (reused)
//   sS  [Br * Bc]      - attention scores / P / dS (reused)
//   sD  [Br]           - D = rowsum(dO * O) for each query row
//   sdQ [Br * d]        - dQ accumulator
//   sL  [Br]           - logsumexp for each query row
template <int Br, int Bc, typename T>
__global__ void FlashAttnBwdKernel(const T *__restrict__ dO_global,  // [B, H_q, N, d]
                                   const T *__restrict__ Q,          // [B, H_q, N, d]
                                   const T *__restrict__ K,          // [B, H_kv, N, d]
                                   const T *__restrict__ V,          // [B, H_kv, N, d]
                                   const T *__restrict__ O,          // [B, H_q, N, d]
                                   const float *__restrict__ L,      // [B, H_q, N]
                                   float *__restrict__ dQ_global,    // [B, H_q, N, d] (float)
                                   float *__restrict__ dK_global,    // [B, H_kv, N, d] (float)
                                   float *__restrict__ dV_global,    // [B, H_kv, N, d] (float)
                                   int N, int d, int H_q, int H_kv, float scale, bool is_causal, float dropout_p,
                                   unsigned long long rng_seed) {
    const int q_tile_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / H_q;
    const int head_idx = bh_idx % H_q;
    const int kv_head_idx = H_kv == H_q ? head_idx : head_idx / (H_q / H_kv);
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int q_start = q_tile_idx * Br;
    if (q_start >= N) {
        return;
    }
    const int q_len = min(Br, N - q_start);

    // Pointers
    const int64_t q_head_offset = ((int64_t)batch_idx * H_q + head_idx) * N * d;
    const int64_t kv_head_offset = ((int64_t)batch_idx * H_kv + kv_head_idx) * N * d;
    const T *Q_ptr = Q + q_head_offset + (int64_t)q_start * d;
    const T *dO_ptr = dO_global + q_head_offset + (int64_t)q_start * d;
    const T *O_ptr = O + q_head_offset + (int64_t)q_start * d;
    const float *L_ptr = L + ((int64_t)batch_idx * H_q + head_idx) * N + q_start;
    float *dQ_out = dQ_global + q_head_offset + (int64_t)q_start * d;
    const T *K_base = K + kv_head_offset;
    const T *V_base = V + kv_head_offset;
    float *dK_base = dK_global + kv_head_offset;
    float *dV_base = dV_global + kv_head_offset;

    // Shared memory
    extern __shared__ float smem[];
    float *sQ = smem;
    float *sdO = sQ + Br * d;
    float *sKV = sdO + Br * d;
    float *sS = sKV + Bc * d;
    float *sD = sS + Br * Bc;
    float *sdQ = sD + Br;
    float *sL = sdQ + Br * d;

    // Load Q and dO
    for (int idx = tid; idx < q_len * d; idx += num_threads) {
        int r = idx / d;
        int c = idx % d;
        sQ[r * d + c] = common::cuda::Cast<float>(Q_ptr[r * d + c]);
        sdO[r * d + c] = common::cuda::Cast<float>(dO_ptr[r * d + c]);
    }
    // Load L (logsumexp)
    for (int qi = tid; qi < q_len; qi += num_threads) {
        sL[qi] = L_ptr[qi];
    }
    // Compute D[qi] = sum_c dO[qi][c] * O[qi][c]
    for (int qi = tid; qi < q_len; qi += num_threads) {
        float d_val = 0.0f;
        for (int c = 0; c < d; ++c) {
            d_val += common::cuda::Cast<float>(dO_ptr[qi * d + c]) * common::cuda::Cast<float>(O_ptr[qi * d + c]);
        }
        sD[qi] = d_val;
    }
    // Initialize dQ accumulator
    for (int idx = tid; idx < q_len * d; idx += num_threads) {
        sdQ[idx] = 0.0f;
    }
    __syncthreads();

    const int num_kv_tiles = (N + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * Bc;
        const int kv_len = min(Bc, N - kv_start);

        if (is_causal && kv_start > q_start + q_len - 1) {
            break;
        }

        // --- Load K tile ---
        for (int idx = tid; idx < kv_len * d; idx += num_threads) {
            int r = idx / d;
            int c = idx % d;
            sKV[r * d + c] = common::cuda::Cast<float>(K_base[(kv_start + r) * d + c]);
        }
        __syncthreads();

        // --- Recompute S = Q @ K^T * scale, then P = exp(S - L) ---
        for (int idx = tid; idx < q_len * Bc; idx += num_threads) {
            int qi = idx / Bc;
            int ki = idx % Bc;
            if (ki < kv_len) {
                float dot = 0.0f;
                for (int c = 0; c < d; ++c) {
                    dot += sQ[qi * d + c] * sKV[ki * d + c];
                }
                dot *= scale;

                if (is_causal && (kv_start + ki) > (q_start + qi)) {
                    sS[qi * Bc + ki] = 0.0f;
                } else {
                    float p = expf(dot - sL[qi]);
                    if (dropout_p > 0.0f && p > 0.0f) {
                        unsigned long long counter = (unsigned long long)(batch_idx * H_q + head_idx) * N * N
                                                   + (unsigned long long)(q_start + qi) * N + (kv_start + ki);
                        float r = philox_uniform(counter, rng_seed);
                        p = (r < dropout_p) ? 0.0f : p / (1.0f - dropout_p);
                    }
                    sS[qi * Bc + ki] = p;
                }
            } else {
                sS[qi * Bc + ki] = 0.0f;
            }
        }
        __syncthreads();

        // --- dV += P^T @ dO (before overwriting sKV with V) ---
        // dV[ki][c] += sum_qi P[qi][ki] * dO[qi][c]
        // Write to float buffer via atomicAdd (safe for GQA)
        for (int idx = tid; idx < kv_len * d; idx += num_threads) {
            int ki = idx / d;
            int c = idx % d;
            float acc = 0.0f;
            for (int qi = 0; qi < q_len; ++qi) {
                acc += sS[qi * Bc + ki] * sdO[qi * d + c];
            }
            atomicAdd(&dV_base[(kv_start + ki) * d + c], acc);
        }
        __syncthreads();

        // --- Load V tile into sKV (reuse space since K is no longer needed) ---
        for (int idx = tid; idx < kv_len * d; idx += num_threads) {
            int r = idx / d;
            int c = idx % d;
            sKV[r * d + c] = common::cuda::Cast<float>(V_base[(kv_start + r) * d + c]);
        }
        __syncthreads();

        // --- Compute dS = P * (dP - D), where dP[qi][ki] = sum_c dO[qi][c] * V[ki][c] ---
        for (int idx = tid; idx < q_len * kv_len; idx += num_threads) {
            int qi = idx / kv_len;
            int ki = idx % kv_len;
            float dp = 0.0f;
            for (int c = 0; c < d; ++c) {
                dp += sdO[qi * d + c] * sKV[ki * d + c];
            }
            float p = sS[qi * Bc + ki];
            sS[qi * Bc + ki] = p * (dp - sD[qi]); // dS overwrites P
        }
        __syncthreads();

        // --- Reload K tile for dQ and dK computation ---
        for (int idx = tid; idx < kv_len * d; idx += num_threads) {
            int r = idx / d;
            int c = idx % d;
            sKV[r * d + c] = common::cuda::Cast<float>(K_base[(kv_start + r) * d + c]);
        }
        __syncthreads();

        // dQ += dS @ K * scale
        for (int idx = tid; idx < q_len * d; idx += num_threads) {
            int qi = idx / d;
            int c = idx % d;
            float acc = 0.0f;
            for (int ki = 0; ki < kv_len; ++ki) {
                acc += sS[qi * Bc + ki] * sKV[ki * d + c];
            }
            sdQ[qi * d + c] += acc * scale;
        }

        // dK += dS^T @ Q * scale (atomicAdd to float buffer for GQA safety)
        for (int idx = tid; idx < kv_len * d; idx += num_threads) {
            int ki = idx / d;
            int c = idx % d;
            float acc = 0.0f;
            for (int qi = 0; qi < q_len; ++qi) {
                acc += sS[qi * Bc + ki] * sQ[qi * d + c];
            }
            atomicAdd(&dK_base[(kv_start + ki) * d + c], acc * scale);
        }
        __syncthreads();
    }

    // Write dQ to float buffer
    for (int idx = tid; idx < q_len * d; idx += num_threads) {
        dQ_out[idx] = sdQ[idx];
    }
}

// ============================================================================
// Kernel to convert float gradient buffer to target dtype
// ============================================================================
template <typename T>
__global__ void ConvertFloatToType(const float *__restrict__ src, T *__restrict__ dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = common::cuda::Cast<T>(src[idx]);
    }
}

} // anonymous namespace

// ============================================================================
// Launch helpers
// ============================================================================

template <typename T>
void LaunchFlashAttnForward(const std::shared_ptr<Tensor> &Q, const std::shared_ptr<Tensor> &K,
                            const std::shared_ptr<Tensor> &V, std::shared_ptr<Tensor> &O, std::shared_ptr<Tensor> &L,
                            float scale, bool is_causal, float dropout_p, cudaStream_t stream) {
    const auto &dims = Q->Dims();
    const int B = dims[0];
    const int H_q = dims[1];
    const int N = dims[2];
    const int head_dim = dims[3];
    const int H_kv = K->Dims()[1];

    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int NUM_THREADS = 128;

    // Shared memory: sQ[Br*d] + sKV[Bc*d] + sS[Br*Bc] + row_m[Br] + row_l[Br] + sO[Br*d]
    size_t smem_size = (size_t)(Br * head_dim + Bc * head_dim + Br * Bc + Br + Br + Br * head_dim) * sizeof(float);

    dim3 grid((N + Br - 1) / Br, B * H_q);
    dim3 block(NUM_THREADS);

    unsigned long long rng_seed = 42;

    FlashAttnFwdKernel<Br, Bc, T><<<grid, block, smem_size, stream>>>(
        static_cast<const T *>(Q->DataPtr()), static_cast<const T *>(K->DataPtr()),
        static_cast<const T *>(V->DataPtr()), static_cast<T *>(O->DataPtr()), static_cast<float *>(L->DataPtr()), N,
        head_dim, H_q, H_kv, scale, is_causal, dropout_p, rng_seed);
}

template <typename T>
void LaunchFlashAttnBackward(const std::shared_ptr<Tensor> &dO, const std::shared_ptr<Tensor> &Q,
                             const std::shared_ptr<Tensor> &K, const std::shared_ptr<Tensor> &V,
                             const std::shared_ptr<Tensor> &O, const std::shared_ptr<Tensor> &L,
                             std::shared_ptr<Tensor> &dQ, std::shared_ptr<Tensor> &dK, std::shared_ptr<Tensor> &dV,
                             float scale, bool is_causal, float dropout_p, cudaStream_t stream) {
    const auto &dims = Q->Dims();
    const int B = dims[0];
    const int H_q = dims[1];
    const int N = dims[2];
    const int head_dim = dims[3];
    const int H_kv = K->Dims()[1];

    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int NUM_THREADS = 128;

    // Shared memory: sQ[Br*d] + sdO[Br*d] + sKV[Bc*d] + sS[Br*Bc] + sD[Br] + sdQ[Br*d] + sL[Br]
    size_t smem_size
        = (size_t)(Br * head_dim * 2 + Bc * head_dim + Br * Bc + Br + Br * head_dim + Br) * sizeof(float);

    dim3 grid((N + Br - 1) / Br, B * H_q);
    dim3 block(NUM_THREADS);

    unsigned long long rng_seed = 42;

    // Allocate float buffers for gradient accumulation (required for atomicAdd with GQA + bf16)
    auto dQ_float = std::make_shared<Tensor>(Q->Dims(), DataType::kFLOAT32, Q->GetDevice());
    auto dK_float = std::make_shared<Tensor>(K->Dims(), DataType::kFLOAT32, K->GetDevice());
    auto dV_float = std::make_shared<Tensor>(V->Dims(), DataType::kFLOAT32, V->GetDevice());

    cudaMemsetAsync(dQ_float->DataPtr(), 0, dQ_float->NumElements() * sizeof(float), stream);
    cudaMemsetAsync(dK_float->DataPtr(), 0, dK_float->NumElements() * sizeof(float), stream);
    cudaMemsetAsync(dV_float->DataPtr(), 0, dV_float->NumElements() * sizeof(float), stream);

    FlashAttnBwdKernel<Br, Bc, T><<<grid, block, smem_size, stream>>>(
        static_cast<const T *>(dO->DataPtr()), static_cast<const T *>(Q->DataPtr()),
        static_cast<const T *>(K->DataPtr()), static_cast<const T *>(V->DataPtr()),
        static_cast<const T *>(O->DataPtr()), static_cast<const float *>(L->DataPtr()),
        static_cast<float *>(dQ_float->DataPtr()), static_cast<float *>(dK_float->DataPtr()),
        static_cast<float *>(dV_float->DataPtr()), N, head_dim, H_q, H_kv, scale, is_causal, dropout_p, rng_seed);

    // Convert float gradients to target dtype
    if constexpr (std::is_same_v<T, float>) {
        // Already float: just copy the data
        cudaMemcpyAsync(dQ->DataPtr(), dQ_float->DataPtr(), dQ_float->NumElements() * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(dK->DataPtr(), dK_float->DataPtr(), dK_float->NumElements() * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(dV->DataPtr(), dV_float->DataPtr(), dV_float->NumElements() * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        // Convert float -> T (e.g., bf16)
        constexpr int kConvertThreads = 256;
        int64_t nQ = dQ_float->NumElements();
        int64_t nK = dK_float->NumElements();
        int64_t nV = dV_float->NumElements();

        ConvertFloatToType<T><<<(nQ + kConvertThreads - 1) / kConvertThreads, kConvertThreads, 0, stream>>>(
            static_cast<const float *>(dQ_float->DataPtr()), static_cast<T *>(dQ->DataPtr()), nQ);
        ConvertFloatToType<T><<<(nK + kConvertThreads - 1) / kConvertThreads, kConvertThreads, 0, stream>>>(
            static_cast<const float *>(dK_float->DataPtr()), static_cast<T *>(dK->DataPtr()), nK);
        ConvertFloatToType<T><<<(nV + kConvertThreads - 1) / kConvertThreads, kConvertThreads, 0, stream>>>(
            static_cast<const float *>(dV_float->DataPtr()), static_cast<T *>(dV->DataPtr()), nV);
    }
}

// ============================================================================
// Dispatcher-registered functions
// ============================================================================

std::vector<std::shared_ptr<Tensor>> FlashAttentionForward(const std::shared_ptr<Tensor> &query,
                                                           const std::shared_ptr<Tensor> &key,
                                                           const std::shared_ptr<Tensor> &value, float scale,
                                                           bool is_causal, float dropout_p) {
    const auto &dims = query->Dims();
    auto dtype = query->Dtype();
    auto device = query->GetDevice();

    auto output = std::make_shared<Tensor>(dims, dtype, device);
    auto logsumexp
        = std::make_shared<Tensor>(std::vector<int64_t>{dims[0], dims[1], dims[2]}, DataType::kFLOAT32, device);

    auto stream = GetCudaStream(device);

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchFlashAttnForward<float>(query, key, value, output, logsumexp, scale, is_causal,
                                                         dropout_p, stream);),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchFlashAttnForward<nv_bfloat16>(query, key, value, output, logsumexp, scale, is_causal,
                                                               dropout_p, stream);),
                      DataType::kBFLOAT16)
    default:
        LOG(FATAL) << "FlashAttention forward: unsupported dtype";
    }

    return {output, logsumexp};
}

std::vector<std::shared_ptr<Tensor>>
FlashAttentionBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &query,
                       const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
                       const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &logsumexp, float scale,
                       bool is_causal, float dropout_p) {
    auto dtype = query->Dtype();
    auto device = query->GetDevice();

    auto dQ = std::make_shared<Tensor>(query->Dims(), dtype, device);
    auto dK = std::make_shared<Tensor>(key->Dims(), dtype, device);
    auto dV = std::make_shared<Tensor>(value->Dims(), dtype, device);

    auto stream = GetCudaStream(device);

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchFlashAttnBackward<float>(grad_output, query, key, value, output, logsumexp, dQ, dK, dV,
                                                          scale, is_causal, dropout_p, stream);),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchFlashAttnBackward<nv_bfloat16>(grad_output, query, key, value, output, logsumexp, dQ,
                                                                dK, dV, scale, is_causal, dropout_p, stream);),
                      DataType::kBFLOAT16)
    default:
        LOG(FATAL) << "FlashAttention backward: unsupported dtype";
    }

    return {dQ, dK, dV};
}

} // namespace infini_train::kernels::cuda

// Register kernels with the dispatcher
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, FlashAttentionForward,
                infini_train::kernels::cuda::FlashAttentionForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, FlashAttentionBackward,
                infini_train::kernels::cuda::FlashAttentionBackward)

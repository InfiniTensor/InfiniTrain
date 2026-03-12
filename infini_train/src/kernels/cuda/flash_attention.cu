#include "infini_train/src/kernels/cuda/flash_attention.h"

#include <cuda_runtime.h>
#include <iostream>
#include <limits>

#include "glog/logging.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

// A simple FlashAttention kernel implementation (forward pass)
// Assumptions for this implementation:
// - Head dimension D is fixed (e.g. 64) or handled via template
// - Float32 precision (for simplicity and stability in this demo)
// - Supports causal masking
// - Supports scaling (1/sqrt(D))

// Constants for block sizes
constexpr int Br = 32; // Block size for rows (Q)
constexpr int Bc = 32; // Block size for columns (K, V)
constexpr int D_static = 64; // Head dimension (typical for GPT-2 small / Llama-3 small)
constexpr int D_max = 256;

// Helper for loading from HBM to Shared Memory
__device__ void load_block(const float* src, float* dst, int rows, int cols, int stride, int tid, int total_threads) {
    int elements = rows * cols;
    for (int i = tid; i < elements; i += total_threads) {
        int r = i / cols;
        int c = i % cols;
        dst[r * cols + c] = src[r * stride + c];
    }
}

// Helper for storing to HBM
__device__ void store_block(float* dst, const float* src, int rows, int cols, int stride, int tid, int total_threads) {
    int elements = rows * cols;
    for (int i = tid; i < elements; i += total_threads) {
        int r = i / cols;
        int c = i % cols;
        dst[r * stride + c] = src[r * cols + c];
    }
}

// Helper for GEMM: C = A * B^T
// A: (M, K), B: (N, K), C: (M, N)
// Simplified naive implementation
__device__ void gemm_ab_t(const float* A, const float* B, float* C, int M, int N, int K, int tid, int total_threads) {
    int elements = M * N;
    for (int i = tid; i < elements; i += total_threads) {
        int r = i / N;
        int c = i % N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[r * K + k] * B[c * K + k];
        }
        C[r * N + c] = sum;
    }
}

// Helper for GEMM: C = A * B
// A: (M, K), B: (K, N), C: (M, N)
__device__ void gemm_ab(const float* A, const float* B, float* C, int M, int N, int K, int tid, int total_threads) {
    int elements = M * N;
    for (int i = tid; i < elements; i += total_threads) {
        int r = i / N;
        int c = i % N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = sum; // Accumulate logic is handled outside (C += ...)
    }
}

// Helper for GEMM accumulation: C += A * B
__device__ void gemm_ab_accum(const float* A, const float* B, float* C, int M, int N, int K, int tid, int total_threads) {
    int elements = M * N;
    for (int i = tid; i < elements; i += total_threads) {
        int r = i / N;
        int c = i % N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] += sum;
    }
}

__global__ void FlashAttentionForwardKernel(const float* Q, const float* K, const float* V, float* O, float* L,
                                            int B, int T, int H, int D,
                                            float softmax_scale, bool is_causal) {
    // Grid: (B, H)
    // Block: (Br, D) or sufficient threads
    // Each block processes one attention head for one sequence
    
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // Offset to the specific head
    int batch_head_offset = (batch_idx * H + head_idx) * T * D;
    const float* q_base = Q + batch_head_offset;
    const float* k_base = K + batch_head_offset;
    const float* v_base = V + batch_head_offset;
    float* o_base = O + batch_head_offset;
    
    // LSE offset: (B, H, T)
    float* l_base = L + (batch_idx * H + head_idx) * T;

    // Shared memory allocation
    // Need space for:
    // Qi: Br * D
    // Kj: Bc * D
    // Vj: Bc * D
    // Sij: Br * Bc
    // Pij: Br * Bc (can reuse Sij)
    // Oi: Br * D
    // li: Br
    // mi: Br
    
    // Total simplified shared memory layout:
    // [Qi (Br*D)] [Kj (Bc*D)] [Vj (Bc*D)] [Sij (Br*Bc)] [Oi (Br*D)] [li (Br)] [mi (Br)]
    // For Br=32, Bc=32, D=64:
    // Qi: 32*64 = 2048
    // Kj: 32*64 = 2048
    // Vj: 32*64 = 2048
    // Sij: 32*32 = 1024
    // Oi: 32*64 = 2048
    // li: 32
    // mi: 32
    // Total floats: ~9280 floats = ~37KB (fits in 48KB shared mem)
    
    extern __shared__ float sram[];
    float* s_Qi = sram;
    float* s_Kj = s_Qi + Br * D;
    float* s_Vj = s_Kj + Bc * D;
    float* s_Sij = s_Vj + Bc * D;
    float* s_Oi = s_Sij + Br * Bc;
    float* s_li = s_Oi + Br * D;
    float* s_mi = s_li + Br;

    // Outer loop: iterate over blocks of Q
    // T is split into Tr blocks of size Br
    int Tr = (T + Br - 1) / Br;
    int Tc = (T + Bc - 1) / Bc;

    for (int i = 0; i < Tr; ++i) {
        int row_start = i * Br;
        int row_end = min(row_start + Br, T);
        int valid_rows = row_end - row_start;

        // 1. Load Qi from HBM to SRAM
        load_block(q_base + row_start * D, s_Qi, valid_rows, D, D, tid, total_threads);
        
        // 2. Initialize Oi, li, mi in SRAM
        // Oi = 0
        for (int k = tid; k < Br * D; k += total_threads) s_Oi[k] = 0.0f;
        // li = 0
        for (int k = tid; k < Br; k += total_threads) s_li[k] = 0.0f;
        // mi = -inf
        for (int k = tid; k < Br; k += total_threads) s_mi[k] = -1e20f; // -inf

        __syncthreads();

        // Inner loop: iterate over blocks of K, V
        for (int j = 0; j < Tc; ++j) {
            int col_start = j * Bc;
            int col_end = min(col_start + Bc, T);
            int valid_cols = col_end - col_start;

            // Causal masking optimization: if row block is entirely before col block, skip
            if (is_causal && col_start > row_end) continue; // strictly upper triangular blocks

            // 3. Load Kj, Vj from HBM to SRAM
            load_block(k_base + col_start * D, s_Kj, valid_cols, D, D, tid, total_threads);
            load_block(v_base + col_start * D, s_Vj, valid_cols, D, D, tid, total_threads);
            
            __syncthreads();

            // 4. Compute Sij = Qi * Kj^T
            // s_Qi: (valid_rows, D), s_Kj: (valid_cols, D) -> s_Sij: (valid_rows, valid_cols)
            gemm_ab_t(s_Qi, s_Kj, s_Sij, valid_rows, valid_cols, D, tid, total_threads);
            
            __syncthreads();

            // 5. Apply Mask, Scale, Softmax
            // Iterate over Sij elements
            for (int k = tid; k < valid_rows * valid_cols; k += total_threads) {
                int r = k / valid_cols;
                int c = k % valid_cols;
                int global_row = row_start + r;
                int global_col = col_start + c;
                
                if (is_causal && global_col > global_row) {
                    s_Sij[k] = -1e20f; // Masked out
                } else {
                    s_Sij[k] *= softmax_scale;
                }
            }
            __syncthreads();

            // 6. Online Softmax Update
            // For each row in current block
            for (int r = tid; r < valid_rows; r += total_threads) {
                float m_prev = s_mi[r];
                
                // Find max in current row of Sij
                float m_curr = -1e20f;
                for (int c = 0; c < valid_cols; ++c) {
                    float val = s_Sij[r * valid_cols + c];
                    if (val > m_curr) m_curr = val;
                }
                
                // New max
                float m_new = max(m_prev, m_curr);
                
                // Compute Pij (exp(Sij - m_new)) and rowsum
                float l_curr = 0.0f;
                for (int c = 0; c < valid_cols; ++c) {
                    float val = s_Sij[r * valid_cols + c];
                    float p = expf(val - m_new);
                    s_Sij[r * valid_cols + c] = p; // Store Pij in place of Sij
                    l_curr += p;
                }
                
                // Update li and mi
                float l_prev = s_li[r];
                // l_new = exp(m_prev - m_new) * l_prev + l_curr
                float alpha = expf(m_prev - m_new);
                float l_new = alpha * l_prev + l_curr;
                
                s_mi[r] = m_new;
                s_li[r] = l_new;
                
                // Rescale Oi: Oi = Oi * alpha
                for (int d = 0; d < D; ++d) {
                    s_Oi[r * D + d] *= alpha;
                }
            }
            __syncthreads();
            
            // 7. Compute Oi += Pij * Vj
            // s_Sij (Pij): (valid_rows, valid_cols), s_Vj: (valid_cols, D) -> s_Oi: (valid_rows, D)
            // Accumulate into s_Oi
            // gemm_ab_accum(A, B, C, M, N, K) -> C(M, N) += A(M, K) * B(K, N)
            // M = valid_rows
            // N = D
            // K = valid_cols
            gemm_ab_accum(s_Sij, s_Vj, s_Oi, valid_rows, D, valid_cols, tid, total_threads);
            
            __syncthreads();
        }

        // 8. Finalize Oi and store to HBM
        // Oi = Oi / li
        // Store li + mi as LSE (LogSumExp) = log(li) + mi
        for (int r = tid; r < valid_rows; r += total_threads) {
            float li = s_li[r];
            float mi = s_mi[r];
            float inv_li = 1.0f / (li + 1e-6f); // Avoid div by zero
            
            for (int d = 0; d < D; ++d) {
                s_Oi[r * D + d] *= inv_li;
            }
            
            // Store LSE for backward
            // LSE = mi + log(li)
            l_base[row_start + r] = mi + logf(li + 1e-6f);
        }
        __syncthreads();

        // Store Oi to HBM
        store_block(o_base + row_start * D, s_Oi, valid_rows, D, D, tid, total_threads);
        __syncthreads();
    }
}

__global__ void FlashAttentionBackwardKernel(const float *Q, const float *K, const float *V, const float *dO, float *dQ,
                                             float *dK, float *dV, const float *L, int T, int H, int D,
                                             float softmax_scale, bool is_causal) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;
    int tid = threadIdx.x;
    if (D > D_max) {
        return;
    }

    size_t base = static_cast<size_t>(b * H + h) * T * D;
    const float *q = Q + base;
    const float *k = K + base;
    const float *v = V + base;
    const float *go = dO + base;
    float *gq = dQ + base;
    float *gk = dK + base;
    float *gv = dV + base;
    const float *l = L + (b * H + h) * T;

    int row_off = i * D;
    float l_val = l[i]; // LSE for current row

    // Pass 1: Compute dpsum
    float dpsum = 0.0f;
    for (int j = 0; j < T; ++j) {
        if (is_causal && j > i) {
            continue;
        }
        int col_off = j * D;
        
        // Compute S
        float s = 0.0f;
        for (int d = 0; d < D; ++d) {
            s += q[row_off + d] * k[col_off + d];
        }
        
        // Compute P = exp(S * scale - L)
        float p = expf(s * softmax_scale - l_val);
        
        // Compute dP (partial) = dO . V
        float dp = 0.0f;
        for (int d = 0; d < D; ++d) {
            dp += go[row_off + d] * v[col_off + d];
        }
        
        dpsum += dp * p;
    }

    // Initialize dQ accumulator
    for (int d = tid; d < D; d += blockDim.x) {
        gq[row_off + d] = 0.0f;
    }
    __syncthreads();

    // Pass 2: Compute Gradients
    for (int j = 0; j < T; ++j) {
        if (is_causal && j > i) {
            continue;
        }
        int col_off = j * D;
        
        // Recompute S and P
        float s = 0.0f;
        for (int d = 0; d < D; ++d) {
            s += q[row_off + d] * k[col_off + d];
        }
        float p = expf(s * softmax_scale - l_val);
        
        // Recompute dP
        float dp = 0.0f;
        for (int d = 0; d < D; ++d) {
            dp += go[row_off + d] * v[col_off + d];
        }
        
        // Compute dS
        float ds = p * (dp - dpsum);
        
        // Update dQ (accumulate in registers/shared, then write)
        // Here we write directly to global memory gq (initialized to 0)
        // Since this thread block owns row i, no atomic needed for gq
        for (int d = tid; d < D; d += blockDim.x) {
            atomicAdd(gq + row_off + d, softmax_scale * ds * k[col_off + d]);
        }

        // Update dK and dV (atomic needed)
        for (int d = tid; d < D; d += blockDim.x) {
            atomicAdd(gk + col_off + d, softmax_scale * ds * q[row_off + d]);
            atomicAdd(gv + col_off + d, p * go[row_off + d]);
        }
    }
}

void FlashAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output, Tensor &softmax_lse,
                           float dropout_p, float softmax_scale, bool is_causal, const Device &device) {
    // Get dimensions
    auto dims = q.Dims(); // (B, H, T, D)
    int B = dims[0];
    int H = dims[1];
    int T = dims[2];
    int D = dims[3];

    // Check data type - only supporting float32 for stub
    if (q.Dtype() != DataType::kFLOAT32) {
        LOG(WARNING) << "FlashAttention currently only supports float32";
        return;
    }
    
    // Check D
    if (D != D_static) {
        LOG(WARNING) << "FlashAttention currently only supports D=" << D_static << ", but got " << D;
        // Fallback or error? For now, let's proceed but warn.
        // Ideally we should use templates or dynamic shared mem sizing.
        // If D > 64, this kernel will likely crash or produce garbage due to fixed shared mem layout.
        if (D > D_static) {
             LOG(ERROR) << "Head dimension " << D << " too large for current implementation (max " << D_static << ")";
             return;
        }
    }

    const float* q_ptr = static_cast<const float*>(q.DataPtr());
    const float* k_ptr = static_cast<const float*>(k.DataPtr());
    const float* v_ptr = static_cast<const float*>(v.DataPtr());
    float* out_ptr = static_cast<float*>(output.DataPtr());
    float* l_ptr = static_cast<float*>(softmax_lse.DataPtr());

    // Grid: (B, H)
    dim3 grid(B, H);
    // Threads: enough to cover operations. 
    // Br*D = 32*64 = 2048 elements. Max threads per block usually 1024.
    // We iterate inside kernel, so 256 threads is fine.
    dim3 block(256);
    
    // Shared memory size
    // See layout in kernel
    size_t sram_size = (Br * D + Bc * D + Bc * D + Br * Bc + Br * D + Br + Br) * sizeof(float);
    
    // Ensure we are on the correct stream if provided (device.stream)
    cudaStream_t stream = 0; // TODO: get from device

    FlashAttentionForwardKernel<<<grid, block, sram_size, stream>>>(
        q_ptr, k_ptr, v_ptr, out_ptr, l_ptr, B, T, H, D, softmax_scale, is_causal
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "CUDA Error in FlashAttentionForward: " << cudaGetErrorString(err);
    }
}

void FlashAttentionBackward(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &grad_output, Tensor &grad_q,
                            Tensor &grad_k, Tensor &grad_v, const Tensor &softmax_lse, float softmax_scale,
                            bool is_causal, const Device &device) {
    auto dims = q.Dims();
    int B = dims[0];
    int H = dims[1];
    int T = dims[2];
    int D = dims[3];
    if (q.Dtype() != DataType::kFLOAT32 || grad_output.Dtype() != DataType::kFLOAT32 || D > D_max) {
        LOG(FATAL) << "FlashAttentionBackward only supports float32 with head_dim <= " << D_max;
    }

    auto *q_ptr = static_cast<const float *>(q.DataPtr());
    auto *k_ptr = static_cast<const float *>(k.DataPtr());
    auto *v_ptr = static_cast<const float *>(v.DataPtr());
    auto *go_ptr = static_cast<const float *>(grad_output.DataPtr());
    auto *gq_ptr = static_cast<float *>(grad_q.DataPtr());
    auto *gk_ptr = static_cast<float *>(grad_k.DataPtr());
    auto *gv_ptr = static_cast<float *>(grad_v.DataPtr());
    auto *l_ptr = static_cast<const float *>(softmax_lse.DataPtr());

    size_t bytes = grad_q.SizeInBytes();
    cudaMemset(gq_ptr, 0, bytes);
    cudaMemset(gk_ptr, 0, bytes);
    cudaMemset(gv_ptr, 0, bytes);

    dim3 grid(B, H, T);
    dim3 block(128);
    FlashAttentionBackwardKernel<<<grid, block>>>(q_ptr, k_ptr, v_ptr, go_ptr, gq_ptr, gk_ptr, gv_ptr, l_ptr, T, H, D,
                                                  softmax_scale, is_causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "CUDA Error in FlashAttentionBackward: " << cudaGetErrorString(err);
    }
}

} // namespace infini_train::kernels::cuda

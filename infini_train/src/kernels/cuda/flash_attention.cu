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

// Helper to load float4
// src_stride: stride of src in floats
// dst_stride: stride of dst in floats (must be multiple of 4)
__device__ __forceinline__ void load_float4(const float* src, float* dst, int rows, int cols, int src_stride, int dst_stride, int tid, int total_threads) {
    int cols_vec = cols / 4;
    int elements_vec = rows * cols_vec;
    
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    float4* dst_vec = reinterpret_cast<float4*>(dst);
    
    int dst_stride_vec = dst_stride / 4;

    for (int i = tid; i < elements_vec; i += total_threads) {
        int r = i / cols_vec;
        int c = i % cols_vec;
        dst_vec[r * dst_stride_vec + c] = src_vec[r * (src_stride / 4) + c];
    }
}

// Helper to store float4
__device__ __forceinline__ void store_float4(float* dst, const float* src, int rows, int cols, int src_stride, int dst_stride, int tid, int total_threads) {
    int cols_vec = cols / 4;
    int elements_vec = rows * cols_vec;
    
    float4* dst_vec = reinterpret_cast<float4*>(dst);
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    
    int src_stride_vec = src_stride / 4;

    for (int i = tid; i < elements_vec; i += total_threads) {
        int r = i / cols_vec;
        int c = i % cols_vec;
        dst_vec[r * (dst_stride / 4) + c] = src_vec[r * src_stride_vec + c];
    }
}

// Helper to load and transpose float4
// dst_stride: stride of dst in floats (must be multiple of 4)
// dst is (cols, rows) logically
__device__ __forceinline__ void load_transpose_float4(const float* src, float* dst, int rows, int cols, int src_stride, int dst_stride, int tid, int total_threads) {
    int cols_vec = cols / 4;
    int elements_vec = rows * cols_vec;
    
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    
    for (int i = tid; i < elements_vec; i += total_threads) {
        int r = i / cols_vec;
        int c_vec = i % cols_vec;
        int c = c_vec * 4;
        
        float4 val = src_vec[r * (src_stride / 4) + c_vec];
        
        // dst[c, r]
        dst[c * dst_stride + r] = val.x;
        dst[(c + 1) * dst_stride + r] = val.y;
        dst[(c + 2) * dst_stride + r] = val.z;
        dst[(c + 3) * dst_stride + r] = val.w;
    }
}

// Helper for GEMM with float4: C = A * B^T
// A: (M, K) with stride_A
// B: (N, K) with stride_B
// C: (M, N) with stride_C
__device__ __forceinline__ void gemm_ab_t_float4(const float* A, const float* B, float* C, int M, int N, int K, 
                                                 int stride_A, int stride_B, int stride_C,
                                                 int tid, int total_threads) {
    int elements = M * N;
    int K_vec = K / 4;
    const float4* A_vec = reinterpret_cast<const float4*>(A);
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    
    int stride_A_vec = stride_A / 4;
    int stride_B_vec = stride_B / 4;

    for (int i = tid; i < elements; i += total_threads) {
        int r = i / N;
        int c = i % N;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_vec; ++k) {
            float4 a = A_vec[r * stride_A_vec + k];
            float4 b = B_vec[c * stride_B_vec + k];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        // C might be padded too
        C[r * stride_C + c] = sum;
    }
}

// Helper for GEMM Accum with float4: C += A * B
// A: (M, K) with stride_A
// B_T: (N, K) with stride_B_T (Transposed B)
// C: (M, N) with stride_C
__device__ __forceinline__ void gemm_ab_accum_float4_transposed_b(const float* A, const float* B_T, float* C, int M, int N, int K, 
                                                                  int stride_A, int stride_B_T, int stride_C,
                                                                  int tid, int total_threads) {
    int elements = M * N;
    int K_vec = K / 4;
    const float4* A_vec = reinterpret_cast<const float4*>(A);
    const float4* B_T_vec = reinterpret_cast<const float4*>(B_T);
    
    int stride_A_vec = stride_A / 4;
    int stride_B_T_vec = stride_B_T / 4;

    for (int i = tid; i < elements; i += total_threads) {
        int r = i / N;
        int c = i % N;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_vec; ++k) {
            float4 a = A_vec[r * stride_A_vec + k];
            float4 b = B_T_vec[c * stride_B_T_vec + k];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        C[r * stride_C + c] += sum;
    }
}

__global__ void FlashAttentionForwardKernel(const float* Q, const float* K, const float* V, float* O, float* L,
                                            int B, int T, int H, int D,
                                            float softmax_scale, bool is_causal) {
    // Grid: (B, H, Tr)
    // Block: (Br, D) or sufficient threads
    // Each block processes one tile of rows (Br) for one attention head
    
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row_tile_idx = blockIdx.z; // Replaces outer loop i
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

    // Padding to avoid shared memory bank conflicts
    constexpr int Pad = 4;
    int D_pad = D + Pad;
    int Bc_pad = Bc + Pad;

    // Shared memory allocation
    // Need space for:
    // Qi: Br * D_pad
    // Kj: Bc * D_pad
    // Vj: D * Bc_pad (Transposed V)
    // Sij: Br * Bc_pad
    // Oi: Br * D_pad
    // li: Br
    // mi: Br
    
    extern __shared__ float sram[];
    float* s_Qi = sram;
    float* s_Kj = s_Qi + Br * D_pad;
    float* s_Vj = s_Kj + Bc * D_pad;
    float* s_Sij = s_Vj + D * Bc_pad; 
    float* s_Oi = s_Sij + Br * Bc_pad;
    float* s_li = s_Oi + Br * D_pad;
    float* s_mi = s_li + Br;

    // T is split into Tr blocks of size Br
    // int Tr = (T + Br - 1) / Br; // Handled by Grid
    int Tc = (T + Bc - 1) / Bc;

    // Single tile processing
    int i = row_tile_idx;
    {
        int row_start = i * Br;
        // Check boundary
        if (row_start >= T) return;
        
        int row_end = min(row_start + Br, T);
        int valid_rows = row_end - row_start;

        // 1. Load Qi from HBM to SRAM
        load_float4(q_base + row_start * D, s_Qi, valid_rows, D, D, D_pad, tid, total_threads);
        
        // 2. Initialize Oi, li, mi in SRAM
        for (int k = tid; k < Br * D_pad; k += total_threads) s_Oi[k] = 0.0f; // Note: Clearing padding is safe but wasteful, strictly only need Br*D? No, we use stride D_pad.
        for (int k = tid; k < Br; k += total_threads) s_li[k] = 0.0f;
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
            load_float4(k_base + col_start * D, s_Kj, valid_cols, D, D, D_pad, tid, total_threads);
            // Load V and transpose to (D, valid_cols)
            load_transpose_float4(v_base + col_start * D, s_Vj, valid_cols, D, D, Bc_pad, tid, total_threads);
            
            __syncthreads();

            // 4. Compute Sij = Qi * Kj^T
            gemm_ab_t_float4(s_Qi, s_Kj, s_Sij, valid_rows, valid_cols, D, D_pad, D_pad, Bc_pad, tid, total_threads);
            
            __syncthreads();

            // 5. Apply Mask, Scale, Softmax
            for (int k = tid; k < valid_rows * valid_cols; k += total_threads) {
                int r = k / valid_cols;
                int c = k % valid_cols;
                int global_row = row_start + r;
                int global_col = col_start + c;
                
                // Use Bc_pad stride
                float* s_Sij_ptr = s_Sij + r * Bc_pad + c;
                
                if (is_causal && global_col > global_row) {
                    *s_Sij_ptr = -1e20f; // Masked out
                } else {
                    *s_Sij_ptr *= softmax_scale;
                }
            }
            __syncthreads();

            // 6. Online Softmax Update
            for (int r = tid; r < valid_rows; r += total_threads) {
                float m_prev = s_mi[r];
                
                // Find max in current row of Sij
                float m_curr = -1e20f;
                for (int c = 0; c < valid_cols; ++c) {
                    float val = s_Sij[r * Bc_pad + c];
                    if (val > m_curr) m_curr = val;
                }
                
                float m_new = max(m_prev, m_curr);
                float l_curr = 0.0f;
                for (int c = 0; c < valid_cols; ++c) {
                    float val = s_Sij[r * Bc_pad + c];
                    float p = expf(val - m_new);
                    s_Sij[r * Bc_pad + c] = p; 
                    l_curr += p;
                }
                
                float l_prev = s_li[r];
                float alpha = expf(m_prev - m_new);
                float l_new = alpha * l_prev + l_curr;
                
                s_mi[r] = m_new;
                s_li[r] = l_new;
                
                for (int d = 0; d < D; ++d) {
                    s_Oi[r * D_pad + d] *= alpha;
                }
            }
            __syncthreads();
            
            // 7. Compute Oi += Pij * Vj
            gemm_ab_accum_float4_transposed_b(s_Sij, s_Vj, s_Oi, valid_rows, D, valid_cols, Bc_pad, Bc_pad, D_pad, tid, total_threads);
            
            __syncthreads();
        }

        // 8. Finalize Oi and store to HBM
        for (int r = tid; r < valid_rows; r += total_threads) {
            float li = s_li[r];
            float mi = s_mi[r];
            float inv_li = 1.0f / (li + 1e-6f); // Avoid div by zero
            
            for (int d = 0; d < D; ++d) {
                s_Oi[r * D_pad + d] *= inv_li;
            }
            
            l_base[row_start + r] = mi + logf(li + 1e-6f);
        }
        __syncthreads();

        // Store Oi to HBM
        store_float4(o_base + row_start * D, s_Oi, valid_rows, D, D_pad, D, tid, total_threads);
        // __syncthreads(); // Not needed at end of kernel
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

    // Grid: (B, H, Tr)
    int Tr = (T + Br - 1) / Br;
    dim3 grid(B, H, Tr);
    // Threads: enough to cover operations. 
    // Br*D = 32*64 = 2048 elements. Max threads per block usually 1024.
    // We iterate inside kernel, so 256 threads is fine.
    dim3 block(128);
    
    // Shared memory size
    // Padding
    constexpr int Pad = 4;
    int D_pad = D + Pad;
    int Bc_pad = Bc + Pad;
    
    // Need space for:
    // Qi: Br * D_pad
    // Kj: Bc * D_pad
    // Vj: D * Bc_pad
    // Sij: Br * Bc_pad
    // Oi: Br * D_pad
    // li: Br
    // mi: Br
    
    size_t sram_size = (Br * D_pad + Bc * D_pad + D * Bc_pad + Br * Bc_pad + Br * D_pad + Br + Br) * sizeof(float);
    
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

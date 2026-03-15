#include "infini_train/src/kernels/cuda/flash_attention.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <limits>

#include "glog/logging.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

using namespace nvcuda;

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

// WMMA Constants
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

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

// Helper to load float4 and convert to half
// dst_stride: stride of dst in halfs
__device__ __forceinline__ void load_float4_to_half(const float* src, half* dst, int rows, int cols, int src_stride, int dst_stride, int tid, int total_threads) {
    int cols_vec = cols / 4;
    int elements_vec = rows * cols_vec;
    
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    // Store as half2 if possible, but let's stick to simple casting for now to avoid alignment issues if dst_stride is odd
    // But D_pad usually multiple of 8 (e.g. 64+4=68? 68 is multiple of 4 halfs (8 bytes)? No, 68*2 = 136 bytes. 136/8 = 17. Yes.)
    
    for (int i = tid; i < elements_vec; i += total_threads) {
        int r = i / cols_vec;
        int c_vec = i % cols_vec;
        int c = c_vec * 4;
        
        float4 val = src_vec[r * (src_stride / 4) + c_vec];
        
        // Manual unroll store
        dst[r * dst_stride + c] = __float2half(val.x);
        dst[r * dst_stride + c + 1] = __float2half(val.y);
        dst[r * dst_stride + c + 2] = __float2half(val.z);
        dst[r * dst_stride + c + 3] = __float2half(val.w);
    }
}

// Helper to load and transpose float4 and convert to half
// dst_stride: stride of dst in halfs
// dst is (cols, rows) logically
__device__ __forceinline__ void load_transpose_float4_to_half(const float* src, half* dst, int rows, int cols, int src_stride, int dst_stride, int tid, int total_threads) {
    int cols_vec = cols / 4;
    int elements_vec = rows * cols_vec;
    
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    
    for (int i = tid; i < elements_vec; i += total_threads) {
        int r = i / cols_vec;
        int c_vec = i % cols_vec;
        int c = c_vec * 4;
        
        float4 val = src_vec[r * (src_stride / 4) + c_vec];
        
        // dst[c, r]
        dst[c * dst_stride + r] = __float2half(val.x);
        dst[(c + 1) * dst_stride + r] = __float2half(val.y);
        dst[(c + 2) * dst_stride + r] = __float2half(val.z);
        dst[(c + 3) * dst_stride + r] = __float2half(val.w);
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

// Helper for GEMM with float4: C += A * B^T
// A: (M, K) with stride_A
// B: (N, K) with stride_B
// C: (M, N) with stride_C
__device__ __forceinline__ void gemm_ab_accum_float4_transposed_b(const float* A, const float* B, float* C, int M, int N, int K, 
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
        // Accumulate
        C[r * stride_C + c] += sum;
    }
}

__global__ void FlashAttentionForwardKernelWMMA(const float* Q, const float* K, const float* V, float* O, float* L,
                                            int B, int T, int H, int D,
                                            float softmax_scale, bool is_causal) {
    // Grid: (B, H, Tr)
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row_tile_idx = blockIdx.z;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Offset to the specific head
    int batch_head_offset = (batch_idx * H + head_idx) * T * D;
    const float* q_base = Q + batch_head_offset;
    const float* k_base = K + batch_head_offset;
    const float* v_base = V + batch_head_offset;
    float* o_base = O + batch_head_offset;
    float* l_base = L + (batch_idx * H + head_idx) * T;

    // Padding (multiple of 8/16 for alignment)
    // 64 + 8 = 72. 72 * 2 bytes = 144 bytes. 144 is multiple of 16 (144/16=9). Good.
    constexpr int Pad = 8; 
    int D_pad = D + Pad;
    int Bc_pad = Bc + Pad;

    // Shared Memory Layout
    // s_Qi: Br * D_pad (half)
    // s_Kj: Bc * D_pad (half)
    // s_Vj: D * Bc_pad (half, transposed V)
    // s_Sij: Br * Bc_pad (float) - reuse for P
    // s_Oi: Br * D_pad (float) - accumulator
    // s_li, s_mi: Br (float)
    
    extern __shared__ char smem[];
    half* sram = reinterpret_cast<half*>(smem);
    half* s_Qi = sram;
    half* s_Kj = s_Qi + Br * D_pad;
    half* s_Vj = s_Kj + Bc * D_pad;
    
    // Float buffers start after half buffers (need alignment)
    // Align to 16 bytes (float4)
    size_t half_offset = (Br * D_pad + Bc * D_pad + D * Bc_pad);
    if (half_offset % 2 != 0) half_offset += 1; // Align to 4 bytes (float)
    // Actually align to 4 floats (16 bytes) -> 8 halfs
    while (half_offset % 8 != 0) half_offset++;
    
    float* s_Sij = reinterpret_cast<float*>(sram + half_offset);
    float* s_Oi = s_Sij + Br * Bc_pad;
    float* s_li = s_Oi + Br * D_pad;
    float* s_mi = s_li + Br;
    
    // WMMA Fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag; // For Oi

    int Tc = (T + Bc - 1) / Bc;
    
    // Initialize Oi fragments to 0
    // Each warp maintains a part of Oi.
    // However, Oi is (Br, D). 32x64.
    // Split into 8 16x16 tiles.
    // 4 Warps. Each Warp handles 2 tiles.
    // To simplify, we can keep Oi in Shared Memory (s_Oi) and accumulate there, 
    // or keep in registers (fragments) if possible.
    // If we keep in fragments, we need to scale them (multiply by alpha).
    // WMMA accumulator scaling: acc = acc * alpha + beta * C.
    // But we need `acc *= alpha`.
    // WMMA doesn't support direct scalar multiplication of accumulator.
    // We have to iterate over fragment elements? fragment elements are accessible.
    // Let's store Oi in fragments.
    
    // Warp mapping for Oi (32x64):
    // Warp 0: [0:16, 0:16], [0:16, 16:32] ? No, D is dim 1.
    // Oi tiles: (0,0), (0,1), (0,2), (0,3) [Row 0-15]
    //           (1,0), (1,1), (1,2), (1,3) [Row 16-31]
    // Total 8 tiles. 4 Warps.
    // Warp 0: (0,0), (0,1)
    // Warp 1: (0,2), (0,3)
    // Warp 2: (1,0), (1,1)
    // Warp 3: (1,2), (1,3)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> oi_frag[2];
    for (int k = 0; k < 2; ++k) wmma::fill_fragment(oi_frag[k], 0.0f);
    
    // Initialize li, mi
    for (int k = tid; k < Br; k += blockDim.x) {
        s_li[k] = 0.0f;
        s_mi[k] = -INFINITY;
    }

    int i = row_tile_idx;
    int row_start = i * Br;
    if (row_start >= T) return;
    int row_end = min(row_start + Br, T);
    int valid_rows = row_end - row_start;

    // Initialize shared memory (half part) to 0 to avoid NaN from padding/garbage
    // This is crucial because WMMA might read padded/invalid regions (e.g. s_Vj columns)
    // and if they contain NaN/Inf, it propagates to the result even if multiplied by 0.
    int total_halfs = Br * D_pad + Bc * D_pad + D * Bc_pad;
    half zero_h = __float2half(0.0f);
    for (int k = tid; k < total_halfs; k += blockDim.x) {
        sram[k] = zero_h;
    }
    __syncthreads();

    // 1. Load Qi
    load_float4_to_half(q_base + row_start * D, s_Qi, valid_rows, D, D, D_pad, tid, blockDim.x);
    
    __syncthreads();

    for (int j = 0; j < Tc; ++j) {
        int col_start = j * Bc;
        int col_end = min(col_start + Bc, T);
        int valid_cols = col_end - col_start;

        if (is_causal && col_start > row_end) continue;

        // 2. Load Kj, Vj
        load_float4_to_half(k_base + col_start * D, s_Kj, valid_cols, D, D, D_pad, tid, blockDim.x);
        load_transpose_float4_to_half(v_base + col_start * D, s_Vj, valid_cols, D, D, Bc_pad, tid, blockDim.x);
        
        __syncthreads();

        // 3. Compute Sij = Qi * Kj^T
        // Sij (32x32). 4 Tiles.
        // Warp 0: (0,0) -> Rows 0-15, Cols 0-15
        // Warp 1: (0,1) -> Rows 0-15, Cols 16-31
        // Warp 2: (1,0) -> Rows 16-31, Cols 0-15
        // Warp 3: (1,1) -> Rows 16-31, Cols 16-31
        
        int warp_row = (warp_id / 2) * 16;
        int warp_col = (warp_id % 2) * 16;
        
        wmma::fill_fragment(c_frag, 0.0f);
        
        // Loop over D (K dimension of GEMM)
        for (int k = 0; k < D; k += WMMA_K) {
            // Load Qi sub-tile
            wmma::load_matrix_sync(a_frag, s_Qi + warp_row * D_pad + k, D_pad);
            // Load Kj sub-tile (as col_major for K^T)
            // s_Kj is (Bc, D_pad) row major.
            // We want K^T (D, Bc). Subtile at (k, warp_col).
            // This is rows k..k+15, cols warp_col..warp_col+15 of K^T.
            // Corresponds to cols k..k+15, rows warp_col..warp_col+15 of K.
            // Address: s_Kj + warp_col * D_pad + k.
            // Layout: col_major tells WMMA to treat memory as column-major B.
            // Matches our row-major K.
            wmma::load_matrix_sync(b_frag, s_Kj + warp_col * D_pad + k, D_pad);
            
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store Sij to Shared Memory for Softmax
        wmma::store_matrix_sync(s_Sij + warp_row * Bc_pad + warp_col, c_frag, Bc_pad, wmma::mem_row_major);
        
        __syncthreads();

        // 4. Apply Mask, Scale, Softmax
        // Standard float ops in Shared Memory
        for (int k = tid; k < valid_rows * valid_cols; k += blockDim.x) {
            int r = k / valid_cols;
            int c = k % valid_cols;
            int global_row = row_start + r;
            int global_col = col_start + c;
            
            float* val_ptr = s_Sij + r * Bc_pad + c;
            if (is_causal && global_col > global_row) {
                *val_ptr = -1e20f;
            } else {
                *val_ptr *= softmax_scale;
            }
        }
        __syncthreads();

    // 4. Apply Mask, Scale, Softmax
        // Standard float ops in Shared Memory
        for (int k = tid; k < valid_rows * valid_cols; k += blockDim.x) {
            int r = k / valid_cols;
            int c = k % valid_cols;
            int global_row = row_start + r;
            int global_col = col_start + c;
            
            float* val_ptr = s_Sij + r * Bc_pad + c;
            if (is_causal && global_col > global_row) {
                *val_ptr = -INFINITY;
            } else {
                *val_ptr *= softmax_scale;
            }
        }
        __syncthreads();

        // Online Softmax
        for (int r = tid; r < valid_rows; r += blockDim.x) {
            float m_prev = s_mi[r];
            float m_curr = -INFINITY;
            for (int c = 0; c < valid_cols; ++c) {
                float val = s_Sij[r * Bc_pad + c];
                if (val > m_curr) m_curr = val;
            }
            float m_new = max(m_prev, m_curr);
            
            if (tid == 0 && r == 0) {
                 // Debug print
                 // printf("Tid 0: j=%d, m_prev=%f, m_curr=%f, m_new=%f, s_Sij[0]=%f\n", j, m_prev, m_curr, m_new, s_Sij[0]);
            }

            float l_curr = 0.0f;
            for (int c = 0; c < valid_cols; ++c) {
                float val = s_Sij[r * Bc_pad + c];
                float p = 0.0f;
                if (m_new > -INFINITY) {
                     p = expf(val - m_new);
                }
                s_Sij[r * Bc_pad + c] = p; // Store P (float)
                l_curr += p;
            }
            // Zero out invalid columns for GEMM 2 (P * V)
            // This is crucial because GEMM 2 sums over columns of P (up to Bc)
            // and garbage in padded columns would corrupt the result.
            for (int c = valid_cols; c < Bc; ++c) {
                s_Sij[r * Bc_pad + c] = 0.0f;
            }
            
            float l_prev = s_li[r];
            float alpha = 0.0f;
            if (m_new > -INFINITY) {
                 alpha = expf(m_prev - m_new);
            }
            float l_new = alpha * l_prev + l_curr;
            
            s_mi[r] = m_new;
            s_li[r] = l_new;
            
            // Store alpha to s_Kj (reuse buffer, cast to float)
            // s_Kj is half*, but no longer needed for this iteration (GEMM 1 done)
            // s_Qi MUST be preserved for next iteration!
            float* s_alpha = reinterpret_cast<float*>(s_Kj);
            s_alpha[r] = alpha; 
        }
        __syncthreads();
        
        // Store Oi fragments to s_Oi (float)
        int row_offset = (warp_id / 2) * 16;
        int col_offset_0 = (warp_id % 2) * 32;     // 0 or 32
        int col_offset_1 = col_offset_0 + 16;      // 16 or 48
        
        wmma::store_matrix_sync(s_Oi + row_offset * D_pad + col_offset_0, oi_frag[0], D_pad, wmma::mem_row_major);
        wmma::store_matrix_sync(s_Oi + row_offset * D_pad + col_offset_1, oi_frag[1], D_pad, wmma::mem_row_major);
        
        __syncthreads();
        
        // Scale s_Oi
        for (int k = tid; k < valid_rows * D; k += blockDim.x) {
            int r = k / D;
            int c = k % D;
            float* s_alpha = reinterpret_cast<float*>(s_Kj);
            float alpha = s_alpha[r];
            s_Oi[r * D_pad + c] *= alpha;
        }
        __syncthreads();
        
        // Load Oi fragments back
        wmma::load_matrix_sync(oi_frag[0], s_Oi + row_offset * D_pad + col_offset_0, D_pad, wmma::mem_row_major);
        wmma::load_matrix_sync(oi_frag[1], s_Oi + row_offset * D_pad + col_offset_1, D_pad, wmma::mem_row_major);
        
        // Convert s_Sij (float) to half for next GEMM
        // In-place conversion is safe because sizeof(half) < sizeof(float)
        half* s_P = reinterpret_cast<half*>(s_Sij);
        for (int k = tid; k < Br * Bc_pad; k += blockDim.x) {
            float val = s_Sij[k];
            s_P[k] = __float2half(val);
        }
        __syncthreads();

        // Compute P * V
        // P in s_Sij (half). V in s_Vj (half, transposed).
        // P (Br, Bc). V (Bc, D).
        // Warp 0: Rows 0-15 (P), Cols 0-15 (V) -> Oi tile (0,0)
        //         Rows 0-15 (P), Cols 16-31 (V) -> Oi tile (0,1)
        
        // Loop over K (Bc dimension of GEMM)
        for (int k = 0; k < Bc; k += WMMA_K) {
            // Load P sub-tile (A)
            // s_Sij is (Br, Bc_pad) row major.
            // A_frag needs (16, 16).
            // Address: s_Sij + row_offset * Bc_pad + k.
            // Wait, s_Sij is float*. Need to cast to half*?
            // Yes, we stored halfs there.
            // half* s_P = reinterpret_cast<half*>(s_Sij); // Moved up
            wmma::load_matrix_sync(a_frag, s_P + row_offset * Bc_pad + k, Bc_pad);
            
            // Load V sub-tile (B)
            // s_Vj is (D_pad, Bc_pad) row major.
            // We need V (Bc, D) sub-tile at (k, col_offset).
            // This is rows k..k+15, cols col_offset..col_offset+15 of V.
            // Corresponds to cols k..k+15, rows col_offset..col_offset+15 of V^T.
            // Address: s_Vj + col_offset * Bc_pad + k.
            // Layout: col_major.
            
            // Tile 0 (col_offset_0)
            wmma::load_matrix_sync(b_frag, s_Vj + col_offset_0 * Bc_pad + k, Bc_pad);
            wmma::mma_sync(oi_frag[0], a_frag, b_frag, oi_frag[0]);
            
            // Tile 1 (col_offset_1)
            wmma::load_matrix_sync(b_frag, s_Vj + col_offset_1 * Bc_pad + k, Bc_pad);
            wmma::mma_sync(oi_frag[1], a_frag, b_frag, oi_frag[1]);
        }
        __syncthreads();
    }
    
    // Finalize Oi
    // Store Oi to s_Oi first
    int row_offset = (warp_id / 2) * 16;
    int col_offset_0 = (warp_id % 2) * 32;     // 0 or 32
    int col_offset_1 = col_offset_0 + 16;      // 16 or 48
    
    wmma::store_matrix_sync(s_Oi + row_offset * D_pad + col_offset_0, oi_frag[0], D_pad, wmma::mem_row_major);
    wmma::store_matrix_sync(s_Oi + row_offset * D_pad + col_offset_1, oi_frag[1], D_pad, wmma::mem_row_major);
    
    __syncthreads();
    
    // Normalize and Store to HBM
    for (int r = tid; r < valid_rows; r += blockDim.x) {
        float li = s_li[r];
        float mi = s_mi[r];
        float inv_li = 1.0f / (li + 1e-6f);
        
        for (int d = 0; d < D; ++d) {
            s_Oi[r * D_pad + d] *= inv_li;
        }
        
        l_base[row_start + r] = mi + logf(li + 1e-6f);
    }
    __syncthreads();
    
    // Store result to global memory
     // s_Oi has stride D_pad. out_ptr has stride D.
     // D is multiple of 4 (64). D_pad is multiple of 4 (72).
     // Vectorized copy row by row.
     
     float* out_ptr = o_base + row_start * D;
     int vec_cols = D / 4;
     
     for (int k = tid; k < valid_rows * vec_cols; k += blockDim.x) {
         int r = k / vec_cols;
         int c_vec = k % vec_cols;
         int c = c_vec * 4;
         
         float4 val = *reinterpret_cast<float4*>(&s_Oi[r * D_pad + c]);
         *reinterpret_cast<float4*>(&out_ptr[r * D + c]) = val;
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
    
    extern __shared__ char smem[];
    float* sram = reinterpret_cast<float*>(smem);
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

// Backward Kernel Helper: Atomic Add Block to Global
__device__ __forceinline__ void atomic_add_block(float* dest, const float* src, int rows, int cols, int dst_stride, int src_stride, int tid, int total_threads) {
    int elements = rows * cols;
    for (int i = tid; i < elements; i += total_threads) {
        int r = i / cols;
        int c = i % cols;
        atomicAdd(&dest[r * dst_stride + c], src[r * src_stride + c]);
    }
}

// Helper for GEMM: C += A^T * B
// A: (K, M) row-major. A^T is (M, K).
// B: (K, N) row-major.
// C: (M, N) row-major.
// M=Bc, N=D, K=Br.
__device__ __forceinline__ void gemm_at_b_accum(const float* A, const float* B, float* C, int M, int N, int K, 
                                                 int stride_A, int stride_B, int stride_C,
                                                 int tid, int total_threads) {
    int elements = M * N;
    for (int i = tid; i < elements; i += total_threads) {
        int r = i / N; // 0..M-1
        int c = i % N; // 0..N-1
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[k, r] * B[k, c]
            sum += A[k * stride_A + r] * B[k * stride_B + c];
        }
        C[r * stride_C + c] += sum;
    }
}

// Helper for GEMM: C += A * B
// A: (M, K) row-major.
// B: (K, N) row-major.
// C: (M, N) row-major.
// M=Br, N=D, K=Bc.
__device__ __forceinline__ void gemm_ab_accum(const float* A, const float* B, float* C, int M, int N, int K, 
                                                 int stride_A, int stride_B, int stride_C,
                                                 int tid, int total_threads) {
    int elements = M * N;
    for (int i = tid; i < elements; i += total_threads) {
        int r = i / N;
        int c = i % N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[r, k] * B[k, c]
            sum += A[r * stride_A + k] * B[k * stride_B + c];
        }
        C[r * stride_C + c] += sum;
    }
}

__global__ void FlashAttentionBackwardKernel(const float *Q, const float *K, const float *V, const float *dO, float *dQ,
                                             float *dK, float *dV, const float *L, int T, int H, int D,
                                             float softmax_scale, bool is_causal) {
    // Grid: (B, H, Tr)
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row_tile_idx = blockIdx.z;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // Check dimensions
    if (D > D_max) return;

    // Offsets
    size_t batch_head_offset = static_cast<size_t>(batch_idx * H + head_idx) * T * D;
    const float *q_base = Q + batch_head_offset;
    const float *k_base = K + batch_head_offset;
    const float *v_base = V + batch_head_offset;
    const float *do_base = dO + batch_head_offset;
    float *dq_base = dQ + batch_head_offset;
    float *dk_base = dK + batch_head_offset;
    float *dv_base = dV + batch_head_offset;
    const float *l_base = L + (batch_idx * H + head_idx) * T;

    // Constants
    constexpr int Br_bw = 32;
    constexpr int Bc_bw = 32;
    // Align strides to 4 for float4
    int D_pad = (D + 3) / 4 * 4; 
    
    // Shared Memory
    extern __shared__ float sram_bw[];
    float* s_Qi = sram_bw;                  // Br * D
    float* s_dOi = s_Qi + Br_bw * D_pad;    // Br * D
    float* s_Kj = s_dOi + Br_bw * D_pad;    // Bc * D
    float* s_Vj = s_Kj + Bc_bw * D_pad;     // Bc * D
    float* s_Sij = s_Vj + Bc_bw * D_pad;    // Br * Bc (Reusable)
    // dKj and dVj accumulators
    float* s_dKj = s_Sij + Br_bw * Bc_bw;   // Bc * D
    float* s_dVj = s_dKj + Bc_bw * D_pad;   // Bc * D
    
    // Current Tile
    int row_start = row_tile_idx * Br_bw;
    if (row_start >= T) return;
    int row_end = min(row_start + Br_bw, T);
    int valid_rows = row_end - row_start;

    // 1. Load Qi, dOi, Li
    load_float4(q_base + row_start * D, s_Qi, valid_rows, D, D, D_pad, tid, total_threads);
    load_float4(do_base + row_start * D, s_dOi, valid_rows, D, D, D_pad, tid, total_threads);
    
    // Load Li into registers? Or shared if accessed frequently. 
    // We access Li[r] in inner loop. Let's keep in register or shared.
    // Optimization: dpsum needs Li.
    
    // Initialize dQi accumulator in Global Memory? No, we compute dQi locally and write once.
    // We can reuse s_Sij or allocate s_dQi. 
    // Let's allocate s_dQi at the end or reuse.
    // Ideally s_dQi needs to be accumulated.
    // s_dQi size Br * D.
    // We can reuse s_dKj buffer for s_dQi? No, we need s_dKj for atomicAdd.
    // Let's add s_dQi to shared mem layout.
    float* s_dQi = s_dVj + Bc_bw * D_pad; // Br * D
    
    for (int k = tid; k < Br_bw * D_pad; k += total_threads) s_dQi[k] = 0.0f;
    
    __syncthreads();

    // 2. Compute dpsum (Delta) = sum_j (P_ij * (dO_i . V_j))
    // We need to iterate over all j to compute dpsum first?
    // Yes, dS depends on dpsum.
    // We can store dpsum[Br] in shared memory.
    float* s_dpsum = s_dQi + Br_bw * D_pad; // Br
    for (int k = tid; k < Br_bw; k += total_threads) s_dpsum[k] = 0.0f;
    
    int Tc = (T + Bc_bw - 1) / Bc_bw;

    // Pass 1: Compute dpsum
    for (int j = 0; j < Tc; ++j) {
        int col_start = j * Bc_bw;
        int col_end = min(col_start + Bc_bw, T);
        int valid_cols = col_end - col_start;
        
        if (is_causal && col_start > row_end) continue;

        // Load Kj, Vj
        // Note: For Forward we need V^T (D, Bc).
        // For Backward Pass 1 dP = dO * V^T. dO(Br, D), V(Bc, D).
        // dO * V^T needs V as (D, Bc) or (Bc, D)^T.
        // gemm_ab_t_float4 computes A * B^T.
        // So we need A=dO (Br, D) and B=V (Bc, D).
        // So we should load V as (Bc, D) row-major.
        
        load_float4(k_base + col_start * D, s_Kj, valid_cols, D, D, D_pad, tid, total_threads);
        load_float4(v_base + col_start * D, s_Vj, valid_cols, D, D, D_pad, tid, total_threads); // V (Bc, D)
        
        __syncthreads();
        
        // Sij = Qi * Kj^T
        // s_Qi (Br, D), s_Kj (Bc, D) -> Sij (Br, Bc)
        gemm_ab_t_float4(s_Qi, s_Kj, s_Sij, valid_rows, valid_cols, D, D_pad, D_pad, Bc_bw, tid, total_threads);
        
        // Compute dP_partial = dOi * Vj^T
        // s_dOi (Br, D), s_Vj (Bc, D) -> dP (Br, Bc)
        // Reusing s_Kj buffer for dP is risky if we needed s_Kj later?
        // We compute Sij first, then we don't need s_Kj for this pass.
        // So reuse s_Kj for dP.
        float* s_dP = s_Kj;
        gemm_ab_t_float4(s_dOi, s_Vj, s_dP, valid_rows, valid_cols, D, D_pad, D_pad, Bc_bw, tid, total_threads);
        
        __syncthreads();
        
        // Accumulate dpsum
        for (int k = tid; k < valid_rows * valid_cols; k += total_threads) {
            int r = k / valid_cols;
            int c = k % valid_cols;
            int global_row = row_start + r;
            int global_col = col_start + c;
            
            float val = s_Sij[r * Bc_bw + c]; // Sij
            if (is_causal && global_col > global_row) {
                // Masked, no contribution
            } else {
                float l_val = l_base[global_row];
                float p = expf(val * softmax_scale - l_val);
                float dp = s_dP[r * Bc_bw + c];
                
                // dpsum[r] += p * dp
                atomicAdd(&s_dpsum[r], p * dp);
            }
        }
        __syncthreads();
    }
    
    // Pass 2: Compute Gradients
    for (int j = 0; j < Tc; ++j) {
        int col_start = j * Bc_bw;
        int col_end = min(col_start + Bc_bw, T);
        int valid_cols = col_end - col_start;
        
        if (is_causal && col_start > row_end) continue;

        // Load Kj, Vj again
        load_float4(k_base + col_start * D, s_Kj, valid_cols, D, D, D_pad, tid, total_threads);
        load_float4(v_base + col_start * D, s_Vj, valid_cols, D, D, D_pad, tid, total_threads);
        
        // Initialize dKj, dVj accumulators to 0
        for (int k = tid; k < Bc_bw * D_pad; k += total_threads) {
            s_dKj[k] = 0.0f;
            s_dVj[k] = 0.0f;
        }
        
        __syncthreads();
        
        // Recompute Sij = Qi * Kj^T
        gemm_ab_t_float4(s_Qi, s_Kj, s_Sij, valid_rows, valid_cols, D, D_pad, D_pad, Bc_bw, tid, total_threads);
        
        __syncthreads();
        
        // Compute dSij
        // dSij = Pij * (dPij - dpsum[i])
        // dPij = dOi . Vj^T
        // We need dPij again. Can we avoid recomputing GEMM?
        // We need to recompute dPij (dOi * Vj^T).
        // Let's reuse s_dVj buffer for dPij temporarily? 
        // No, s_dVj is Bc*D. dPij is Br*Bc. Size matches (32*32=1024 vs 32*64=2048).
        // But s_dVj needs to accumulate results.
        // We can compute dPij element-wise? No.
        // We must recompute GEMM dOi * Vj^T.
        // Where to store it?
        // We have s_Sij (Br, Bc). We can store dSij there.
        // But we need Pij to compute dSij.
        // So:
        // 1. Compute dPij -> Store in temp buffer (e.g. s_dVj).
        // 2. Compute Pij from s_Sij.
        // 3. Compute dSij = Pij * (dPij - dpsum) -> Store in s_Sij.
        // 4. Zero out s_dVj (it was used as temp).
        
        float* s_dP = s_dVj; // Reuse
        gemm_ab_t_float4(s_dOi, s_Vj, s_dP, valid_rows, valid_cols, D, D_pad, D_pad, Bc_bw, tid, total_threads);
        
        __syncthreads();
        
        for (int k = tid; k < valid_rows * valid_cols; k += total_threads) {
            int r = k / valid_cols;
            int c = k % valid_cols;
            int global_row = row_start + r;
            int global_col = col_start + c;
            
            float sij = s_Sij[r * Bc_bw + c];
            float ds = 0.0f;
            float p = 0.0f;
            
            if (!is_causal || global_col <= global_row) {
                float l_val = l_base[global_row];
                p = expf(sij * softmax_scale - l_val);
                float dp = s_dP[r * Bc_bw + c];
                ds = p * (dp - s_dpsum[r]);
                ds *= softmax_scale; // Scale gradient
            }
            
            s_Sij[r * Bc_bw + c] = ds; // Store dSij
            // Also accumulate dVj here?
            // dVj += Pij^T * dOi.
            // Wait, dVj term is Pij^T * dOi.
            // Pij is (Br, Bc). dOi is (Br, D).
            // Pij^T is (Bc, Br).
            // (Bc, Br) * (Br, D) -> (Bc, D).
            // Standard GEMM.
            // But we need Pij, not dSij for dVj!
            // Ah. dV_j = sum_i P_ij * dO_i.
            // So we need to store Pij somewhere or compute contribution immediately.
            // We replaced s_Sij with dSij.
            // We lost Pij.
            // We can compute contribution to dVj BEFORE overwriting s_Sij with dSij.
            
            // Re-plan:
            // 1. Compute dP -> s_dP (s_dVj).
            // 2. Loop k:
            //    Compute Pij.
            //    Accumulate dVj contribution? No, that's a GEMM.
            //    We can't do element-wise GEMM easily.
            //    We need Pij in a matrix form for GEMM.
            
            // We need storage for Pij AND dSij?
            // s_Sij holds Sij.
            // Convert s_Sij -> Pij (in place).
            // Then compute dVj += Pij^T * dOi.
            // Then compute dSij = Pij * (dP - dpsum).
            // We need dP.
            // If we store dP in s_dVj, we can't accumulate dVj there yet.
            // We need another buffer?
            // We have s_dKj (Bc*D) free?
            // Yes, we compute dKj later.
            // So store dP in s_dKj?
            // dKj is (Bc, D). dP is (Br, Bc).
            // Size: 32*64 vs 32*32. Fits.
        }
        __syncthreads();
        
        // Correct Sequence:
        // 1. Compute dP = dOi * Vj^T -> Store in s_dKj (Temp).
        gemm_ab_t_float4(s_dOi, s_Vj, s_dKj, valid_rows, valid_cols, D, D_pad, D_pad, Bc_bw, tid, total_threads);
        __syncthreads();
        
        // 2. Compute Pij (from s_Sij), then dVj contribution, then dSij.
        // Wait, dVj needs Pij^T * dOi.
        // If we do GEMM, we need Pij in memory.
        // So convert s_Sij -> Pij.
        for (int k = tid; k < valid_rows * valid_cols; k += total_threads) {
            int r = k / valid_cols;
            int c = k % valid_cols;
            int global_row = row_start + r;
            int global_col = col_start + c;
            if (!is_causal || global_col <= global_row) {
                float l_val = l_base[global_row];
                s_Sij[r * Bc_bw + c] = expf(s_Sij[r * Bc_bw + c] * softmax_scale - l_val);
            } else {
                s_Sij[r * Bc_bw + c] = 0.0f;
            }
        }
        __syncthreads();
        
        // 3. Accumulate dVj += Pij^T * dOi.
        // Pij (Br, Bc). dOi (Br, D).
        // 3. Accumulate dVj += Pij^T * dOi.
        // A=Pij (Br, Bc), B=dOi (Br, D).
        // gemm_at_b_accum: M=Bc, N=D, K=Br. stride_A=Bc_bw, stride_B=D_pad, stride_C=D_pad.
        gemm_at_b_accum(s_Sij, s_dOi, s_dVj, Bc_bw, D, valid_rows, Bc_bw, D_pad, D_pad, tid, total_threads);
        
        // 4. Compute dSij = Pij * (dP - dpsum).
        // dP is in s_dKj.
        // Pij is in s_Sij.
        // Result dSij -> s_Sij.
        for (int k = tid; k < valid_rows * valid_cols; k += total_threads) {
            int r = k / valid_cols;
            int c = k % valid_cols;
            float p = s_Sij[r * Bc_bw + c];
            float dp = s_dKj[r * Bc_bw + c]; // Read dP
            float ds = p * (dp - s_dpsum[r]) * softmax_scale;
            s_Sij[r * Bc_bw + c] = ds;
        }
        __syncthreads();
        
        // 5. Clear s_dKj (it was holding dP).
        for (int k = tid; k < Bc_bw * D_pad; k += total_threads) s_dKj[k] = 0.0f;
        __syncthreads();
        
        // 6. Accumulate dKj += dSij^T * Qi.
        // dSij (Br, Bc). Qi (Br, D).
        // (Bc, Br) * (Br, D) -> (Bc, D).
        // A=dSij, B=Qi.
        // gemm_at_b_accum: M=Bc, N=D, K=Br. stride_A=Bc_bw, stride_B=D_pad, stride_C=D_pad.
        gemm_at_b_accum(s_Sij, s_Qi, s_dKj, Bc_bw, D, valid_rows, Bc_bw, D_pad, D_pad, tid, total_threads);
        
        // 7. Accumulate dQi += dSij * Kj.
        // dSij (Br, Bc). Kj (Bc, D).
        // (Br, Bc) * (Bc, D) -> (Br, D).
        // A=dSij, B=Kj.
        // gemm_ab_accum: M=Br, N=D, K=Bc. stride_A=Bc_bw, stride_B=D_pad, stride_C=D_pad.
        gemm_ab_accum(s_Sij, s_Kj, s_dQi, valid_rows, D, valid_cols, Bc_bw, D_pad, D_pad, tid, total_threads);
        
        __syncthreads();
        
        // 8. Atomic Add dKj, dVj to Global
        atomic_add_block(dk_base + col_start * D, s_dKj, valid_cols, D, D, D_pad, tid, total_threads);
        atomic_add_block(dv_base + col_start * D, s_dVj, valid_cols, D, D, D_pad, tid, total_threads);
        __syncthreads();
    }
    
    // Store dQi
    store_float4(dq_base + row_start * D, s_dQi, valid_rows, D, D_pad, D, tid, total_threads);
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
    // Threads: 128 (4 Warps) for WMMA Kernel
    dim3 block(128);
    
    // Shared memory size for WMMA Kernel
    // Padding
    constexpr int Pad = 8;
    int D_pad = D + Pad;
    int Bc_pad = Bc + Pad;
    
    // Layout:
    // s_Qi: Br * D_pad (half) = 32 * (64+8) * 2 = 4608 bytes
    // s_Kj: Bc * D_pad (half) = 32 * (64+8) * 2 = 4608 bytes
    // s_Vj: D * Bc_pad (half) = 64 * (32+8) * 2 = 5120 bytes
    // s_Sij: Br * Bc_pad (float) = 32 * (32+8) * 4 = 5120 bytes
    // s_Oi: Br * D_pad (float) = 32 * (64+8) * 4 = 9216 bytes
    // s_li: Br * 4 = 128 bytes
    // s_mi: Br * 4 = 128 bytes
    // Alignment padding between half and float buffers (max 16 bytes)
    
    size_t half_size = (Br * D_pad + Bc * D_pad + D * Bc_pad) * sizeof(half);
    // Align half_size to 16 bytes
    if (half_size % 16 != 0) half_size += (16 - (half_size % 16));
    
    size_t float_size = (Br * Bc_pad + Br * D_pad + Br + Br) * sizeof(float);
    
    size_t sram_size = half_size + float_size;
    
    // Ensure we are on the correct stream if provided (device.stream)
    cudaStream_t stream = 0; // TODO: get from device

    FlashAttentionForwardKernelWMMA<<<grid, block, sram_size, stream>>>(
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

    // Grid: (B, H, Tr)
    int Tr = (T + 32 - 1) / 32;
    dim3 grid(B, H, Tr);
    dim3 block(128);
    
    // Shared Mem Size for Backward
    // float buffers: s_Qi, s_dOi, s_Kj, s_Vj, s_Sij, s_dKj, s_dVj
    // D=64. D_pad=64. (Aligned to 4 float4 = 16 floats? No, just multiple of 4).
    // Let's use D_pad = (D+3)/4*4.
    int D_pad = (D + 3) / 4 * 4; 
    int Br = 32;
    int Bc = 32;
    
    // Size calculation
    // s_Qi: Br * D (we used D not D_pad in load_float4 dest stride?)
    // Kernel uses D for load stride?
    // load_float4(..., s_Qi, ..., D_pad...) 
    // Kernel uses D_pad stride for shared memory.
    
    size_t sram_size = 0;
    sram_size += Br * D_pad * sizeof(float); // s_Qi
    sram_size += Br * D_pad * sizeof(float); // s_dOi
    sram_size += Bc * D_pad * sizeof(float); // s_Kj
    sram_size += Bc * D_pad * sizeof(float); // s_Vj
    sram_size += Br * Bc * sizeof(float);    // s_Sij
    sram_size += Bc * D_pad * sizeof(float); // s_dKj
    sram_size += Bc * D_pad * sizeof(float); // s_dVj
    // s_dQi reuses s_dVj + ... or we allocate it?
    // Kernel: float* s_dQi = s_dVj + Bc_bw * D_pad;
    sram_size += Br * D_pad * sizeof(float); // s_dQi
    // s_dpsum reuses s_dQi + ...
    sram_size += Br * sizeof(float); // s_dpsum
    
    // Enable >48KB Shared Memory
    cudaFuncSetAttribute(FlashAttentionBackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sram_size);
    
    FlashAttentionBackwardKernel<<<grid, block, sram_size>>>(q_ptr, k_ptr, v_ptr, go_ptr, gq_ptr, gk_ptr, gv_ptr, l_ptr, T, H, D,
                                                  softmax_scale, is_causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "CUDA Error in FlashAttentionBackward: " << cudaGetErrorString(err);
    }
}

} // namespace infini_train::kernels::cuda

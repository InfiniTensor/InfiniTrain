#include <cmath>
#include <cstddef>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/kernels/cuda/flash_attention.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

/**
 * FlashAttention Forward Kernel
 *
 * This kernel implements the FlashAttention algorithm for efficient attention computation.
 * It uses tiling and recomputation to reduce memory access and improve performance.
 *
 * Args:
 *   output: Output tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
 *   query: Query tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
 *   key: Key tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]
 *   value: Value tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]
 *   attn_mask: Optional attention mask tensor
 *   logsumexp: Save logsumexp for backward pass
 *   scale: Scaling factor for attention scores
 *   is_causal: Whether to apply causal masking
 *   enable_gqa: Whether to enable grouped-query attention
 *   dropout_p: Dropout probability
 *   dropout_seed: Random seed for dropout (for reproducibility)
 *   batch_size, target_seq_len, src_seq_len: Tensor dimensions
 *   q_heads, kv_heads, head_dim: Attention head dimensions
 *
 * Note: Actual kernel implementation is not included in this version.
 */
template <typename T>
__device__ T warp_reduce_sum(T val){
#pragma unroll//短循环自动展开,省去分支预测,提升效率
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ T myexp(T x) {
    if constexpr(std::is_same<T, __half>::value) {
        float fx = __half2float(x);
        float result = expf(fx);
        return __float2half(result);
    }
    else if constexpr(std::is_same<T, float>::value) {
        return expf(x);  // expf返回float
    }
    else if constexpr(std::is_same<T, double>::value) {
        return exp(x);   // exp返回double
    }
    else{//other types
      return T(0);
    }
}

template <typename T>
__device__ T warp_reduce_max(T val){
#pragma unroll//短循环自动展开,省去分支预测,提升效率
    for(int offset = 16; offset > 0; offset >>= 1){
        T tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > tmp) ? val : tmp;
    }
    return val;
}


/**
 * FlashAttention Forward Kernel
 *
 * This kernel implements the FlashAttention algorithm for efficient attention computation.
 * It uses tiling and recomputation to reduce memory access and improve performance.
 *
 * Args:
 *   output: Output tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
 *   query: Query tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
 *   key: Key tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]
 *   value: Value tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]
 *   attn_mask: Optional attention mask tensor
 *   logsumexp: Save logsumexp for backward pass
 *   scale: Scaling factor for attention scores
 *   is_causal: Whether to apply causal masking
 *   enable_gqa: Whether to enable grouped-query attention
 *   dropout_p: Dropout probability
 *   dropout_seed: Random seed for dropout (for reproducibility)
 *   batch_size, target_seq_len, src_seq_len: Tensor dimensions
 *   q_heads, kv_heads, head_dim: Attention head dimensions
 *
 * Note: Actual kernel implementation is not included in this version.
 */
template <typename T>
__global__ void FlashAttentionForwardKernel(
    T *output, const T *query, const T *key, const T *value, const T *attn_mask,
    float *logsumexp,  // Save logsumexp for backward pass
    float scale, bool is_causal, bool enable_gqa, int64_t dropout_p,
    unsigned long long dropout_seed,  // Use dropout_seed instead of dropout_mask
    int batch_size, int target_seq_len, int src_seq_len,
    int q_heads, int kv_heads, int head_dim) {
    
    int tid_x = threadIdx.x;          // 横向,blockDim.x列 (Bc)
    int tid_y = threadIdx.y;          // 纵向,blockDim.y行 (Br)
    int bid_x = blockIdx.x;           // x方向,总数 = #q_heads
    int bid_y = blockIdx.y;           // y方向,总数 = #batch
    int bid_z = blockIdx.z;           // z方向,总数 = Tr
    const int p = q_heads / kv_heads; // 计算比例系数，GQA支持

    const int Br = blockDim.y;                  // Q纵向每块大小, 32
    const int Bc = blockDim.x;                  // K/V纵向分块大小, 32
    const int Tc = gridDim.z; // 对应原始论文中K/V纵向分块数Tc,其中Bc = 32

    // 定义一系列临时变量
    extern __shared__ char shared_mem[];
    char *ptr = shared_mem;
    
    // 计算中间变量,包括S, P(复用为SP), m_prev, m_new, l_prev, l_new
    double *SP = reinterpret_cast<double *>(ptr); // double SP[Br][Bc]
    ptr += Br * Bc * sizeof(double);
    float *m_prev = reinterpret_cast<float *>(ptr); // float m_prev[Br]
    ptr += Br * sizeof(float);
    float *m_new = reinterpret_cast<float *>(ptr); // float m_new[Br]
    ptr += Br * sizeof(float);
    float *l_prev = reinterpret_cast<float *>(ptr); // float l_prev[Br]
    ptr += Br * sizeof(float);
    float *l_new = reinterpret_cast<float *>(ptr); // float l_new[Br]
    ptr += Br * sizeof(float);

    // 原始数据QKV和计算结果O; 全采用float
    float *Q_sm = reinterpret_cast<float *>(ptr); // float Q_sm[Br][head_dim]
    ptr += Br * head_dim * sizeof(float);
    float *K_T_sm = reinterpret_cast<float *>(ptr); // float K_T_sm[head_dim][Bc]
    ptr += head_dim * Bc * sizeof(float);
    float *V_sm = reinterpret_cast<float *>(ptr); // float V_sm[Bc][head_dim]
    ptr += Bc * head_dim * sizeof(float);
    float *O_sm = reinterpret_cast<float *>(ptr); // float O_sm[Br][head_dim]
    // Note: Removed dropout_sm shared memory allocation

    // 定义访问宏
#define SP_AT(y, x) SP[y * Bc + x]
#define Q_sm_AT(y, x) Q_sm[y * head_dim + x]
#define K_T_sm_AT(y, x) K_T_sm[y * Bc + x]
#define V_sm_AT(y, x) V_sm[y * head_dim + x]
#define O_sm_AT(y, x) O_sm[y * head_dim + x]
    // Note: Removed DROPOUT_SM_AT macro

    /****************************preparation**************************/
    int bound_tid_y = min(Br, target_seq_len - Br * bid_z);

    // preparation-1: load Qi from GM to SM, and reset Oi to 0
    for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
        O_sm_AT(tid_y, idx) = 0.0;
        Q_sm_AT(tid_y, idx) = 0.0;
        if (tid_y < bound_tid_y) {
            Q_sm_AT(tid_y, idx)
                = float(query[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx]);
        }
    }
    __syncthreads();

    // preparation-2: reset m_prev to -INFINITY and l_prev to 0
    if (tid_x == 0) {
        m_prev[tid_y] = -8192.0;
        l_prev[tid_y] = 0.0;
    }
    __syncthreads();

    // Initialize dropout random state with dropout_seed
    curandStatePhilox4_32_10_t state;
    if (dropout_p > 0) {
        unsigned long long seq = bid_y * q_heads * target_seq_len + bid_x * target_seq_len + Br * bid_z + tid_y;
        curand_init(dropout_seed, seq, 0, &state);
    }
/****************************end-of-preparation*************************/

/****************************main-loop**************************/
#pragma unroll 4
    for (int j = 0; j < Tc; ++j) { // 对于每个K/V分块
        bool skip = (is_causal && bid_z < j);
        if (skip) { // early exit, 直接跳过
            __syncthreads();
            continue;
        }

        SP_AT(tid_y, tid_x) = -8192.0;
        __syncthreads();
        int bound_tid_x = min(Bc, src_seq_len - Bc * j);
        bool is_compute = true; // optimization: 分支处理,加速branch-resolving
        if (is_causal) {
            if (bid_z < j) {
                is_compute = false; // 早期退出情况
            } else if (bid_z == j) {
                is_compute = (tid_y >= tid_x); // 对角线以上
            }
        }

        // step-1: load Ki, Vi from GM to SM
#pragma unroll
        for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
            K_T_sm_AT(idx, tid_y) = 0.0;
            V_sm_AT(tid_y, idx) = 0.0;
            if (tid_y < bound_tid_x) {
                K_T_sm_AT(idx, tid_y) = float(
                    K[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
                V_sm_AT(tid_y, idx) = float(
                    V[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
            }
        }
        __syncthreads();

        // step-2: S = Q @ K.T, point-wise
        if (tid_y < bound_tid_y && tid_x < bound_tid_x) {
            float val0 = 0.0;
            if (is_compute) {
#pragma unroll
                for (int k = 0; k < head_dim; ++k) { 
                    val0 += Q_sm_AT(tid_y, k) * K_T_sm_AT(k, tid_x); 
                }
                SP_AT(tid_y, tid_x) = double(val0) * scale;
            }
        }
        __syncthreads();

        // step-3: m_new = max(m_prev, rowMax(S))
        float val1 = float(SP_AT(tid_y, tid_x));
        val1 = warp_reduce_max(val1);
        if (tid_x == 0 && tid_y < bound_tid_y) {
            m_new[tid_y] = (val1 > m_prev[tid_y]) ? val1 : m_prev[tid_y];
        }
        __syncthreads();

        // step-4: P = exp(S - m_new), point-wise
        if (tid_y < bound_tid_y && tid_x < bound_tid_x) {
            if (is_compute) {
                SP_AT(tid_y, tid_x) = myexp<double>(SP_AT(tid_y, tid_x) - double(m_new[tid_y]));
            } else {
                SP_AT(tid_y, tid_x) = 0.0;
            }
        } else {
            SP_AT(tid_y, tid_x) = 0.0;
        }
        __syncthreads();

        // step-5: l_new = exp(m_prev - m_new) * l_prev + rowSum(P)
        float val2 = float(SP_AT(tid_y, tid_x));
        val2 = warp_reduce_sum(val2);
        float exp_result = myexp<float>(m_prev[tid_y] - m_new[tid_y]);
        if (tid_x == 0 && tid_y < bound_tid_y) {
            l_new[tid_y] = exp_result * l_prev[tid_y] + val2;
        }
        __syncthreads();

        // step-5.5: Apply dropout to P (using dropout_seed for reproducibility)
        if (dropout_p > 0 && tid_y < bound_tid_y && tid_x < bound_tid_x) {
            float rand_val = curand_uniform(&state);
            bool keep = rand_val > dropout_p;
            if (keep) {
                SP_AT(tid_y, tid_x) = SP_AT(tid_y, tid_x) / (1.0f - dropout_p);
            } else {
                SP_AT(tid_y, tid_x) = 0.0;
            }
            // Note: Removed dropout_sm mask saving to reduce memory overhead
        }
        __syncthreads();

        // step-6: O = 1/(exp(m_prev - m_new)) * O + P @ V
        if (tid_x < bound_tid_x && tid_y < bound_tid_y) {
            for (int u = tid_x; u < head_dim; u += blockDim.x) {
                float val3 = 0.0;
#pragma unroll
                for (int w = 0; w < Bc; ++w) { 
                    val3 += float(SP_AT(tid_y, w)) * V_sm_AT(w, u);
                }
                O_sm_AT(tid_y, u) = O_sm_AT(tid_y, u) * exp_result + val3;
            }
        }
        __syncthreads();

        // step-7: m_prev <- m_new; l_prev <- l_new
        if (tid_x == 0 && tid_y < bound_tid_y) {
            m_prev[tid_y] = m_new[tid_y];
            l_prev[tid_y] = l_new[tid_y];
        }
        __syncthreads();
    }
/****************************end-of-main-loop**************************/

/****************************post-process****************************/
// O(GM) = O/l_prev, aka O_sm /= l_prev and write Oi from SM to GM
// Also save logsumexp for backward pass
#pragma unroll
    for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
        if (tid_y < bound_tid_y) {
            output[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx]
                = T(O_sm_AT(tid_y, idx) / float(l_prev[tid_y]));
        }
    }
    
    // Save logsumexp for backward pass
    if (tid_x == 0 && tid_y < bound_tid_y) {
        int logsumexp_idx = ((bid_y * q_heads + bid_x) * target_seq_len) + (Br * bid_z + tid_y);
        logsumexp[logsumexp_idx] = m_prev[tid_y] + log(l_prev[tid_y]);
    }
    __syncthreads();
/****************************end-of-post-process****************************/

// 取消访问宏定义
#undef SP_AT
#undef Q_sm_AT
#undef K_T_sm_AT
#undef V_sm_AT
#undef O_sm_AT
}

/**
 * FlashAttention Backward Kernel
 *
 * This kernel implements the backward pass for FlashAttention.
 * It computes gradients for query, key, and value tensors.
 *
 * Args:
 *   grad_query: Gradient for query tensor
 *   grad_key: Gradient for key tensor
 *   grad_value: Gradient for value tensor
 *   query: Query tensor from forward pass
 *   key: Key tensor from forward pass
 *   value: Value tensor from forward pass
 *   output: Output tensor from forward pass
 *   grad_output: Gradient from upstream
 *   logsumexp: Logsumexp tensor from forward pass
 *   dropout_seed: Dropout seed for reproducibility
 *   attn_mask: Optional attention mask tensor
 *   scale: Scaling factor for attention scores
 *   is_causal: Whether causal masking was applied
 *   dropout_p: Dropout probability
 *   enable_gqa: Whether GQA was enabled
 *   batch_size, target_seq_len, src_seq_len: Tensor dimensions
 *   q_heads, kv_heads, head_dim: Attention head dimensions
 */
template <typename T>
__global__ void FlashAttentionBackwardKernel(
    T *grad_query, T *grad_key, T *grad_value,
    const T *query, const T *key, const T *value,
    const T *output, const T *grad_output,
    const float *logsumexp,
    const float *D,  // Precomputed D = rowsum(dO ∘ O)
    unsigned long long dropout_seed,
    const T *attn_mask,
    float scale, bool is_causal, int64_t dropout_p,
    bool enable_gqa,
    int batch_size, int target_seq_len, int src_seq_len,
    int q_heads, int kv_heads, int head_dim) {
    
    // Grid/block dimensions: grid_dims(num_heads_q, batch_size, Tr), block_dim(Bc, Br)
    // where Br corresponds to thread block row index, Bc corresponds to column index
    int tid_x = threadIdx.x;          // 横向, blockDim.x列 (Bc)
    int tid_y = threadIdx.y;          // 纵向, blockDim.y行 (Br)
    int bid_x = blockIdx.x;           // x方向, 总数 = #q_heads
    int bid_y = blockIdx.y;           // y方向, 总数 = #batch
    int bid_z = blockIdx.z;           // z方向, 总数 = Tr
    const int p = q_heads / kv_heads; // GQA比例系数

    const int Br = blockDim.y;                  // Q纵向每块大小, 32
    const int Bc = blockDim.x;                  // K/V纵向分块大小, 32
    const int Tr = gridDim.z; // Q纵向分块数
    const int Tc = (src_seq_len + Bc - 1) / Bc; // K/V纵向分块数

    // Define shared memory
    extern __shared__ char shared_mem[];
    char *ptr = shared_mem;
    
    // D_sm[Br] - D values loaded from HBM
    float *D_sm = reinterpret_cast<float *>(ptr);
    ptr += Br * sizeof(float);
    
    // Q_sm[Br][head_dim], K_T_sm[head_dim][Bc], V_sm[Bc][head_dim]
    float *Q_sm = reinterpret_cast<float *>(ptr);
    ptr += Br * head_dim * sizeof(float);
    float *K_T_sm = reinterpret_cast<float *>(ptr);
    ptr += head_dim * Bc * sizeof(float);
    float *V_sm = reinterpret_cast<float *>(ptr);
    ptr += Bc * head_dim * sizeof(float);
    
    // dO_sm[Br][head_dim], dK_T_sm[head_dim][Bc], dV_sm[Bc][head_dim]
    float *dO_sm = reinterpret_cast<float *>(ptr);
    ptr += Br * head_dim * sizeof(float);
    float *dK_T_sm = reinterpret_cast<float *>(ptr);
    ptr += head_dim * Bc * sizeof(float);
    float *dV_sm = reinterpret_cast<float *>(ptr);
    ptr += Bc * head_dim * sizeof(float);
    
    // S_sm[Br][Bc], P_sm[Br][Bc]
    float *S_sm = reinterpret_cast<float *>(ptr);
    ptr += Br * Bc * sizeof(float);
    float *P_sm = reinterpret_cast<float *>(ptr);
    
    // L_sm[Br] - logsumexp values
    float *L_sm = reinterpret_cast<float *>(ptr);
    
    // dQ_sm[Br][head_dim] - accumulated gradient for Q
    float *dQ_sm = reinterpret_cast<float *>(ptr);
    
    // Define access macros
    #define D_sm_AT(y) D_sm[y]
    #define Q_sm_AT(y, x) Q_sm[y * head_dim + x]
    #define K_T_sm_AT(y, x) K_T_sm[y * Bc + x]
    #define V_sm_AT(y, x) V_sm[y * head_dim + x]
    #define dO_sm_AT(y, x) dO_sm[y * head_dim + x]
    #define dK_T_sm_AT(y, x) dK_T_sm[y * Bc + x]
    #define dV_sm_AT(y, x) dV_sm[y * head_dim + x]
    #define S_sm_AT(y, x) S_sm[y * Bc + x]
    #define P_sm_AT(y, x) P_sm[y * Bc + x]
    #define L_sm_AT(y) L_sm[y]
    #define dQ_sm_AT(y, x) dQ_sm[y * head_dim + x]

    // Initialize dropout random state
    curandStatePhilox4_32_10_t state;
    if (dropout_p > 0) {
        unsigned long long seq = bid_y * q_heads * target_seq_len + bid_x * target_seq_len + Br * bid_z + tid_y;
        curand_init(dropout_seed, seq, 0, &state);
    }

    /****************************Preparation*****************************/
    
    // Initialize dQ_sm to 0 for accumulation
    for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
        dQ_sm_AT(tid_y, idx) = 0.0f;
    }
    __syncthreads();

    // Load D_i from HBM to shared memory
    int q_idx = Br * bid_z + tid_y;  // Global query position within this head
    if (q_idx < target_seq_len) {
        int d_idx = ((bid_y * q_heads + bid_x) * target_seq_len) + q_idx;
        D_sm_AT(tid_y) = D[d_idx];
    } else {
        D_sm_AT(tid_y) = 0.0f;  // Padding for out-of-bounds positions
    }
    __syncthreads();

    // Get D_i from shared memory
    float D_i = D_sm_AT(tid_y);

    /****************************Main Loop - Outer Loop over K/V tiles*****************************/
    for (int j = 0; j < Tc; ++j) {  // For each K/V column tile
        
        // Skip entire tile if causal and this tile is completely to the right
        bool skip_tile = (is_causal && bid_z < j);
        if (skip_tile) {
            __syncthreads();
            continue;
        }

        // Initialize dK_T_sm, dV_sm to 0 for this column tile
        for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
            for (int y = 0; y < Bc; ++y) {
                dK_T_sm_AT(idx, y) = 0.0f;
                dV_sm_AT(y, idx) = 0.0f;
            }
        }
        __syncthreads();

        // Load K_j, V_j from HBM to shared memory
        int bound_tid_x = min(Bc, src_seq_len - Bc * j);
        for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
            K_T_sm_AT(idx, tid_y) = 0.0f;
            V_sm_AT(tid_y, idx) = 0.0f;
            if (tid_y < bound_tid_x) {
                int kv_head_idx = bid_x / p;
                K_T_sm_AT(idx, tid_y) = float(
                    key[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads + kv_head_idx) * head_dim + idx]);
                V_sm_AT(tid_y, idx) = float(
                    value[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads + kv_head_idx) * head_dim + idx]);
            }
        }
        __syncthreads();

        /****************************Inner Loop - Loop over Q tiles*****************************/
        // For this column tile, iterate through all Q row tiles
        for (int i = 0; i < Tr; ++i) {  // For each Q row tile
            
            // Skip this Q tile if causal and it's completely to the left of current K tile
            bool skip_q_tile = (is_causal && i > j);
            if (skip_q_tile) {
                __syncthreads();
                continue;
            }

            // Load Q_i, dO_i, L_i for this row tile
            int q_tile_start = Br * i;
            int q_tile_end = min(q_tile_start + Br, target_seq_len);
            int bound_tid_y = min(Br, q_tile_end - q_tile_start);
            
            int local_q_idx = Br * i + tid_y;
            if (local_q_idx < bound_tid_y) {
                // Load Q_i, dO_i, L_i from HBM
                int global_q_idx = q_tile_start + tid_y;
                int q_tensor_idx = ((bid_y * target_seq_len) + global_q_idx) * q_heads + bid_x;
                
                for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
                    Q_sm_AT(tid_y, idx) = float(query[q_tensor_idx * head_dim + idx]);
                    dO_sm_AT(tid_y, idx) = float(grad_output[q_tensor_idx * head_dim + idx]);
                    L_sm_AT(tid_y) = logsumexp[((bid_y * q_heads + bid_x) * target_seq_len + global_q_idx];
                }
            }
            __syncthreads();

            // Get D_i for this row
            float D_i_row = D_sm_AT(tid_y);

            // Recompute S_ij = Q_i @ K_j^T
            // Apply causal mask at element level if needed
            #pragma unroll
            for (int y = 0; y < Bc; ++y) {
                for (int x = 0; x < Bc; ++x) {
                    float val = 0.0f;
                    if (tid_y < bound_tid_y && tid_x < bound_tid_x) {
                        bool is_compute = true;
                        if (is_causal && i == j) {
                            // On diagonal, only compute upper triangle
                            int global_q_pos = q_tile_start + tid_y;
                            int global_k_pos = Bc * j + x;
                            is_compute = (global_q_pos >= global_k_pos);
                        }
                        
                        if (is_compute) {
                            #pragma unroll
                            for (int k = 0; k < head_dim; ++k) {
                                val += Q_sm_AT(tid_y, k) * K_T_sm_AT(k, y);
                            }
                        }
                    }
                    S_sm_AT(tid_y, x) = val * scale;
                }
            }
            __syncthreads();

            // Recompute P_ij = exp(S_ij - L_i)
            #pragma unroll
            for (int y = 0; y < Bc; ++y) {
                for (int x = 0; x < Bc; ++x) {
                    if (tid_y < bound_tid_y && tid_x < bound_tid_x) {
                        bool is_compute = true;
                        if (is_causal && i == j) {
                            int global_q_pos = q_tile_start + tid_y;
                            int global_k_pos = Bc * j + x;
                            is_compute = (global_q_pos >= global_k_pos);
                        }
                        
                        if (is_compute) {
                            P_sm_AT(tid_y, x) = myexp<float>(S_sm_AT(tid_y, x) - L_sm_AT(tid_y));
                        } else {
                            P_sm_AT(tid_y, x) = 0.0f;
                        }
                    } else {
                        P_sm_AT(tid_y, x) = 0.0f;
                    }
                }
            }
            __syncthreads();

            // Apply dropout to P_ij
            if (dropout_p > 0) {
                #pragma unroll
                for (int y = 0; y < Bc; ++y) {
                    for (int x = 0; x < Bc; ++x) {
                        if (tid_y < bound_tid_y && tid_x < bound_tid_x) {
                            float rand_val = curand_uniform(&state);
                            bool keep = rand_val > dropout_p;
                            if (keep) {
                                P_sm_AT(tid_y, x) = P_sm_AT(tid_y, x) / (1.0f - dropout_p);
                            } else {
                                P_sm_AT(tid_y, x) = 0.0f;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // Compute dV_j += P_ij^T @ dO_i
            if (tid_y < bound_tid_y) {
                #pragma unroll
                for (int x = 0; x < head_dim; x += blockDim.x) {
                    float val = 0.0f;
                    #pragma unroll
                    for (int y = 0; y < Bc; ++y) {
                        val += P_sm_AT(tid_y, y) * dO_sm_AT(tid_y, x);
                    }
                    atomicAdd(&dV_sm_AT(y, x), val);
                }
            }
            __syncthreads();

            // Compute dP_ij = dO_i @ V_j^T
            #pragma unroll
            for (int y = 0; y < Bc; ++y) {
                for (int x = 0; x < Bc; ++x) {
                    float val = 0.0f;
                    if (tid_y < bound_tid_y && tid_x < bound_tid_x) {
                        #pragma unroll
                        for (int k = 0; k < head_dim; ++k) {
                            val += dO_sm_AT(tid_y, k) * V_sm_AT(y, k);
                        }
                    }
                    S_sm_AT(tid_y, x) = val;  // Reuse S_sm as temporary storage for dP_ij
                } else {
                    S_sm_AT(tid_y, x) = 0.0f;
                }
            }
            __syncthreads();

            // Apply dropout to dP_ij
            if (dropout_p > 0) {
                #pragma unroll
                for (int y = 0; y < Bc; ++y) {
                    for (int x = 0; x < Bc; ++x) {
                        if (tid_y < bound_tid_y && tid_x < bound_tid_x) {
                            float rand_val = curand_uniform(&state);
                            bool keep = rand_val > dropout_p;
                            if (keep) {
                                S_sm_AT(tid_y, x) = S_sm_AT(tid_y, x) / (1.0f - dropout_p);
                            } else {
                                S_sm_AT(tid_y, x) = 0.0f;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // Compute dS_ij = P_ij * (dP_ij - D_i)
            #pragma unroll
            for (int y = 0; y < Bc; ++y) {
                for (int x = 0; x < Bc; ++x) {
                    if (tid_y < bound_tid_y && tid_x < bound_tid_x) {
                        S_sm_AT(tid_y, x) = P_sm_AT(tid_y, x) * (S_sm_AT(tid_y, x) - D_i_row);
                    } else {
                        S_sm_AT(tid_y, x) = 0.0f;
                    }
                }
            }
            __syncthreads();

            // Compute dK_j += dS_ij^T @ Q_i
            if (tid_x < head_dim) {
                #pragma unroll
                for (int y = 0; y < Bc; ++y) {
                    float val = 0.0f;
                    #pragma unroll
                    for (int x = 0; x < Br; ++x) {
                        val += S_sm_AT(x, y) * Q_sm_AT(x, tid_x);
                    }
                    atomicAdd(&dK_T_sm_AT(tid_x, y), val * scale);
                }
            }
            __syncthreads();

            // Compute dQ_i += dS_ij @ K_j
            if (tid_y < bound_tid_y) {
                #pragma unroll
                for (int x = 0; x < head_dim; x += blockDim.x) {
                    float val = 0.0f;
                    #pragma unroll
                    for (int y = 0; y < Bc; ++y) {
                        val += S_sm_AT(tid_y, y) * K_T_sm_AT(x, y);
                    }
                    dQ_sm_AT(tid_y, x) += val * scale;
                }
            }
            __syncthreads();
        }  // End of inner loop over Q tiles

        // Write back dK_j, dV_j to HBM
        // Note: For GQA, dK_j and dV_j need to be accumulated across multiple q-head blocks
        int kv_head_idx = bid_x / p;
        int k_tile_start = Bc * j;
        int k_tile_end = min(k_tile_start + Bc, src_seq_len);
        
        for (int y = 0; y < Bc; ++y) {
            int global_k_idx = k_tile_start + y;
            if (global_k_idx < src_seq_len) {
                for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
                    // dK: [batch_size, src_seq_len, kv_heads, head_dim]
                    int k_tensor_idx = ((bid_y * src_seq_len) + global_k_idx) * kv_heads + kv_head_idx;
                    grad_key[k_tensor_idx * head_dim + idx] += T(dK_T_sm_AT(idx, y));
                    
                    // dV: [batch_size, src_seq_len, kv_heads, head_dim]
                    grad_value[k_tensor_idx * head_dim + idx] += T(dV_sm_AT(y, idx));
                }
            }
        }
    }  // End of outer loop over K/V tiles

    // Write back dQ_i to HBM
    int q_tile_start = Br * bid_z;
    int q_tile_end = min(q_tile_start + Br, target_seq_len);
    int bound_tid_y = min(Br, q_tile_end - q_tile_start);
    
    if (tid_y < bound_tid_y) {
        int global_q_idx = q_tile_start + tid_y;
        int q_tensor_idx = ((bid_y * target_seq_len) + global_q_idx) * q_heads + bid_x;
        
        for (int idx = tid_x; idx < head_dim; idx += blockDim.x) {
            // dQ: [batch_size, target_seq_len, q_heads, head_dim]
            grad_query[q_tensor_idx * head_dim + idx] += T(dQ_sm_AT(tid_y, idx));
        }
    }

    // Undefine access macros
    #undef D_sm_AT
    #undef Q_sm_AT
    #undef K_T_sm_AT
    #undef V_sm_AT
    #undef dO_sm_AT
    #undef dK_T_sm_AT
    #undef dV_sm_AT
    #undef S_sm_AT
    #undef P_sm_AT
    #undef L_sm_AT
    #undef dQ_sm_AT
}

/**
 * FlashAttention Forward Function
 *
 * This is the main entry point for FlashAttention forward computation.
 * It creates the output tensor and launches the appropriate kernel based on data type.
 *
 * Args:
 *   query: Query tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
 *   key: Key tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]
 *   value: Value tensor of shape [batch_size, seq_len_k, num_heads_kv, head_dim]
 *   attn_mask: Optional attention mask tensor
 *   scale: Scaling factor for attention scores
 *   is_causal: Whether to apply causal masking
 *   dropout_p: Dropout probability
 *   enable_gqa: Whether to enable grouped-query attention
 *
 * Returns:
 *   Output tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
 *   logsumexp: Logsumexp tensor for backward pass [batch_size, num_heads, seq_len_q]
 *   dropout_seed: Dropout seed for backward pass [1]
 */

template <typename T>
FlashAttentionForwardOutput FlashAttentionForwardImpl(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                                    const std::shared_ptr<Tensor> &value,
                                                    const std::shared_ptr<Tensor> &attn_mask, float scale, bool is_causal,
                                                    int64_t dropout_p, bool enable_gqa) {
    auto dtype = query->Dtype();
    const auto &query_dims = query->Dims();

    // Output shape: [batch_size, seq_len_q, num_heads, head_dim]
    std::vector<int64_t> output_dims = {query_dims[0], query_dims[1], query_dims[2], query_dims[3]};
    auto output = std::make_shared<Tensor>(output_dims, dtype, query->GetDevice());

    // Allocate logsumexp tensor for backward pass
    // Shape: [batch_size, num_heads, seq_len_q]
    std::vector<int64_t> logsumexp_dims = {query_dims[0], query_dims[2], query_dims[1]};
    auto logsumexp = std::make_shared<Tensor>(logsumexp_dims, DataType::kFLOAT32, query->GetDevice());
    float *logsumexp_ptr = static_cast<float *>(logsumexp->DataPtr());

    // Allocate dropout_seed tensor for backward pass
    // Shape: [1]
    unsigned long long dropout_seed = 0;
    std::shared_ptr<Tensor> dropout_seed_tensor;
    if (dropout_p > 0) {
        std::vector<int64_t> dropout_seed_dims = {1};
        dropout_seed_tensor = std::make_shared<Tensor>(dropout_seed_dims, DataType::kUINT64, query->GetDevice());
        dropout_seed = static_cast<unsigned long long>(std::time(nullptr));
        unsigned long long *dropout_seed_ptr = static_cast<unsigned long long *>(dropout_seed_tensor->DataPtr());
        *dropout_seed_ptr = dropout_seed;
    }

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchFlashAttentionForward<float>(output, query, key, value, attn_mask, scale, is_causal,
                                                               dropout_p, enable_gqa, logsumexp_ptr, dropout_seed_ptr);),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchFlashAttentionForward<nv_bfloat16>(output, query, key, value, attn_mask, scale,
                                                                     is_causal, dropout_p, enable_gqa, logsumexp_ptr, dropout_seed_ptr);),
                      DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA FlashAttention forward: 'Unsupported data type'");
    }

    FlashAttentionForwardOutput result;
    result.output = output;
    result.logsumexp = logsumexp;
    result.dropout_seed = dropout_seed_tensor;
    return result;
}

/**
 * Launch FlashAttention Forward Kernel
 *
 * This function sets up grid and block dimensions and launches FlashAttention forward kernel.
 *
 * Args:
 *   output: Output tensor
 *   query: Query tensor
 *   key: Key tensor
 *   value: Value tensor
 *   attn_mask: Optional attention mask tensor
 *   scale: Scaling factor
 *   is_causal: Whether to apply causal masking
 *   dropout_p: Dropout probability
 *   enable_gqa: Whether to enable GQA
 */
template <typename T>
void LaunchFlashAttentionForward(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &query,
                                 const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
                                 const std::shared_ptr<Tensor> &attn_mask, float scale, bool is_causal,
                                 int64_t dropout_p, bool enable_gqa, float *logsumexp_ptr, unsigned long long *dropout_seed_ptr) {

    const auto &query_dims = query->Dims();
    const auto &key_dims = key->Dims();
    const auto &value_dims = value->Dims();

    // Expected shapes:
    // query: [batch_size, seq_len_q, num_heads, head_dim]
    // key: [batch_size, seq_len_k, num_heads_kv, head_dim]
    // value: [batch_size, seq_len_k, num_heads_kv, head_dim]
    // output: [batch_size, seq_len_q, num_heads, head_dim]

    int64_t batch_size = query_dims[0];
    int64_t seq_len_q = query_dims[1];
    int64_t num_heads = query_dims[2];
    int64_t head_dim = query_dims[3];
    int64_t seq_len_k = key_dims[1];
    int64_t num_heads_kv = key_dims[2];

    CHECK_EQ(key_dims[3], head_dim) << "Key head dimension must match query head dimension";
    CHECK_EQ(value_dims[3], head_dim) << "Value head dimension must match query head dimension";
    CHECK_EQ(value_dims[1], seq_len_k) << "Value sequence length must match key sequence length";
    CHECK_EQ(value_dims[2], num_heads_kv) << "Value number of KV heads must match key";

    if (enable_gqa) {
        CHECK(num_heads % num_heads_kv == 0) << "Number of query heads must be divisible by number of KV heads for GQA";
    } else {
        CHECK_EQ(num_heads, num_heads_kv) << "Number of query and KV heads must match for standard attention";
    }

    T *output_ptr = static_cast<T *>(output->DataPtr());
    const T *query_ptr = static_cast<const T *>(query->DataPtr());
    const T *key_ptr = static_cast<const T *>(key->DataPtr());
    const T *value_ptr = static_cast<const T *>(value->DataPtr());
    const T *attn_mask_ptr = attn_mask ? static_cast<const T *>(attn_mask->DataPtr()) : nullptr;

    // Set up grid and block dimensions according to FlashAttention v2
    // block_dim(Br, Bc) where Br = Bc = 32
    // grid_dim(query_heads, batch_size, Tr) where Tr = ceil(seq_len_q / Br)
    constexpr int Br = 32;
    constexpr int Bc = 32;
    int64_t Tr = (seq_len_q + Br - 1) / Br;
    
    dim3 block_dims(Bc, Br);  // (blockDim.x, blockDim.y) = (Bc, Br)
    dim3 grid_dims(num_heads, batch_size, Tr);  // (gridDim.x, gridDim.y, gridDim.z) = (num_heads, batch_size, Tr)

    // Calculate shared memory size (removed dropout_sm allocation)
    // SP[Br][Bc] (double) + m_prev[Br] (float) + m_new[Br] (float) + l_prev[Br] (float) + l_new[Br] (float)
    // + Q_sm[Br][head_dim] (float) + K_T_sm[head_dim][Bc] (float) + V_sm[Bc][head_dim] (float) + O_sm[Br][head_dim] (float)
    size_t shared_mem_size = Br * Bc * sizeof(double)  // SP
                          + 4 * Br * sizeof(float)     // m_prev, m_new, l_prev, l_new
                          + (Br + Bc + Bc + Br) * head_dim) * sizeof(float);  // Q_sm, K_T_sm, V_sm, O_sm
    // Note: Removed dropout_sm[Br][Bc] (bool) allocation

    auto device = output->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    FlashAttentionForwardKernel<T><<<grid_dims, block_dims, shared_mem_size, cuda_stream>>>(
        output_ptr, query_ptr, key_ptr, value_ptr, attn_mask_ptr,
        logsumexp_ptr, scale, is_causal, enable_gqa, dropout_p, dropout_seed,
        batch_size, seq_len_q, seq_len_k, num_heads, num_heads_kv, head_dim);
}

/**
 * Launch FlashAttention Backward Kernel
 *
 * This function sets up grid and block dimensions and launches FlashAttention backward kernel.
 *
 * Args:
 *   grad_query: Gradient tensor for query
 *   grad_key: Gradient tensor for key
 *   grad_value: Gradient tensor for value
 *   query: Query tensor from forward pass
 *   key: Key tensor from forward pass
 *   value: Value tensor from forward pass
 *   output: Output tensor from forward pass
 *   grad_output: Gradient from upstream
 *   logsumexp: Logsumexp tensor from forward pass
 *   dropout_seed: Dropout seed for reproducibility
 *   attn_mask: Optional attention mask tensor
 *   scale: Scaling factor
 *   is_causal: Whether causal masking was applied
 *   dropout_p: Dropout probability
 *   enable_gqa: Whether GQA was enabled
 */
template <typename T>
void LaunchFlashAttentionBackward(const std::shared_ptr<Tensor> &grad_query, const std::shared_ptr<Tensor> &grad_key,
                                 const std::shared_ptr<Tensor> &grad_value, const std::shared_ptr<Tensor> &query,
                                 const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
                                 const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &grad_output,
                                 const std::shared_ptr<Tensor> &logsumexp,
                                 unsigned long long dropout_seed,
                                 const std::shared_ptr<Tensor> &attn_mask,
                                 float scale, bool is_causal, int64_t dropout_p, bool enable_gqa) {

    const auto &query_dims = query->Dims();
    const auto &key_dims = key->Dims();

    // Expected shapes:
    // query: [batch_size, seq_len_q, num_heads, head_dim]
    // key: [batch_size, seq_len_k, num_heads_kv, head_dim]
    // value: [batch_size, seq_len_k, num_heads_kv, head_dim]
    // output: [batch_size, seq_len_q, num_heads, head_dim]

    int64_t batch_size = query_dims[0];
    int64_t seq_len_q = query_dims[1];
    int64_t num_heads = query_dims[2];
    int64_t head_dim = query_dims[3];
    int64_t seq_len_k = key_dims[1];
    int64_t num_heads_kv = key_dims[2];

    CHECK_EQ(key_dims[3], head_dim) << "Key head dimension must match query head dimension";
    CHECK_EQ(value->Dims()[3], head_dim) << "Value head dimension must match query head dimension";
    CHECK_EQ(value->Dims()[1], seq_len_k) << "Value sequence length must match key sequence length";
    CHECK_EQ(value->Dims()[2], num_heads_kv) << "Value number of KV heads must match key";

    if (enable_gqa) {
        CHECK(num_heads % num_heads_kv == 0) << "Number of query heads must be divisible by number of KV heads for GQA";
    } else {
        CHECK_EQ(num_heads, num_heads_kv) << "Number of query and KV heads must match for standard attention";
    }

    // Precompute D = rowsum(dO ∘ O) before main backward loop
    // D shape: [batch_size, seq_len_q, num_heads]
    // dO shape: [batch_size, seq_len_q, num_heads, head_dim]
    // O shape: [batch_size, seq_len_q, num_heads, head_dim]
    auto D = function::Sum(grad_output * output, 3, false);

    T *grad_query_ptr = static_cast<T *>(grad_query->DataPtr());
    T *grad_key_ptr = static_cast<T *>(grad_key->DataPtr());
    T *grad_value_ptr = static_cast<T *>(grad_value->DataPtr());
    const T *query_ptr = static_cast<const T *>(query->DataPtr());
    const T *key_ptr = static_cast<const T *>(key->DataPtr());
    const T *value_ptr = static_cast<const T *>(value->DataPtr());
    const T *output_ptr = static_cast<const T *>(output->DataPtr());
    const T *grad_output_ptr = static_cast<const T *>(grad_output->DataPtr());
    const float *logsumexp_ptr = static_cast<const float *>(logsumexp->DataPtr());
    const float *D_ptr = static_cast<const float *>(D->DataPtr());
    const T *attn_mask_ptr = attn_mask ? static_cast<const T *>(attn_mask->DataPtr()) : nullptr;

    // Set up grid and block dimensions according to FlashAttention v2 backward
    // block_dim(Bc, Br) where Bc = Br = 32
    // grid_dims(num_heads_q, batch_size, Tr) where Tr = ceil(seq_len_q / Br)
    constexpr int Br = 32;
    constexpr int Bc = 32;
    int64_t Tr = (seq_len_q + Br - 1) / Br;

    dim3 block_dims(Bc, Br);  // (blockDim.x, blockDim.y) = (Bc, Br)
    dim3 grid_dims(num_heads, batch_size, Tr);  // (gridDim.x, gridDim.y, gridDim.z) = (num_heads, batch_size, Tr)

    // Calculate shared memory size for backward pass
    // Q_sm[Br][head_dim], K_T_sm[head_dim][Bc], V_sm[Bc][head_dim]
    // dO_sm[Br][head_dim], dK_T_sm[head_dim][Bc], dV_sm[Bc][head_dim]
    // S_sm[Br][Bc], P_sm[Br][Bc]
    // D_sm[Br] - D values loaded from HBM to shared memory
    size_t shared_mem_size = Br * head_dim * sizeof(float)     // Q_sm
                           + head_dim * Bc * sizeof(float)    // K_T_sm
                           + Bc * head_dim * sizeof(float)    // V_sm
                           + Br * head_dim * sizeof(float)     // dO_sm
                           + head_dim * Bc * sizeof(float)    // dK_T_sm
                           + Bc * head_dim * sizeof(float)    // dV_sm
                           + Br * Bc * sizeof(float)          // S_sm or (dP_sm when compute dP_sm = dO_i @ V_j^T \in R^{Br*Bc})
                           + Br * Bc * sizeof(float)         // P_sm or (dS_sm when compute dS_sm = P_sm_ij pointwise multiplied by (dP_sm_ij - D_i) \in R^{Br*Bc})
                           + Br * sizeof(float)        // L_i
                           + Br * sizeof(float);              // D_sm (loaded from HBM)

    auto device = query->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    FlashAttentionBackwardKernel<T><<<grid_dims, block_dims, shared_mem_size, cuda_stream>>>(
        grad_query_ptr, grad_key_ptr, grad_value_ptr,
        query_ptr, key_ptr, value_ptr, output_ptr, grad_output_ptr,
        logsumexp_ptr, D_ptr, dropout_seed, attn_mask_ptr,
        scale, is_causal, dropout_p, enable_gqa,
        batch_size, seq_len_q, seq_len_k, num_heads, num_heads_kv, head_dim);
}

/**
 * FlashAttention Backward Function
 *
 * This is the main entry point for FlashAttention backward computation.
 * It creates gradient tensors and launches the appropriate kernel based on data type.
 *
 * Args:
 *   grad_query: Gradient tensor for query
 *   grad_key: Gradient tensor for key
 *   grad_value: Gradient tensor for value
 *   query: Query tensor from forward pass
 *   key: Key tensor from forward pass
 *   value: Value tensor from forward pass
 *   output: Output tensor from forward pass
 *   grad_output: Gradient from upstream
 *   attn_mask: Optional attention mask tensor
 *   scale: Scaling factor
 *   is_causal: Whether causal masking was applied
 *   dropout_p: Dropout probability (not implemented)
 *   enable_gqa: Whether GQA was enabled
 *
 * Returns:
 *   Tuple of (grad_query, grad_key, grad_value) tensors
 */
template <typename T>
std::vector<std::shared_ptr<Tensor>> FlashAttentionBackwardImpl(const std::shared_ptr<Tensor> &grad_query, const std::shared_ptr<Tensor> &grad_key,
                                                              const std::shared_ptr<Tensor> &grad_value, const std::shared_ptr<Tensor> &query,
                                                              const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
                                                              const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &grad_output,
                                                              const std::shared_ptr<Tensor> &logsumexp,
                                                              const std::shared_ptr<Tensor> &dropout_seed,
                                                              const std::shared_ptr<Tensor> &attn_mask,
                                                              float scale, bool is_causal, int64_t dropout_p, bool enable_gqa) {
    auto dtype = query->Dtype();

    // Create gradient tensors with same shapes as inputs
    auto grad_query = std::make_shared<Tensor>(query->Dims(), dtype, query->GetDevice());
    auto grad_key = std::make_shared<Tensor>(key->Dims(), dtype, key->GetDevice());
    auto grad_value = std::make_shared<Tensor>(value->Dims(), dtype, value->GetDevice());

    // Initialize gradients to zero
    DispatchFunc<INFINI_ALL_TYPES>(dtype, [=]<typename T>() { grad_query->Fill<T>(0); }, "CUDA FlashAttentionBackward");
    DispatchFunc<INFINI_ALL_TYPES>(dtype, [=]<typename T>() { grad_key->Fill<T>(0); }, "CUDA FlashAttentionBackward");
    DispatchFunc<INFINI_ALL_TYPES>(dtype, [=]<typename T>() { grad_value->Fill<T>(0); }, "CUDA FlashAttentionBackward");

    // Get dropout seed value
    unsigned long long dropout_seed_value = 0;
    if (dropout_seed) {
        dropout_seed_value = *static_cast<const unsigned long long *>(dropout_seed->DataPtr());
    }

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchFlashAttentionBackward<float>(grad_query, grad_key, grad_value, query, key, value, output, grad_output,
                                                               logsumexp, dropout_seed_value, attn_mask, scale, is_causal,
                                                               dropout_p, enable_gqa);),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchFlashAttentionBackward<nv_bfloat16>(grad_query, grad_key, grad_value, query, key, value, output, grad_output,
                                                                   logsumexp, dropout_seed_value, attn_mask, scale, is_causal,
                                                                   dropout_p, enable_gqa);),
                      DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA FlashAttention backward: 'Unsupported data type'");
    }

    return {grad_query, grad_key, grad_value};
}

// Non-template wrapper functions for registration
FlashAttentionForwardOutput FlashAttentionForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                               const std::shared_ptr<Tensor> &value,
                                               const std::shared_ptr<Tensor> &attn_mask, float scale, bool is_causal,
                                               int64_t dropout_p, bool enable_gqa) {
    auto dtype = query->Dtype();
    const auto &query_dims = query->Dims();

    // Output shape: [batch_size, seq_len_q, num_heads, head_dim]
    std::vector<int64_t> output_dims = {query_dims[0], query_dims[1], query_dims[2], query_dims[3]};
    auto output = std::make_shared<Tensor>(output_dims, dtype, query->GetDevice());

    // Allocate logsumexp tensor for backward pass
    // Shape: [batch_size, num_heads, seq_len_q]
    std::vector<int64_t> logsumexp_dims = {query_dims[0], query_dims[2], query_dims[1]};
    auto logsumexp = std::make_shared<Tensor>(logsumexp_dims, DataType::kFLOAT32, query->GetDevice());
    float *logsumexp_ptr = static_cast<float *>(logsumexp->DataPtr());

    // Allocate dropout_seed tensor for backward pass
    // Shape: [1]
    unsigned long long dropout_seed = 0;
    std::shared_ptr<Tensor> dropout_seed_tensor;
    if (dropout_p > 0) {
        std::vector<int64_t> dropout_seed_dims = {1};
        dropout_seed_tensor = std::make_shared<Tensor>(dropout_seed_dims, DataType::kUINT64, query->GetDevice());
        dropout_seed = static_cast<unsigned long long>(std::time(nullptr));
        unsigned long long *dropout_seed_ptr = static_cast<unsigned long long *>(dropout_seed_tensor->DataPtr());
        *dropout_seed_ptr = dropout_seed;
    }

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchFlashAttentionForward<float>(output, query, key, value, attn_mask, scale, is_causal,
                                                               dropout_p, enable_gqa, logsumexp_ptr,
                                                               dropout_seed_tensor ? static_cast<unsigned long long*>(dropout_seed_tensor->DataPtr()) : nullptr);),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchFlashAttentionForward<nv_bfloat16>(output, query, key, value, attn_mask, scale,
                                                                     is_causal, dropout_p, enable_gqa, logsumexp_ptr,
                                                                     dropout_seed_tensor ? static_cast<unsigned long long*>(dropout_seed_tensor->DataPtr()) : nullptr);),
                      DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA FlashAttention forward: 'Unsupported data type'");
    }

    FlashAttentionForwardOutput result;
    result.output = output;
    result.logsumexp = logsumexp;
    result.dropout_seed = dropout_seed_tensor;
    return result;
}

std::vector<std::shared_ptr<Tensor>> FlashAttentionBackward(const std::shared_ptr<Tensor> &grad_query, const std::shared_ptr<Tensor> &grad_key,
                                                          const std::shared_ptr<Tensor> &grad_value, const std::shared_ptr<Tensor> &query,
                                                          const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
                                                          const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &grad_output,
                                                          const std::shared_ptr<Tensor> &logsumexp,
                                                          const std::shared_ptr<Tensor> &dropout_seed,
                                                          const std::shared_ptr<Tensor> &attn_mask,
                                                          float scale, bool is_causal, int64_t dropout_p, bool enable_gqa) {
    auto dtype = query->Dtype();

    // Create gradient tensors with same shapes as inputs
    auto grad_query = std::make_shared<Tensor>(query->Dims(), dtype, query->GetDevice());
    auto grad_key = std::make_shared<Tensor>(key->Dims(), dtype, key->GetDevice());
    auto grad_value = std::make_shared<Tensor>(value->Dims(), dtype, value->GetDevice());

    // Initialize gradients to zero
    DispatchFunc<INFINI_ALL_TYPES>(dtype, [=]<typename T>() { grad_query->Fill<T>(0); }, "CUDA FlashAttentionBackward");
    DispatchFunc<INFINI_ALL_TYPES>(dtype, [=]<typename T>() { grad_key->Fill<T>(0); }, "CUDA FlashAttentionBackward");
    DispatchFunc<INFINI_ALL_TYPES>(dtype, [=]<typename T>() { grad_value->Fill<T>(0); }, "CUDA FlashAttentionBackward");

    // Get dropout seed value
    unsigned long long dropout_seed_value = 0;
    if (dropout_seed) {
        dropout_seed_value = *static_cast<const unsigned long long *>(dropout_seed->DataPtr());
    }

    switch (dtype) {
        DISPATCH_CASE(WRAP(LaunchFlashAttentionBackward<float>(grad_query, grad_key, grad_value, query, key, value, output, grad_output,
                                                               logsumexp, dropout_seed_value, attn_mask, scale, is_causal,
                                                               dropout_p, enable_gqa);),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchFlashAttentionBackward<nv_bfloat16>(grad_query, grad_key, grad_value, query, key, value, output, grad_output,
                                                                    logsumexp, dropout_seed_value, attn_mask, scale, is_causal,
                                                                    dropout_p, enable_gqa);),
                      DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA FlashAttention backward: 'Unsupported data type'");
    }

    return {grad_query, grad_key, grad_value};
}

} // namespace infini_train::kernels::cuda

// Register FlashAttention kernels with the dispatcher
#define REGISTER_CUDA_FLASHATTENTION_KERNEL(kernel_name) \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FLASHATTENTION_KERNEL(FlashAttentionForward)
REGISTER_CUDA_FLASHATTENTION_KERNEL(FlashAttentionBackward)

#undef REGISTER_CUDA_FLASHATTENTION_KERNEL

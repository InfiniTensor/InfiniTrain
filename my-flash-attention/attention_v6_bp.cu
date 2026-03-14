#include <cuda_bf16.h>
#include "common.h"
#include <random>
const int MMA_M = 16;
const int MMA_K = 16;
const int MMA_N = 8;

// per tile A is m x k: 16 x 16
// per tile B is n x k: 8 x 16
// per tile C is m x n: 16 x 8
__device__ __forceinline__
void ABT(const int a_height, const int b_height, const int dim, auto &a_rmem, auto &b_rmem, auto &c_rmem ){
    for(int mma_id_a = 0; mma_id_a < a_height / MMA_M; mma_id_a++){
        for(int mma_id_b = 0; mma_id_b < b_height / MMA_N; mma_id_b++){
            for(int mma_id_d = 0; mma_id_d < dim / MMA_K; mma_id_d++){
                mma_m16n8k16(a_rmem[mma_id_a][mma_id_d], b_rmem[mma_id_b][mma_id_d], c_rmem[mma_id_a][mma_id_b]);
            }
        }
    }
}

// per tile A is k x m: 16 x 16
// per tile B is k x n: 16 x 8
// per tile C is m x n: 16 x 8
__device__ __forceinline__
void ATB(const int height, const int a_dim, const int b_dim, auto &a_rmem, auto &b_rmem, auto &c_rmem){
    for(int mma_id_a_dim = 0; mma_id_a_dim < a_dim / MMA_M; mma_id_a_dim++){
        for(int mma_id_b_dim = 0; mma_id_b_dim < b_dim / MMA_N; mma_id_b_dim++){
            for(int mma_id_h = 0; mma_id_h < height / MMA_K; mma_id_h++){
                mma_m16n8k16(a_rmem[mma_id_h][mma_id_a_dim], b_rmem[mma_id_h][mma_id_b_dim], c_rmem[mma_id_a_dim][mma_id_b_dim]);
            }
        }
    }
}

//per tile A is m x k: 16 x 16
//per tile B is k x n: 16 x 8
//per tile C is m x n: 16 x 8
__device__ __forceinline__
void AB(const int m, const int k, const int n, auto &a_rmem, auto &b_rmem, auto &c_rmem){
    for(int mma_id_m = 0; mma_id_m < m / MMA_M; mma_id_m++){
        for(int mma_id_n = 0; mma_id_n < n / MMA_N; mma_id_n++){
            for(int mma_id_k = 0; mma_id_k < k / MMA_K; mma_id_k++){
                mma_m16n8k16(a_rmem[mma_id_m][mma_id_k], b_rmem[mma_id_k][mma_id_n], c_rmem[mma_id_m][mma_id_n]);
            }
        }
    }
}

// get P
// apply causal mask
// apply padding mask
__device__ __forceinline__
void compute_P(const int q_height, const int kv_height, const int dim, auto &q_rmem, auto &kv_rmem, auto &s_rmem, auto &P_smem, auto &L,
               bool is_causal, int q_start, int kv_start, int kv_len, int lane_id, int warp_id, int BLOCK_KV){
                for(int mma_id_q = 0; mma_id_q < q_height / MMA_M; mma_id_q++){
                    for(int mma_id_kv = 0; mma_id_kv < kv_height / MMA_N; mma_id_kv++){
                        for(int mma_id_d = 0; mma_id_d < dim / MMA_K; mma_id_d++){
                            mma_m16n8k16(q_rmem[mma_id_q][mma_id_d], kv_rmem[mma_id_kv][mma_id_d], s_rmem[mma_id_q][mma_id_kv]);
                        }
                        for(int i = 0; i < 4; i++){
                            int q_local_idx = mma_id_q * MMA_M + (lane_id >> 2) + 8 * (i >= 2);
                            int kv_local_idx =  mma_id_kv * MMA_N + (lane_id % 4) * 2 + (i & 0x1);
                            int q_global_idx = q_start + q_local_idx;
                            int kv_global_idx = kv_start + kv_local_idx;
                            bool mask = (kv_global_idx >= kv_len) || (is_causal && kv_global_idx > q_global_idx);
                            s_rmem[mma_id_q][mma_id_kv][i] = mask?0:expf(s_rmem[mma_id_q][mma_id_kv][i] - L[q_local_idx]);

                            // P_smem: [BLOCK_Q, BLOCK_KV], row-major
                            // byte offset = (q_local_idx * BLOCK_KV + warp_id * kv_height + kv_local_idx) * sizeof(nv_bfloat16)
                            uint32_t byte_off = (q_local_idx * BLOCK_KV + warp_id * kv_height + kv_local_idx) * sizeof(nv_bfloat16);
                            uint32_t swz_off  = P_smem + swizzle<128>(byte_off);  // BLOCK_KV=64
                            nv_bfloat16* dst = reinterpret_cast<nv_bfloat16*>(__cvta_shared_to_generic(swz_off));
                            *dst = __float2bfloat16(s_rmem[mma_id_q][mma_id_kv][i]);
                        }
                    }
                }
}

// per tile P: 16 x 16, nv_bfloat16
// per tile dP: 16 x 8, float
// per tile dS: 16 x 16, nv_bfloat16
// D: float
__device__ __forceinline__
void get_S(auto &p_rmem, auto &dp_rmem, auto &d, int height, int width, int lane_id){
    for(int mma_id_h = 0; mma_id_h < height / MMA_M; mma_id_h++){
        for(int mma_id_w = 0; mma_id_w < width / MMA_N; mma_id_w++){
            float* dp_regs = dp_rmem[mma_id_h][mma_id_w]; // 4个float

            nv_bfloat162* this_p_rmem = reinterpret_cast<nv_bfloat162 *>(p_rmem[mma_id_h][mma_id_w / 2]);
            int row_idx1 = mma_id_h * MMA_M + (lane_id >> 2);
            int row_idx2 = row_idx1 + 8;
            this_p_rmem[(mma_id_w % 2) * 2] = this_p_rmem[(mma_id_w % 2) * 2] * __float22bfloat162_rn({dp_regs[0] - d[row_idx1], dp_regs[1] - d[row_idx1]});
            this_p_rmem[(mma_id_w % 2) * 2 + 1] = this_p_rmem[(mma_id_w % 2) * 2 + 1] * __float22bfloat162_rn({dp_regs[2] - d[row_idx2], dp_regs[3] - d[row_idx2]});
        }
    }
}

__device__ __forceinline__
void write_dQ(auto &dq_rmem, auto &d_q, int height, int width, int lane_id, int q_start, int q_len){
    for(int mma_id_q = 0; mma_id_q < height / MMA_M; mma_id_q++){
        for(int mma_id_d = 0; mma_id_d < width / MMA_N; mma_id_d++){
            int q_local_idx = mma_id_q * MMA_M + (lane_id >> 2);
            int q_global_idx = q_local_idx + q_start;
            int d_local_idx = (lane_id % 4) * 2 ;
            if(q_global_idx < q_len){
                float *this_dq = &d_q[q_global_idx * width + mma_id_d * MMA_N + d_local_idx];
                atomicAdd(this_dq,     dq_rmem[mma_id_q][mma_id_d][0]);
                atomicAdd(this_dq + 1, dq_rmem[mma_id_q][mma_id_d][1]);
            }
            if(q_global_idx + 8 < q_len){
                float *this_dq = &d_q[(q_global_idx + 8) * width + mma_id_d * MMA_N + d_local_idx];
                atomicAdd(this_dq,     dq_rmem[mma_id_q][mma_id_d][2]);
                atomicAdd(this_dq + 1, dq_rmem[mma_id_q][mma_id_d][3]);
            }
        }
    }
}

__device__ __forceinline__
void write_dkv(auto &d_kv_rmem, auto &d_kv, int height, int width, int lane_id, int kv_start, int kv_len){
    for(int mma_id_kv = 0; mma_id_kv < height / MMA_M; mma_id_kv++){
        for(int mma_id_d = 0; mma_id_d < width / MMA_N; mma_id_d++){
            int kv_local_idx = mma_id_kv * MMA_M + (lane_id >> 2);
            int kv_global_idx = kv_start + kv_local_idx;
            int d_local_idx = (lane_id % 4) * 2;
            if(kv_global_idx < kv_len){
                float2 vals = {d_kv_rmem[mma_id_kv][mma_id_d][0], d_kv_rmem[mma_id_kv][mma_id_d][1]};
                float2 *this_dkv = reinterpret_cast<float2 *>(&d_kv[kv_global_idx * width + mma_id_d * MMA_N + d_local_idx]);
                *this_dkv = vals;
            }
            if(kv_global_idx + 8 < kv_len){
                float2 vals = {d_kv_rmem[mma_id_kv][mma_id_d][2], d_kv_rmem[mma_id_kv][mma_id_d][3]};
                float2 *this_dkv = reinterpret_cast<float2 *>(&d_kv[(kv_global_idx + 8) * width + mma_id_d * MMA_N + d_local_idx]);
                *this_dkv = vals;
            }
        }
    }
}

// no gqa +  causal
template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__global__ void flash_atten_bakward_1(
const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    const nv_bfloat16 *O,
    const float *L,
    const float *D,
    const float *dO,
    float *dQ,
    float *d_temp_K, // [batch_size, q_head, kv_len, dim]
    float *d_temp_V, // [batch_size, q_head, kv_len, dim]
    int bs,
    int q_head,
    int kv_head,
    int q_len,
    int kv_len,
    int head_dim,
    bool is_causal = false,
    int q_kv_ratio = 1
){
    //basic information
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_kv_blocks = cdiv(kv_len, BLOCK_KV);
    const int bs_id = bid / num_kv_blocks;
    const int batch_id = bs_id / q_head;
    const int q_head_id = bs_id % q_head;
    const int kv_head_id = q_head_id / q_kv_ratio;
    const int kv_block_id = bid % num_kv_blocks;
    const int WARP_KV = BLOCK_KV / NUM_WARPS;
    //当前thread block要处理的初始位置
    Q += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM); // process [q_len, DIM]
    dQ += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM);
    K += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM + kv_block_id * BLOCK_KV * DIM); // process [BLOCK_KV, DIM]
    V += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM + kv_block_id * BLOCK_KV * DIM); // process [BLOCK_KV, DIM]
    O += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM); // process [q_len, DIM]
    dO += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM); // process [q_len, DIM]
    d_temp_K += (batch_id * q_head * kv_len * DIM + q_head_id * kv_len * DIM); 
    d_temp_V += (batch_id * q_head * kv_len * DIM + q_head_id * kv_len * DIM);// process [BLOCK_KV, DIM]
    L += (batch_id * q_head * q_len + q_head_id * q_len); //process [q_len,]
    D += (batch_id * q_head * q_len + q_head_id * q_len);
    // Load K, V HBM -> SRAM [BLOCK_KV, DIM]
    // initialize dK,dV: [BLOCK_KV, DIM]
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t K_smem = __cvta_generic_to_shared(smem);
    const uint32_t Q_smem = K_smem;
    const uint32_t V_smem = Q_smem + max(BLOCK_KV, BLOCK_Q) * DIM * sizeof(nv_bfloat16);
    const uint32_t L_smem = V_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    const uint32_t D_smem = L_smem + BLOCK_Q * sizeof(float);
    const uint32_t dO_smem = D_smem + BLOCK_Q * sizeof(float);
    const uint32_t P_smem = dO_smem + BLOCK_Q * DIM * sizeof(float);
    //for ldmatrix: 计算每个线程要load的行和列，并且要swizzle一下
    uint32_t  K_smem_thread, V_smem_thread;
    {
        const int row_off = warp_id * WARP_KV + lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    const int kv_valid_rows = min(BLOCK_KV, kv_len - kv_block_id * BLOCK_KV);
    if(kv_valid_rows == BLOCK_KV){
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid);
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid);
    }
    else{
        gloabl_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid, kv_valid_rows);
        gloabl_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid, kv_valid_rows);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // K,V,dK, dV: shared -> registers
    uint32_t K_rmem[WARP_KV / MMA_N][DIM / MMA_K][2];
    // TBD: modify V_rmem
    uint32_t V_rmem[WARP_KV / MMA_N][DIM / MMA_K][2];

    // Load K,V registers: shared -> register
    for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_N; mma_id_kv++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
            uint32_t k_addr = K_smem_thread;
            uint32_t v_addr = V_smem_thread;
            k_addr += mma_id_kv * MMA_N * DIM *sizeof(nv_bfloat16);
            k_addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
            v_addr += mma_id_kv * MMA_N * DIM *sizeof(nv_bfloat16);
            v_addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
            ldmatrix_x2(
                K_rmem[mma_id_kv][mma_id_d], k_addr
            );
            ldmatrix_x2(V_rmem[mma_id_kv][mma_id_d], v_addr);
        }
    }
    uint32_t Q_rmem[BLOCK_Q / MMA_M][DIM / MMA_K][4];
    uint32_t dO_right_rmem[BLOCK_Q / MMA_K][DIM / MMA_N][2];
    uint32_t dO_left_rmem[BLOCK_Q / MMA_M][DIM / MMA_K][4];
    // uint32_t P_rmem[BLOCK_Q / MMA_K][WARP_KV / MMA_M][4];
    
    float dK_rmem[WARP_KV / MMA_M][DIM / MMA_N][4] = {};
    float dV_rmem[WARP_KV / MMA_M][DIM / MMA_N][4] = {};
    int kv_start = kv_block_id * BLOCK_KV + WARP_KV * warp_id;

    for(int off_q = 0; off_q < q_len; off_q+= BLOCK_Q){

        //为S，P分配 registers
        float S_rmem[BLOCK_Q / MMA_M][WARP_KV / MMA_N][4] = {};
        float dP_rmem[BLOCK_Q / MMA_M][WARP_KV / MMA_N][4] = {};
        float dQ_rmem[BLOCK_Q / MMA_M][DIM / MMA_N][4] = {};
        // Load Q,dO: [BLOCK_Q, DIM] from HBM -> shared
        int q_valid_rows = min(BLOCK_Q, q_len - off_q);
        if(q_valid_rows == BLOCK_Q){
            global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
            global_to_shared_swizzle_float2bfloat16<BLOCK_Q, DIM, TB_SIZE>(dO_smem, dO, DIM, tid);
        }
        else{
            gloabl_to_shared_swizzle_padded<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, q_valid_rows);
            global_to_shared_swizzle_float2bfloat16_padded<BLOCK_Q, DIM, TB_SIZE>(dO_smem, dO, DIM, tid, q_valid_rows);
        }
        for(int i = tid; i < BLOCK_Q; i += TB_SIZE){
            int idx = i + off_q;
            if(idx < q_len){ 
                // load L
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                :: "r"((uint32_t)(L_smem + i * sizeof(float)))
                , "l"(&L[i])); 
                
                // Load D
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                :: "r"((uint32_t)(D_smem + i * sizeof(float)))
                , "l"(&D[i])  
                );
            }
        }
        
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();
        uint32_t Q_smem_thread,dO_left_smem_thread, dO_right_smem_thread;
    {
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        dO_left_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (dO_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        dO_right_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (dO_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
        // Load Q, dO_rmem_left: shared -> registers
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_M; mma_id_q++){
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
                uint32_t q_addr = Q_smem_thread;
                uint32_t do_left_addr = dO_left_smem_thread;
                q_addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
                q_addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);

                do_left_addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
                do_left_addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], q_addr);
                ldmatrix_x4(dO_left_rmem[mma_id_q][mma_id_d], do_left_addr);
            }
        }
        // Load dO_rmem_right: shared -> registers
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_K; mma_id_q ++){
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d ++){
                uint32_t addr = dO_right_smem_thread;
                    addr += mma_id_q * MMA_K * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
                    ldmatrix_x2(dO_right_rmem[mma_id_q][mma_id_d], addr);
            }
        }

        // 1) get P
        float* L_smem_ptr = reinterpret_cast<float*>(__cvta_shared_to_generic(L_smem));
        compute_P(BLOCK_Q, WARP_KV, DIM, Q_rmem, K_rmem, S_rmem, P_smem, L_smem_ptr, is_causal, off_q, kv_start, kv_len, lane_id, warp_id, BLOCK_KV);
        
        // load P x4 trans
        uint32_t P_rmem[BLOCK_Q / MMA_M][WARP_KV / MMA_K][4];
        uint32_t p_smem_thread;
        {
            const int row_off = lane_id % 16;
            const int col_off = lane_id / 16 * 8;
            p_smem_thread = swizzle<BLOCK_KV * sizeof(nv_bfloat16)> (P_smem + (row_off * BLOCK_KV + warp_id * WARP_KV + col_off) * sizeof(nv_bfloat16));   
        }
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_M; mma_id_q++){
            for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_K; mma_id_kv++){
                uint32_t addr = p_smem_thread;
                addr += mma_id_q * MMA_M * BLOCK_KV * sizeof(nv_bfloat16);
                addr ^= (warp_id * WARP_KV + mma_id_kv * MMA_K) * sizeof(nv_bfloat16);
                ldmatrix_x4_trans(P_rmem[mma_id_q][mma_id_kv],addr);
            }
        } 
        // 2) update dV: dV <- dV + P.T @ dO
        ATB(BLOCK_Q, WARP_KV, DIM, P_rmem, dO_right_rmem, dV_rmem);

        // 3) get dP: dP = dO @ V.T
        ABT(BLOCK_Q, WARP_KV, DIM, dO_left_rmem, V_rmem, dP_rmem);

        // 4) get dS: dS = P * (dP - D)
        // now p is ds
        // load P x4
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_M; mma_id_q++){
            for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_K; mma_id_kv++){
                uint32_t addr = p_smem_thread;
                addr += mma_id_q * MMA_M * BLOCK_KV * sizeof(nv_bfloat16);
                addr ^= (warp_id * WARP_KV + mma_id_kv * MMA_K) * sizeof(nv_bfloat16);
                ldmatrix_x4(P_rmem[mma_id_q][mma_id_kv],addr);
            }
        }
        float* D_ptr = reinterpret_cast<float*>(__cvta_shared_to_generic(D_smem));
        get_S(P_rmem, dP_rmem, D_ptr, BLOCK_Q, WARP_KV, lane_id);

        
        // 5) get dQ: dQ <- dQ + dS @ K
        // Load K rmem trans
        uint32_t new_K_rmem[WARP_KV / MMA_K][DIM / MMA_N][2];
        for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_K; mma_id_kv++){
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                uint32_t k_addr = K_smem_thread;
                k_addr += mma_id_kv * MMA_N * DIM *sizeof(nv_bfloat16);
                k_addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x2_trans(new_K_rmem[mma_id_kv][mma_id_d], k_addr);
            }
        }
        AB(BLOCK_Q, WARP_KV, DIM, P_rmem, new_K_rmem, dQ_rmem);
        // write dS to share memory
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_M; mma_id_q++){
            for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_K; mma_id_kv++){
                nv_bfloat162* regs = reinterpret_cast<nv_bfloat162*>(P_rmem[mma_id_q][mma_id_kv]);
                int row0 = lane_id >> 2;
                int col0 = (lane_id % 4) * 2;
                for(int i = 0; i < 2; i++){
                    for(int j = 0; j < 2; j++){
                        uint32_t byte_off = (mma_id_q * MMA_M + row0 + 8 * i) * BLOCK_KV * sizeof(nv_bfloat16) + (warp_id * WARP_KV + mma_id_kv * MMA_K + col0 + 8 * j) * sizeof(nv_bfloat16);
                        uint32_t swz_off  = P_smem + swizzle<128>(byte_off);  // BLOCK_KV=64
                        nv_bfloat162* dst = reinterpret_cast<nv_bfloat162*>(__cvta_shared_to_generic(swz_off));
                        *dst = regs[i * 2 + j];
                    }
                }

            }
        }

        // 6) get dK: dK <- dK + dS.T @ Q
        // load ds trans, load q trans
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_M; mma_id_q++){
            for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_K; mma_id_kv++){
                uint32_t addr = p_smem_thread;
                addr += mma_id_q * MMA_M * BLOCK_KV * sizeof(nv_bfloat16);
                addr ^= (warp_id * WARP_KV + mma_id_kv * MMA_K) * sizeof(nv_bfloat16);
                ldmatrix_x4_trans(P_rmem[mma_id_q][mma_id_kv],addr);
            }
        }
        uint32_t  Q_rmem_thread_new;
        {
        const int row_off =  lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        Q_rmem_thread_new = swizzle<DIM * sizeof(nv_bfloat16)> (Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        }
        uint32_t Q_new_rmem[BLOCK_Q / MMA_K][DIM / MMA_N][2];
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_K; mma_id_q++){
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                uint32_t addr = Q_rmem_thread_new;
                addr += mma_id_q * MMA_K * DIM * sizeof(nv_bfloat16);
                addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
                ldmatrix_x2_trans(Q_new_rmem[mma_id_q][mma_id_d], addr);
            }
        }
        ATB(BLOCK_Q, BLOCK_KV, DIM, P_rmem, Q_rmem, dK_rmem);

        // 7) atomic add and write to dQ
        write_dQ(dQ_rmem, dQ, BLOCK_Q, DIM, lane_id, off_q, q_len);

        // 8) update address: Q, O, dO, dQ,  L, D
        Q += q_valid_rows * DIM;
        O += q_valid_rows * DIM;
        dO += q_valid_rows * DIM;
        L += q_valid_rows;
        D += q_valid_rows;
    }

    // last: write to dK, dV
    write_dkv(dK_rmem, d_temp_K, WARP_KV, DIM, lane_id, kv_start, kv_len);
    write_dkv(dV_rmem, d_temp_V, WARP_KV, DIM, lane_id, kv_start, kv_len);
}

int main(){
    const int batch_size = 4;
    const int q_head = 8;
    const int kv_head = 8;
    const int q_len = 256;
    const int kv_len = 256;
    const int head_dim = 64;
    bool is_causal = true;

    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    constexpr  int q_N = batch_size * q_head * q_len * head_dim;
    constexpr int kv_N = batch_size * kv_head * kv_len * head_dim;
    constexpr int l_N = batch_size * q_head * q_len;
    std::vector<__nv_bfloat16> q_in(q_N), k_in(kv_N), v_in(kv_N), o_in(q_N);
    std::vector<float> l_in(l_N), d_in(l_N), do_in(q_N), dq_in(q_N), dk_in(kv_N), dv_in(kv_N);

    #pragma unroll
    for(int i = 0; i < q_N; i++){
        q_in[i] = __float2bfloat16(dis(gen));
        o_in[i] = __float2bfloat16(dis(gen));
        do_in[i] = dis(gen);
        dq_in[i] = dis(gen);
    }

    #pragma unroll
    for(int i = 0; i < kv_N; i++){
        k_in[i] = __float2bfloat16(dis(gen));
        v_in[i] = __float2bfloat16(dis(gen));
        dk_in[i] = dis(gen);
        dv_in[i] = dis(gen);
    }

    #pragma unroll
    for(int i = 0; i < l_N; i++){
        l_in[i] = dis(gen);
        d_in[i] = dis(gen);
    }

    // allocate device memory
    nv_bfloat16 *q, *k, *v, *o;
    float *l, *d, *d_o, *d_q, *d_k, *d_v;

    cudaMalloc(&q, q_N * sizeof(nv_bfloat16)); 
    cudaMalloc(&k, kv_N * sizeof(nv_bfloat16));
    cudaMalloc(&v, kv_N * sizeof(nv_bfloat16));
    cudaMalloc(&o, q_N * sizeof(nv_bfloat16));

    cudaMalloc(&l, l_N * sizeof(float));
    cudaMalloc(&d, l_N * sizeof(float));
    cudaMalloc(&d_o, q_N * sizeof(float));
    cudaMalloc(&d_q, q_N * sizeof(float));
    cudaMalloc(&d_k, kv_N * sizeof(float));
    cudaMalloc(&d_v, kv_N * sizeof(float));

    // host -> device
    cudaMemcpy(q, q_in.data(), q_N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(k, k_in.data(), kv_N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(v, v_in.data(), kv_N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(o, o_in.data(), q_N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);

    cudaMemcpy(l, l_in.data(), l_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d, d_in.data(), l_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, do_in.data(), q_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, dq_in.data(), q_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, dk_in.data(), kv_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, dv_in.data(), kv_N * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int KV_BLOCKS = 64;
    constexpr int Q_BLOCKS = 64;
    constexpr int kv_blocks = batch_size * q_head * cdiv(kv_len, KV_BLOCKS);
    constexpr int TB_SIZE = 128;
    constexpr int NUM_WARPS = 4;
    constexpr int DIM = 64;

    constexpr int smem_size = (KV_BLOCKS > Q_BLOCKS ? KV_BLOCKS : Q_BLOCKS) * DIM * sizeof(nv_bfloat16) +
                    KV_BLOCKS * DIM * sizeof(nv_bfloat16) + 2 * Q_BLOCKS * sizeof(float) +
                    Q_BLOCKS * DIM * sizeof(float) + Q_BLOCKS * KV_BLOCKS * sizeof(nv_bfloat16);

    flash_atten_bakward_1<Q_BLOCKS, KV_BLOCKS, DIM, NUM_WARPS><<<kv_blocks, TB_SIZE, smem_size>>>(
        q, k, v, o,
        l, d, 
        d_o, d_q, d_k, d_v,
        batch_size, q_head, kv_head, q_len, kv_len, head_dim,
        is_causal
    );
    cudaStreamSynchronize(0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

}
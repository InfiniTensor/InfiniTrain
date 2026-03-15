#include <cuda_bf16.h>
#include "common.h"
#include <random>
const int MMA_M = 16;
const int MMA_K = 16;
const int MMA_N = 8;
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
void compute_P(const int q_height, const int kv_height, const int dim, auto &q_rmem, auto &kv_rmem, auto &s_rmem, auto &P_smem, auto &L,
               bool is_causal, int q_start, int kv_start, int kv_len, int lane_id, int warp_id, int BLOCK_KV, float scale){
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
                            s_rmem[mma_id_q][mma_id_kv][i] = mask?0:expf(s_rmem[mma_id_q][mma_id_kv][i]*scale - L[q_local_idx]);
                            uint32_t byte_off = (q_local_idx * BLOCK_KV + warp_id * kv_height + kv_local_idx) * sizeof(nv_bfloat16);
                            uint32_t swz_off  = P_smem + swizzle<128>(byte_off);  // BLOCK_KV=64
                            nv_bfloat16* dst = reinterpret_cast<nv_bfloat16*>(__cvta_shared_to_generic(swz_off));
                            *dst = __float2bfloat16(s_rmem[mma_id_q][mma_id_kv][i]);
                        }
                    }
                }
}


__device__ __forceinline__
void write_dQ(auto &dq_rmem, auto &d_q, int height, int width, int lane_id, int q_start, int q_len, float scale){
    for(int mma_id_q = 0; mma_id_q < height / MMA_M; mma_id_q++){
        for(int mma_id_d = 0; mma_id_d < width / MMA_N; mma_id_d++){
            int q_local_idx = mma_id_q * MMA_M + (lane_id >> 2);
            int q_global_idx = q_local_idx + q_start;
            int d_local_idx = (lane_id % 4) * 2 ;
            if(q_global_idx < q_len){
                float *this_dq = &d_q[q_global_idx * width + mma_id_d * MMA_N + d_local_idx];
                atomicAdd(this_dq,     dq_rmem[mma_id_q][mma_id_d][0] * scale);
                atomicAdd(this_dq + 1, dq_rmem[mma_id_q][mma_id_d][1] * scale);
            }
            if(q_global_idx + 8 < q_len){
                float *this_dq = &d_q[(q_global_idx + 8) * width + mma_id_d * MMA_N + d_local_idx];
                atomicAdd(this_dq,     dq_rmem[mma_id_q][mma_id_d][2] * scale);
                atomicAdd(this_dq + 1, dq_rmem[mma_id_q][mma_id_d][3] * scale);
            }
        }
    }
}

__device__ __forceinline__
void write_dkv(auto &d_kv_rmem, auto &d_kv, int height, int width, int lane_id, int kv_start, int kv_len, float scale){
    for(int mma_id_kv = 0; mma_id_kv < height / MMA_M; mma_id_kv++){
        for(int mma_id_d = 0; mma_id_d < width / MMA_N; mma_id_d++){
            int kv_local_idx = mma_id_kv * MMA_M + (lane_id >> 2);
            int kv_global_idx = kv_start + kv_local_idx;
            int d_local_idx = (lane_id % 4) * 2;
            if(kv_global_idx < kv_len){
                float2 vals = {d_kv_rmem[mma_id_kv][mma_id_d][0] * scale, d_kv_rmem[mma_id_kv][mma_id_d][1] * scale};
                float2 *this_dkv = reinterpret_cast<float2 *>(&d_kv[kv_global_idx * width + mma_id_d * MMA_N + d_local_idx]);
                *this_dkv = vals;
            }
            if(kv_global_idx + 8 < kv_len){
                float2 vals = {d_kv_rmem[mma_id_kv][mma_id_d][2] * scale, d_kv_rmem[mma_id_kv][mma_id_d][3] * scale};
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
    float scale,
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
    const uint32_t K_smem  = __cvta_generic_to_shared(smem);
    const uint32_t V_smem  = K_smem  + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    const uint32_t Q_smem  = V_smem  + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    const uint32_t L_smem  = Q_smem  + BLOCK_Q  * DIM * sizeof(nv_bfloat16);
    const uint32_t D_smem  = L_smem  + BLOCK_Q  * sizeof(float);
    const uint32_t dO_smem = D_smem  + BLOCK_Q  * sizeof(float);
    const uint32_t P_smem  = dO_smem + BLOCK_Q  * DIM * sizeof(nv_bfloat16);
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
                    ldmatrix_x2_trans(dO_right_rmem[mma_id_q][mma_id_d], addr);
            }
        }

        // 1) get P
        float* L_smem_ptr = reinterpret_cast<float*>(__cvta_shared_to_generic(L_smem));
        compute_P(BLOCK_Q, WARP_KV, DIM, Q_rmem, K_rmem, S_rmem, P_smem, L_smem_ptr, is_causal, off_q, kv_start, kv_len, lane_id, warp_id, BLOCK_KV, scale);
        
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
                addr ^= (mma_id_kv * MMA_K) * sizeof(nv_bfloat16); // delete warp_id * WARP_KV
                ldmatrix_x4_trans(P_rmem[mma_id_q][mma_id_kv],addr);
                uint32_t tem = P_rmem[mma_id_q][mma_id_kv][1];
                P_rmem[mma_id_q][mma_id_kv][1] = P_rmem[mma_id_q][mma_id_kv][2];
                P_rmem[mma_id_q][mma_id_kv][2] = tem;
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
                addr ^= ( mma_id_kv * MMA_K) * sizeof(nv_bfloat16); // delete warp_id * WARP_KV
                ldmatrix_x4(P_rmem[mma_id_q][mma_id_kv],addr);
            }
        }
        float* D_ptr = reinterpret_cast<float*>(__cvta_shared_to_generic(D_smem));
        get_S(P_rmem, dP_rmem, D_ptr, BLOCK_Q, WARP_KV, lane_id);

        
        // 5) get dQ: dQ <- dQ + dS @ K
        // Load K rmem trans
        {
            const int row_off = lane_id % 16 + warp_id * WARP_KV;
            const int col_off = lane_id / 16 * 8;
            K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        }
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
                addr ^= ( mma_id_kv * MMA_K) * sizeof(nv_bfloat16);  // delete warp_id * WARP_KV
                ldmatrix_x4_trans(P_rmem[mma_id_q][mma_id_kv],addr);
                uint32_t tem = P_rmem[mma_id_q][mma_id_kv][1];
                P_rmem[mma_id_q][mma_id_kv][1] = P_rmem[mma_id_q][mma_id_kv][2];
                P_rmem[mma_id_q][mma_id_kv][2] = tem;
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
        ATB(BLOCK_Q, WARP_KV, DIM, P_rmem, Q_new_rmem, dK_rmem);

        // 7) atomic add and write to dQ
        write_dQ(dQ_rmem, dQ, BLOCK_Q, DIM, lane_id, off_q, q_len, scale);

        // 8) update address: Q, O, dO, dQ,  L, D
        Q += q_valid_rows * DIM;
        O += q_valid_rows * DIM;
        dO += q_valid_rows * DIM;
        L += q_valid_rows;
        D += q_valid_rows;
    }

    // last: write to dK, dV
    write_dkv(dK_rmem, d_temp_K, WARP_KV, DIM, lane_id, kv_start, kv_len, scale);
    write_dkv(dV_rmem, d_temp_V, WARP_KV, DIM, lane_id, kv_start, kv_len, 1.0);
}


// ============ PyTorch Python Binding ============
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_attention_backward(
    torch::Tensor Q,   // [bs, q_head, q_seq, head_dim], bf16
    torch::Tensor K,   // [bs, kv_head, kv_seq, head_dim], bf16
    torch::Tensor V,   // [bs, kv_head, kv_seq, head_dim], bf16
    torch::Tensor O,   // [bs, q_head, q_seq, head_dim], bf16, forward output
    torch::Tensor L,   // [bs, q_head, q_seq], float, logsumexp
    torch::Tensor D,   // [bs, q_head, q_seq], float, sum(dO * O)
    torch::Tensor dO,  // [bs, q_head, q_seq, head_dim], float
    bool is_causal = true
) {
    const int bs = Q.size(0);
    const int q_head = Q.size(1);
    const int q_len = Q.size(2);
    const int head_dim = Q.size(3);
    const int kv_head = K.size(1);
    const int kv_len = K.size(2);

    const int q_kv_ratio = q_head / kv_head;

    // 临时 buffer (每个 kv block 累加)
    auto opts_f32 = Q.options().dtype(torch::kFloat32); // ✅ 继承 Q 的 device (cuda)

    auto dQ     = torch::zeros({bs, q_head, q_len,  head_dim}, opts_f32);
    // ✅ d_temp_K/V 必须在 CUDA 上，用 opts_f32 继承 device
    auto d_temp_K = torch::zeros({bs, q_head, kv_len, head_dim}, opts_f32);
    auto d_temp_V = torch::zeros({bs, q_head, kv_len, head_dim}, opts_f32);


    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int DIM = 64;
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE = NUM_WARPS * 32;

    const int num_kv_blocks = (kv_len + BLOCK_KV - 1) / BLOCK_KV;
    const int num_blocks = bs * q_head * num_kv_blocks;

    constexpr int smem_size =
    BLOCK_KV * DIM * sizeof(nv_bfloat16) +   // K
    BLOCK_KV * DIM * sizeof(nv_bfloat16) +   // V
    BLOCK_Q  * DIM * sizeof(nv_bfloat16) +   // Q
    BLOCK_Q  * sizeof(float) +                // L
    BLOCK_Q  * sizeof(float) +                // D
    BLOCK_Q  * DIM * sizeof(nv_bfloat16) +   // dO  ← bf16
    BLOCK_Q  * BLOCK_KV * sizeof(nv_bfloat16); // P

    auto Q_ptr = reinterpret_cast<nv_bfloat16*>(Q.data_ptr());
    auto K_ptr = reinterpret_cast<nv_bfloat16*>(K.data_ptr());
    auto V_ptr = reinterpret_cast<nv_bfloat16*>(V.data_ptr());
    auto O_ptr = reinterpret_cast<nv_bfloat16*>(O.data_ptr());
    auto L_ptr = L.data_ptr<float>();
    auto D_ptr = D.data_ptr<float>();
    auto dO_ptr = dO.data_ptr<float>();
    auto dQ_ptr = dQ.data_ptr<float>();
    auto d_temp_K_ptr = d_temp_K.data_ptr<float>();
    auto d_temp_V_ptr = d_temp_V.data_ptr<float>();

    cudaStream_t stream = 0;
    const float scale = 1.0f / sqrtf((float)head_dim);
    flash_atten_bakward_1<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS><<<num_blocks, TB_SIZE, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        L_ptr, D_ptr, dO_ptr,
        dQ_ptr, d_temp_K_ptr, d_temp_V_ptr,
        bs, q_head, kv_head, q_len, kv_len, head_dim,
        scale, is_causal, q_kv_ratio
    );

    cudaDeviceSynchronize();

    return std::make_tuple(dQ, d_temp_K, d_temp_V);
}

PYBIND11_MODULE(attention_bp, m) {
    m.def("flash_attention_backward", &flash_attention_backward,
          "Flash Attention Backward");
}
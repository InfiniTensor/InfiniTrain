#include <cuda_bf16.h>
#include <cfloat>
#include "common.h"
// bs和head维度拼接，一个block处理一个BLOCK_Q长度的query
// naive版本，没有causal
template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__global__
void flash_atten_kernel(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    nv_bfloat16 *L_out,  // [bs * q_head * q_len], logsumexp per row
    const float scale,
    int q_len,
    int kv_len,
    int bs,
    int q_head,
    int kv_head,
    int q_kv_ratio = 1
){
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_q_blocks = cdiv(q_len, BLOCK_Q);
    const int bs_id = bid / num_q_blocks;
    const int batch_id = bs_id / q_head;
    const int q_head_id = bs_id % q_head;
    const int kv_head_id = q_head_id / q_kv_ratio;
    const int q_block_id = bid % num_q_blocks;

    //当前thread block要处理的初始位置
    Q += (bs_id * q_len *DIM + q_block_id * BLOCK_Q * DIM);
    K += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM);
    V += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM);
    O += (bs_id * q_len *DIM + q_block_id * BLOCK_Q * DIM);
    L_out += (bs_id * q_len + q_block_id * BLOCK_Q);

    //因为Q只被加载一次，所以Q_smem和K_smem共用一块空间
    // Q_smem: (BLOCK_Q,DIM)
    // K,V smem: (BLOCK_KV,DIM)
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem;
    const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);

    constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

    // mma.m16n8k16
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    // setup registers
    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
    //for ldmatrix: 计算每个线程要load的行和列，并且要swizzle一下
    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        // A tile
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        // B tile
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        // B tile trans
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    // global -> shared
    //最后一个block有可能会产生越界
    const int valid_rows = min(BLOCK_Q, q_len - q_block_id * BLOCK_Q);
    //Load Q [BLOCK_Q, DIM]
    if(valid_rows == BLOCK_Q){
        global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
    }
    else{
        gloabl_to_shared_swizzle_padded<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, valid_rows);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // Q block shared -> registers
    for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q ++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d ++){
            uint32_t addr = Q_smem_thread;
            addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
            addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
            ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }
    }
    __syncthreads();

    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q ++){
            rowmax[mma_id_q][0] = -FLT_MAX;
            rowmax[mma_id_q][1] = -FLT_MAX;
        }
    for(int off_kv = 0; off_kv < kv_len; off_kv += BLOCK_KV){
        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};


        int valid_rows = min(BLOCK_KV, kv_len - off_kv);
        // Load K [BLOCK_KV, DIM], gloabal -> shared
        if(valid_rows == BLOCK_KV){
            global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE> (K_smem, K, DIM, tid);
        }
        else{
            gloabl_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid, valid_rows);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        // K shared -> registers
        for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d ++){
                uint32_t addr = K_smem_thread;
                addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr); 
            }
        }

        // MMA S = Q @ K^T [BLOCK Q, BLOCK KV]
        for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++){
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d ++){
                    mma_m16n8k16(
                        Q_rmem[mma_id_q][mma_id_d],
                        K_rmem[mma_id_kv][mma_id_d],
                        S_rmem[mma_id_q][mma_id_kv]
                    );
                }
            }
        }
        for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q ++){
            //apply softmax scale
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
                for(int reg_id = 0; reg_id < 4; reg_id ++){
                    S_rmem[mma_id_q][mma_id_kv][reg_id] *= scale;
                }
            }

            // rowmax
            float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
                float* regs = S_rmem[mma_id_q][mma_id_kv];
                this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3])); 
            }

            // butterfly reduction within 4 threads, combine results
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

            //new rowmax
            this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]); 

            // rescale for previous O
            float rescale[2];
            rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
            rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d ++){
                O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
            }

            // update new row max
            rowmax[mma_id_q][0] = this_rowmax[0];
            rowmax[mma_id_q][1] = this_rowmax[1];

            // rowsumexp
            float this_rowsumexp[2] = {};
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
                float *regs = S_rmem[mma_id_q][mma_id_kv];
                for(int i = 0; i < 4; i++){
                    regs[i] = __expf(regs[i] - rowmax[mma_id_q][i / 2]);
                }
                this_rowsumexp[0] += regs[0] + regs[1];
                this_rowsumexp[1] += regs[2] + regs[3];

                nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *> (P_rmem[mma_id_q][mma_id_kv / 2]);
                this_P_rmem[(mma_id_kv % 2) *2] = __float22bfloat162_rn({regs[0], regs[1]});
                this_P_rmem[(mma_id_kv % 2) *2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
            }
            // butterfly reduction within 4 threads
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

            // accumulate to total rowsumexp
            rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
            rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
        }
        // Load V [BLOCK_KV, DIM] to share memory
        if(valid_rows == BLOCK_KV){
            global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid);
        }
        else{
            gloabl_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid, valid_rows);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        // shared -> registers
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
            uint32_t addr = V_smem_thread;
            addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);  // row
            addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);  // col
            ldmatrix_x2_trans(V_rmem[mma_id_kv][mma_id_d], addr);
        }

        // MMA O += P @ V [BLOCK_Q, DIM]
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
            mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],
                        V_rmem[mma_id_kv][mma_id_d],
                        O_rmem[mma_id_q][mma_id_d]);

        K += valid_rows * DIM;
        V += valid_rows * DIM;
    }
    
    // write to O
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q ++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d ++){
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
            float *regs = O_rmem[mma_id_q][mma_id_d];
            // const int global_row = q_block_id * BLOCK_Q + warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            regs[0] /= rowsumexp[mma_id_q][0];
            regs[1] /= rowsumexp[mma_id_q][0];
            regs[2] /= rowsumexp[mma_id_q][1];
            regs[3] /= rowsumexp[mma_id_q][1];
            const int local_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int global_row = q_block_id * BLOCK_Q + local_row;

            if(global_row < q_len){
                reinterpret_cast<nv_bfloat162*>(O + local_row * DIM + col)[0] = 
                    __float22bfloat162_rn({regs[0], regs[1]});
            }
            if(global_row + 8 < q_len){
                reinterpret_cast<nv_bfloat162*>(O + (local_row + 8) * DIM + col)[0] =
                    __float22bfloat162_rn({regs[2], regs[3]});
            }
        }
        // write L = rowmax + log(rowsumexp), 每行只需一个线程写
        if ((lane_id & 3) == 0) {
            const int local_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int global_row = q_block_id * BLOCK_Q + local_row;
            if (global_row < q_len)
                L_out[local_row] = __float2bfloat16(
                    rowmax[mma_id_q][0] + logf(rowsumexp[mma_id_q][0]));
            if (global_row + 8 < q_len)
                L_out[local_row + 8] = __float2bfloat16(
                    rowmax[mma_id_q][1] + logf(rowsumexp[mma_id_q][1]));
        }
    }
}

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__global__
void flash_atten_kernel_causal(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    nv_bfloat16 *L_out,
    const float scale,
    int q_len,
    int kv_len,
    int bs,
    int q_head,
    int kv_head,
    int q_kv_ratio = 1
){
    // basic thread information
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    const int WARP_Q = BLOCK_Q / NUM_WARPS;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int num_q_blocks = cdiv(q_len, BLOCK_Q);
    const int bs_id = bid / num_q_blocks; // bs_id :[0, batch_size * q_head )
    const int batch_id = bs_id / q_head;
    const int q_head_id = bs_id % q_head;
    const int kv_head_id = q_head_id / q_kv_ratio;
    const int q_block_id = bid % num_q_blocks;

    // start position of current block that will process
    // each block will take Q: [BLOCK_Q, DIM], K: [kv_len, DIM] and V: [kv_len, DIM] as input
    // output O: [BLOCK_Q, DIM]
    Q += (bs_id * q_len * DIM + q_block_id * BLOCK_Q * DIM);
    K += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM);
    V += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM);
    O += (bs_id * q_len * DIM + q_block_id * BLOCK_Q * DIM);
    L_out += (bs_id * q_len + q_block_id * BLOCK_Q);

    // mma m16n8k16
    const int MMA_M = 16;
    const int MMA_N = 8;
    const int MMA_K = 16;

    // declare share meomory
    // overlap Q_smem with K_smem
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem;
    const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);

    // set up regiters
    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][4];

    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];
    // global -> shared: load Q [BLOCK_Q, DIM]
    int q_valid_rows = min(BLOCK_Q, q_len - BLOCK_Q * q_block_id);
    if(q_valid_rows < BLOCK_Q){
        gloabl_to_shared_swizzle_padded<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, q_valid_rows);
    }
    else{
        global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();
    
    // pre-compute address and swizzling for ldmatrix
    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        // A tile
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        // B tile
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        // B tile trans
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    // shared -> registers
    // WARP_Q 可以被MMA_M整除
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q ++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d ++){
            uint32_t addr = Q_smem_thread;
            addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
            addr ^= mma_id_d * MMA_K *  sizeof(nv_bfloat16);
            ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }
    }
    //需不需要同步呢？
    __syncthreads();
    int causal_offset = 0;
    // kv loop, each iteration, load a block of KV, do online softmax
    int q_start = q_block_id * BLOCK_Q + warp_id * WARP_Q;
    int q_end = q_start + min(WARP_Q, q_len -  q_start);
    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    for(int i = 0; i < WARP_Q / MMA_M; i++){
        rowmax[i][0] = -FLT_MAX;
        rowmax[i][1] = -FLT_MAX;
    }
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
    for(int off_kv = 0; off_kv < kv_len; off_kv += BLOCK_KV){
        // 当前warp的q完全注意不到此时的kv_block, 以及后面的kv_block
        if(off_kv > q_end - 1 + causal_offset){
            break;
        }
        int valid_kv_rows = min(BLOCK_KV, kv_len - off_kv);
        int end_kv = off_kv + valid_kv_rows;

        //global -> shared: load K, V: [BLOCK_KV, DIM]
        if(valid_kv_rows < BLOCK_KV){
            // 如果尾块不够，将会被pad zero
            gloabl_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid, valid_kv_rows);
            gloabl_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid, valid_kv_rows);
        }
        else{
            global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid);
            global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
        //当前warp可以完全注意到此kv block
        if(end_kv - 1 <= q_start + causal_offset){
            //shared -> registers
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d ++){
                    uint32_t addr = K_smem_thread;
                    addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_K *sizeof(nv_bfloat16);
                    ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
                }
            }

            // MMA: S = Q @ K.T [BLOCK_Q, BLOCK_KV] + row max + kv padding mask
            
            for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q ++){
                float this_rowmax[2] = {-FLT_MAX, -FLT_MAX}; //每个BLOCK_KV局部内的rowmax
                for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
                    for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d ++){
                        mma_m16n8k16(
                            Q_rmem[mma_id_q][mma_id_d],
                            K_rmem[mma_id_kv][mma_id_d],
                            S_rmem[mma_id_q][mma_id_kv]
                        );
                    } //这里内存可以优化一下，对于每个mma_id_q, mma_id_kv, 只需要一个S即可
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    for(int i = 0; i < 4; i++){
                        regs[i] *= scale;
                        int row_idx_mma = (lane_id >> 2) + 8 * (i >> 1);
                        int row_idx_global = q_start + mma_id_q * MMA_M + row_idx_mma;
                        int col_idx_mma = (lane_id % 4) * 2 + (i & 0x1);
                        int col_idx_global = off_kv + mma_id_kv * MMA_N + col_idx_mma;
                        // causal mask + padding mask
                        if(col_idx_global > row_idx_global + causal_offset|| col_idx_global >= kv_len){
                            regs[i] = -FLT_MAX;
                        }
                    }
                    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
                }

                //butterfly reduction within 4 threads
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

                // new rowmax
                this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
                this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

                // rescale for previous O
                float rescale[2];
                rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
                rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                    for(int j = 0; j < 4; j++){
                        O_rmem[mma_id_q][mma_id_d][j] *= rescale[j / 2];
                    }
                }

                // save new rowmax
                rowmax[mma_id_q][0] = this_rowmax[0];
                rowmax[mma_id_q][1] = this_rowmax[1];
                
                // rowsumexp
                float this_rowsumexp[2] = {};
                for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
                    regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
                    regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
                    regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);
                    this_rowsumexp[0] += regs[0] + regs[1];
                    this_rowsumexp[1] += regs[2] + regs[3];
                    nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
                    this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
                    this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
                }

                //butterfly reduction within 4 threads
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);
                
                // accumulate to total rowsumexp
                rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
                rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
            }
            
            // V shared -> registers
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++){
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                    uint32_t addr = V_smem_thread;
                    addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
                    ldmatrix_x2_trans(V_rmem[mma_id_kv][mma_id_d], addr);
                }
            }
            // MMA O += P @ V [BLOCK_Q, DIM]
            for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                    for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++){
                        mma_m16n8k16(
                            P_rmem[mma_id_q][mma_id_kv],
                            V_rmem[mma_id_kv][mma_id_d],
                            O_rmem[mma_id_q][mma_id_d]
                        );
                    }
                }
            }
            K += valid_kv_rows * DIM;
            V += valid_kv_rows * DIM;

        }
        //当前warp不能完全注意到此kv block
        else{
            // K, shared -> registers
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv ++){
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d ++){
                    uint32_t addr = K_smem_thread;
                    addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_K *sizeof(nv_bfloat16);
                    ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
                }
            }

            // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
            // padding mask: 对于超出了len_kv的in valid row, 需要mask为-inf
            // causal mask: 对于kv_global_idx > q_global_idx + kv_len - q_len 
            for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
                float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
                for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++){
                    for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
                        mma_m16n8k16(
                            Q_rmem[mma_id_q][mma_id_d],
                            K_rmem[mma_id_kv][mma_id_d],
                            S_rmem[mma_id_q][mma_id_kv]
                        );
                    }
                    float* regs = S_rmem[mma_id_q][mma_id_kv];
                    for(int i = 0; i < 4; i++){
                        regs[i] *= scale;
                        int row_idx_mma = (lane_id >> 2) + 8 * (i >> 1);
                        int row_idx_global =  q_start + mma_id_q * MMA_M + row_idx_mma;
                        int col_idx_mma = (lane_id % 4) * 2 + (i & 0x1);
                        int col_idx_global = off_kv + mma_id_kv * MMA_N + col_idx_mma;

                        // causal mask + padding mask
                        if(col_idx_global > row_idx_global || col_idx_global >= kv_len){
                            regs[i] = -FLT_MAX;
                        }
                    }
                    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));

                }
                // butterfly reduction within 4 threads
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));
                
                //new rowmax
                this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
                this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);
                //rescale previous O
                float rescale[2];
                rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
                rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                    O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                    O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                    O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                    O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
                }
                //save new rowmax
                rowmax[mma_id_q][0] = this_rowmax[0];
                rowmax[mma_id_q][1] = this_rowmax[1];

                //rowsumexp
                float this_rowsumexp[2] = {};
                for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++){
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
                    regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
                    regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
                    regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);
                    this_rowsumexp[0] += (regs[0] + regs[1]);
                    this_rowsumexp[1] += (regs[2] + regs[3]);
                    nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
                    this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
                    this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
                }

                // butterfly reduction within 4 threads
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

                 // accumulate to total rowsumexp
                rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
                rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];

            }
            // V shared -> registers
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++){
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                    uint32_t addr = V_smem_thread;
                    addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
                    ldmatrix_x2_trans(V_rmem[mma_id_kv][mma_id_d], addr);
                }
            }
            // MMA O += P @ V [BLOCK_Q, DIM]
            for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
                    for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++){
                        mma_m16n8k16(
                            P_rmem[mma_id_q][mma_id_kv],
                            V_rmem[mma_id_kv][mma_id_d],
                            O_rmem[mma_id_q][mma_id_d]
                        );
                    }
                }
            }
            K += valid_kv_rows * DIM;
            V += valid_kv_rows * DIM;
        }
    }
    // write to O
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q ++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d ++){
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
            float *regs = O_rmem[mma_id_q][mma_id_d];
            regs[0] /= rowsumexp[mma_id_q][0];
            regs[1] /= rowsumexp[mma_id_q][0];
            regs[2] /= rowsumexp[mma_id_q][1];
            regs[3] /= rowsumexp[mma_id_q][1];
            const int local_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int global_row = q_block_id * BLOCK_Q + local_row;

            if(global_row < q_len){
                reinterpret_cast<nv_bfloat162*>(O + local_row * DIM + col)[0] = 
                    __float22bfloat162_rn({regs[0], regs[1]});
            }
            if(global_row + 8 < q_len){
                reinterpret_cast<nv_bfloat162*>(O + (local_row + 8) * DIM + col)[0] =
                    __float22bfloat162_rn({regs[2], regs[3]});
            }
        }
        // write L = rowmax + log(rowsumexp)
        if ((lane_id & 3) == 0) {
            const int local_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int global_row = q_block_id * BLOCK_Q + local_row;
            if (global_row < q_len)
                L_out[local_row] = __float2bfloat16(
                    rowmax[mma_id_q][0] + logf(rowsumexp[mma_id_q][0]));
            if (global_row + 8 < q_len)
                L_out[local_row + 8] = __float2bfloat16(
                    rowmax[mma_id_q][1] + logf(rowsumexp[mma_id_q][1]));
        }
    }
}

/**
 * @brief flash attention的入口函数，与torch.nn.functional.scaled_dot_product_attention接口保持一致
 * @details TBD
 * @param[in] Q (bs, q_head, q_len, q_head_dim)
 * @param[in] K (bs, kv_head, kv_len, head_dim)
 * @param[in] V (bs, kv_head, kv_len, head_dim)
 * @param[out] O (bs, q_head, q_len, head_dim)
 * @param[in] atten_mask if not nullptr: (bs, q_head, q_len, qv_len)  
 * @note head_dim在gpt2和llama3中均为64, 可以开512个thread, 有512/32=16个warp， BLOCK_Q至少为16 * 16 * 2 = 512
 * @note 每个warp处理32个q_seq, 
 */
void attention_v6(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    nv_bfloat16 *O,
    nv_bfloat16 *L_out,  // [bs * q_head * q_len]
    int bs,
    int q_head,
    int kv_head,
    int q_len,
    int kv_len,
    int head_dim,
    const nv_bfloat16 *atten_mask,
    bool is_causal,
    float dropout_p,
    bool is_gqa,
    cudaStream_t stream
){
    ASSERT_NOT_NULL(Q, K, V, O);
    if(is_causal && atten_mask){
        ERROR("is_causal and atten_mask can't simutanounsly be true");
    }
    // [0, 4, 84, 6] : head_id:4, q_seq: 84, 
    // thread_block: 4 * 2 + 1 = 9
    // 84 - 64 = 20; 第 0 个warp
    const int BLOCK_Q = 64;
    const int BLOCK_KV = 64;
    const int TB_SIZE = 128; // A100 can max support 1024 thread block
    const int WARP_SIZE = 32; // 32 threads per warp
    const int NUM_WARPS = 4; // 128 / 32 = 4
    const int DIM = 64;
    // naive mha
        const int num_blocks = bs * q_head * cdiv(q_len, BLOCK_Q);
        if (head_dim != 64){
            ERROR("current only support head_dim =64");
        }
        const int smem_size = max(BLOCK_Q, BLOCK_KV * 2) * DIM * sizeof(nv_bfloat16);
        float scale = 1.0f / sqrtf((float(DIM)));
        if(is_causal == false){
        cudaFuncSetAttribute(
            flash_atten_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        flash_atten_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>
        <<<num_blocks, TB_SIZE, smem_size, stream>>>(
            Q, K, V, O, L_out, scale, q_len, kv_len, bs, q_head, kv_head, q_head / kv_head
        );
        }
        else{
            cudaFuncSetAttribute(
            flash_atten_kernel_causal<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        flash_atten_kernel_causal<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>
        <<<num_blocks, TB_SIZE, smem_size, stream>>>(
            Q, K, V, O, L_out, scale, q_len, kv_len, bs, q_head, kv_head, q_head / kv_head
        );
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        }

        return;
    

}

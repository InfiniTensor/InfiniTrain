#include <cuda_bf16.h>
#include "common.h"

// D_i = rowsum(dO_i * O_i)
__global__ void compute_D_kernel(
    const nv_bfloat16 *O, const nv_bfloat16 *dO, float *D,
    int q_head, int q_len, int head_dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_head = blockIdx.y; // bs * q_head
    if (row >= q_len) return;

    int base = batch_head * q_len * head_dim;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++)
        sum += __bfloat162float(O[base + row * head_dim + d]) *
               __bfloat162float(dO[base + row * head_dim + d]);
    D[batch_head * q_len + row] = sum;
}

// bf16 -> float conversion kernel
__global__ void bf16_to_float_kernel(const nv_bfloat16 *src, float *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __bfloat162float(src[idx]);
}

// float -> bf16 conversion kernel
__global__ void float_to_bf16_kernel(const float *src, nv_bfloat16 *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

// Helper: write a fragment value to swizzled smem as bf16
// This function is not used in current implementation
__device__ inline
void store_bf16_swizzled_placeholder() {}

// Generic MMA matmul: C += A_smem @ B_smem^T
// A is [M_per_warp=16, K_total] in A_smem with A_stride
// B is [N_tile*8, K_total] in B_smem with B_stride (loaded transposed)
// Result in C_frag[N_tiles][4]
// warp_id selects which 16 rows of A
template <int K_TOTAL, int N_TILES>
__device__ inline
void mma_AB_T(
    float C_frag[N_TILES][4],
    uint32_t A_smem_addr, int A_stride, int A_row_start,
    uint32_t B_smem_addr, int B_stride,
    int lane_id)
{
    for (int n = 0; n < N_TILES; n++) {
        for (int k = 0; k < K_TOTAL / 16; k++) {
            // Load A: 16x16 tile from A_smem
            int ld_row;
            int lid8 = lane_id & 7;
            if (lane_id < 8)       ld_row = A_row_start + lid8;
            else if (lane_id < 16) ld_row = A_row_start + 8 + lid8;
            else if (lane_id < 24) ld_row = A_row_start + (lane_id - 16);
            else                   ld_row = A_row_start + 8 + (lane_id - 24);

            uint32_t A_addr = swizzle<128>(
                A_smem_addr + (ld_row * A_stride + k * 16) * sizeof(nv_bfloat16));
            uint32_t A_frag[4];
            ldmatrix_x4(A_frag, A_addr);

            // Load B: transposed, 8 rows from B_smem
            int b_row = n * 8 + (lane_id % 8);
            uint32_t B_addr = swizzle<128>(
                B_smem_addr + (b_row * B_stride + k * 16) * sizeof(nv_bfloat16));
            uint32_t B_frag[2];
            ldmatrix_x2_trans(B_frag, B_addr);

            mma_m16n8k16(A_frag, B_frag, C_frag[n]);
        }
    }
}

// Store a bf16 value to shared memory via asm (avoids 32->64 bit pointer cast issue)
__device__ inline
void st_shared_bf16(uint32_t addr, nv_bfloat16 val) {
    asm volatile("st.shared.b16 [%0], %1;" :: "r"(addr), "h"(*(uint16_t*)&val));
}

// Write C/D fragment to smem as [ROWS, COLS] with swizzle (row-major)
// Fragment layout: c0->(group_id, tid_in_group*2), c1->(group_id, tid_in_group*2+1),
//                  c2->(group_id+8, tid_in_group*2), c3->(group_id+8, tid_in_group*2+1)
// n_tile gives the column block offset (n_tile * 8)
// warp_id gives the row block offset (warp_id * 16)
__device__ inline
void frag_to_smem_row(
    uint32_t smem_addr, int stride,  // stride in elements
    int warp_id, int lane_id,
    float frag[][4], int n_tiles)
{
    int group_id = lane_id >> 2;
    int tid_in_group = lane_id & 3;
    int r0 = warp_id * 16 + group_id;
    int r1 = r0 + 8;

    for (int n = 0; n < n_tiles; n++) {
        int c0 = n * 8 + tid_in_group * 2;
        int c1 = c0 + 1;

        nv_bfloat16 v0 = __float2bfloat16(frag[n][0]);
        nv_bfloat16 v1 = __float2bfloat16(frag[n][1]);
        nv_bfloat16 v2 = __float2bfloat16(frag[n][2]);
        nv_bfloat16 v3 = __float2bfloat16(frag[n][3]);

        // Compute swizzled addresses
        uint32_t a0 = swizzle<64 * (int)sizeof(nv_bfloat16)>(
            smem_addr + (r0 * stride + c0) * sizeof(nv_bfloat16));
        uint32_t a1 = swizzle<64 * (int)sizeof(nv_bfloat16)>(
            smem_addr + (r0 * stride + c1) * sizeof(nv_bfloat16));
        uint32_t a2 = swizzle<64 * (int)sizeof(nv_bfloat16)>(
            smem_addr + (r1 * stride + c0) * sizeof(nv_bfloat16));
        uint32_t a3 = swizzle<64 * (int)sizeof(nv_bfloat16)>(
            smem_addr + (r1 * stride + c1) * sizeof(nv_bfloat16));

        st_shared_bf16(a0, v0);
        st_shared_bf16(a1, v1);
        st_shared_bf16(a2, v2);
        st_shared_bf16(a3, v3);
    }
}

// Write C/D fragment transposed to smem: store frag[q_row, kv_col] as mat[kv_col, q_row]
__device__ inline
void frag_to_smem_transpose(
    uint32_t smem_addr, int stride,  // stride = BLOCK_Q (columns in transposed layout)
    int warp_id, int lane_id,
    float frag[][4], int n_tiles)
{
    int group_id = lane_id >> 2;
    int tid_in_group = lane_id & 3;
    int q0 = warp_id * 16 + group_id;      // original row
    int q1 = q0 + 8;

    for (int n = 0; n < n_tiles; n++) {
        int kv0 = n * 8 + tid_in_group * 2;  // original col
        int kv1 = kv0 + 1;

        nv_bfloat16 v0 = __float2bfloat16(frag[n][0]);
        nv_bfloat16 v1 = __float2bfloat16(frag[n][1]);
        nv_bfloat16 v2 = __float2bfloat16(frag[n][2]);
        nv_bfloat16 v3 = __float2bfloat16(frag[n][3]);

        // Transposed: [kv, q] layout
        uint32_t a0 = swizzle<64 * (int)sizeof(nv_bfloat16)>(
            smem_addr + (kv0 * stride + q0) * sizeof(nv_bfloat16));
        uint32_t a1 = swizzle<64 * (int)sizeof(nv_bfloat16)>(
            smem_addr + (kv1 * stride + q0) * sizeof(nv_bfloat16));
        uint32_t a2 = swizzle<64 * (int)sizeof(nv_bfloat16)>(
            smem_addr + (kv0 * stride + q1) * sizeof(nv_bfloat16));
        uint32_t a3 = swizzle<64 * (int)sizeof(nv_bfloat16)>(
            smem_addr + (kv1 * stride + q1) * sizeof(nv_bfloat16));

        st_shared_bf16(a0, v0);
        st_shared_bf16(a1, v1);
        st_shared_bf16(a2, v2);
        st_shared_bf16(a3, v3);
    }
}

template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, bool IS_CAUSAL>
__global__ void flash_attention_backward_kernel(
    const nv_bfloat16 *Q,   // [bs, q_head, q_len, dim]
    const nv_bfloat16 *K,   // [bs, kv_head, kv_len, dim]
    const nv_bfloat16 *V,   // [bs, kv_head, kv_len, dim]
    const float *L,          // [bs, q_head, q_len]
    const nv_bfloat16 *dO,  // [bs, q_head, q_len, dim]
    const float *D,          // [bs, q_head, q_len]
    float *dQ_acc,           // [bs, q_head, q_len, dim]
    float *dK_acc,           // [bs, kv_head, kv_len, dim]
    float *dV_acc,           // [bs, kv_head, kv_len, dim]
    int bs, int q_head, int kv_head, int q_len, int kv_len, int head_dim,
    int q_kv_ratio, float scale)
{
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    constexpr int STRIDE = 64; // all matrices are 64-wide

    int num_kv_blocks = cdiv(kv_len, BLOCK_KV);
    int kv_block_idx = blockIdx.x % num_kv_blocks;
    int batch_kv_head = blockIdx.x / num_kv_blocks;
    int batch_idx = batch_kv_head / kv_head;
    int kv_head_idx = batch_kv_head % kv_head;

    int kv_start = kv_block_idx * BLOCK_KV;
    if (kv_start >= kv_len) return;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int group_id = lane_id >> 2;
    int tid_in_group = lane_id & 3;

    // Shared memory: K_s[64x64], V_s[64x64], BufA[64x64], BufB[64x64]
    extern __shared__ char smem[];
    constexpr int BLK_SIZE = 64 * 64 * sizeof(nv_bfloat16); // 8KB

    nv_bfloat16 *smem_bf16 = reinterpret_cast<nv_bfloat16*>(smem);
    const uint32_t K_s_addr = __cvta_generic_to_shared(smem_bf16);
    const uint32_t V_s_addr = __cvta_generic_to_shared(smem_bf16 + BLK_SIZE / sizeof(nv_bfloat16));
    const uint32_t BufA_addr = __cvta_generic_to_shared(smem_bf16 + 2 * BLK_SIZE / sizeof(nv_bfloat16));
    const uint32_t BufB_addr = __cvta_generic_to_shared(smem_bf16 + 3 * BLK_SIZE / sizeof(nv_bfloat16));

    // Load K_j, V_j to smem
    const nv_bfloat16 *K_base = K + ((size_t)batch_idx * kv_head * kv_len + kv_head_idx * kv_len + kv_start) * head_dim;
    const nv_bfloat16 *V_base = V + ((size_t)batch_idx * kv_head * kv_len + kv_head_idx * kv_len + kv_start) * head_dim;

    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_s_addr, K_base, head_dim, tid);
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_s_addr, V_base, head_dim, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // dK, dV accumulators: each warp owns 16 rows of [BLOCK_KV, DIM]
    float dK_frag[8][4] = {};
    float dV_frag[8][4] = {};

    for (int qh_offset = 0; qh_offset < q_kv_ratio; qh_offset++) {
        int q_head_idx = kv_head_idx * q_kv_ratio + qh_offset;
        int num_q_blocks = cdiv(q_len, BLOCK_Q);

        int q_block_start = 0;
        int q_block_end = num_q_blocks;
        if constexpr (IS_CAUSAL) {
            q_block_start = kv_start / BLOCK_Q;
        }

        for (int q_block = q_block_start; q_block < q_block_end; q_block++) {
            int q_start = q_block * BLOCK_Q;
            int q_valid = min(BLOCK_Q, q_len - q_start);

            size_t q_offset = ((size_t)batch_idx * q_head * q_len + q_head_idx * q_len + q_start) * head_dim;
            const nv_bfloat16 *Q_base = Q + q_offset;
            const nv_bfloat16 *dO_base = dO + q_offset;
            const float *L_base = L + (size_t)batch_idx * q_head * q_len + q_head_idx * q_len + q_start;
            const float *D_base = D + (size_t)batch_idx * q_head * q_len + q_head_idx * q_len + q_start;

            // ======== Step 0: Load Q_i -> BufA, dO_i -> BufB ========
            global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(BufA_addr, Q_base, head_dim, tid);
            global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(BufB_addr, dO_base, head_dim, tid);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();

            // ======== Step 1: S = Q @ K^T, then P = exp(S*scale - L) ========
            float S_frag[8][4] = {};
            mma_AB_T<DIM, 8>(S_frag, BufA_addr, STRIDE, warp_id * 16,
                             K_s_addr, STRIDE, lane_id);

            // Scale and compute P
            float P_frag[8][4];
            for (int n = 0; n < 8; n++) {
                int abs_row[4], abs_col[4];
                abs_row[0] = warp_id * 16 + group_id;
                abs_row[1] = abs_row[0];
                abs_row[2] = abs_row[0] + 8;
                abs_row[3] = abs_row[2];
                abs_col[0] = n * 8 + tid_in_group * 2;
                abs_col[1] = abs_col[0] + 1;
                abs_col[2] = abs_col[0];
                abs_col[3] = abs_col[1];

                for (int i = 0; i < 4; i++) {
                    float l_val = (abs_row[i] < q_valid) ? L_base[abs_row[i]] : 0.0f;
                    float s_val = S_frag[n][i] * scale - l_val;

                    bool masked = (abs_row[i] >= q_valid) || (abs_col[i] >= (min(BLOCK_KV, kv_len - kv_start)));
                    if constexpr (IS_CAUSAL) {
                        if ((q_start + abs_row[i]) < (kv_start + abs_col[i])) masked = true;
                    }
                    P_frag[n][i] = masked ? 0.0f : expf(s_val);
                }
            }

            // ======== Step 2: dP = dO @ V^T ========
            float dP_frag[8][4] = {};
            mma_AB_T<DIM, 8>(dP_frag, BufB_addr, STRIDE, warp_id * 16,
                             V_s_addr, STRIDE, lane_id);

            // ======== Step 3: dS = P * (dP - D) ========
            float dS_frag[8][4];
            for (int n = 0; n < 8; n++) {
                int r0 = warp_id * 16 + group_id;
                int r1 = r0 + 8;
                float d0 = (r0 < q_valid) ? D_base[r0] : 0.0f;
                float d1 = (r1 < q_valid) ? D_base[r1] : 0.0f;

                dS_frag[n][0] = P_frag[n][0] * (dP_frag[n][0] - d0);
                dS_frag[n][1] = P_frag[n][1] * (dP_frag[n][1] - d0);
                dS_frag[n][2] = P_frag[n][2] * (dP_frag[n][2] - d1);
                dS_frag[n][3] = P_frag[n][3] * (dP_frag[n][3] - d1);

                // Apply scale to dS
                dS_frag[n][0] *= scale;
                dS_frag[n][1] *= scale;
                dS_frag[n][2] *= scale;
                dS_frag[n][3] *= scale;
            }

            // ======== Step 4: Write P^T to BufA, compute dV += P^T @ dO ========
            // BufA now becomes P^T[BLOCK_KV, BLOCK_Q], BufB still has dO
            frag_to_smem_transpose(BufA_addr, STRIDE, warp_id, lane_id, P_frag, 8);
            __syncthreads();

            // dV += P^T_s @ dO_s^T  (but dO is in BufB with stride=DIM)
            mma_AB_T<BLOCK_Q, 8>(dV_frag, BufA_addr, STRIDE, warp_id * 16,
                                 BufB_addr, STRIDE, lane_id);

            // ======== Step 5: Write dS^T to BufB, reload Q to BufA, compute dK ========
            frag_to_smem_transpose(BufB_addr, STRIDE, warp_id, lane_id, dS_frag, 8);
            // Reload Q_i to BufA
            global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(BufA_addr, Q_base, head_dim, tid);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();

            // dK += dS^T_s @ Q_s^T (Q in BufA with stride=DIM)
            mma_AB_T<BLOCK_Q, 8>(dK_frag, BufB_addr, STRIDE, warp_id * 16,
                                 BufA_addr, STRIDE, lane_id);

            // ======== Step 6: Write dS to BufA, compute dQ += dS @ K ========
            frag_to_smem_row(BufA_addr, STRIDE, warp_id, lane_id, dS_frag, 8);
            __syncthreads();

            float dQ_frag[8][4] = {};
            mma_AB_T<BLOCK_KV, 8>(dQ_frag, BufA_addr, STRIDE, warp_id * 16,
                                  K_s_addr, STRIDE, lane_id);

            // Atomic write dQ to global
            size_t dq_base = ((size_t)batch_idx * q_head * q_len + q_head_idx * q_len + q_start) * head_dim;
            for (int n = 0; n < 8; n++) {
                int c0 = n * 8 + tid_in_group * 2;
                int c1 = c0 + 1;
                int r0 = warp_id * 16 + group_id;
                int r1 = r0 + 8;
                if (r0 < q_valid) {
                    atomicAdd(&dQ_acc[dq_base + r0 * head_dim + c0], dQ_frag[n][0]);
                    atomicAdd(&dQ_acc[dq_base + r0 * head_dim + c1], dQ_frag[n][1]);
                }
                if (r1 < q_valid) {
                    atomicAdd(&dQ_acc[dq_base + r1 * head_dim + c0], dQ_frag[n][2]);
                    atomicAdd(&dQ_acc[dq_base + r1 * head_dim + c1], dQ_frag[n][3]);
                }
            }
            __syncthreads();
        }
    }

    // Write dK, dV to global
    int kv_valid = min(BLOCK_KV, kv_len - kv_start);
    size_t dk_base = ((size_t)batch_idx * kv_head * kv_len + kv_head_idx * kv_len + kv_start) * head_dim;
    for (int n = 0; n < 8; n++) {
        int c0 = n * 8 + tid_in_group * 2;
        int c1 = c0 + 1;
        int r0 = warp_id * 16 + group_id;
        int r1 = r0 + 8;
        if (r0 < kv_valid) {
            dK_acc[dk_base + r0 * head_dim + c0] = dK_frag[n][0];
            dK_acc[dk_base + r0 * head_dim + c1] = dK_frag[n][1];
            dV_acc[dk_base + r0 * head_dim + c0] = dV_frag[n][0];
            dV_acc[dk_base + r0 * head_dim + c1] = dV_frag[n][1];
        }
        if (r1 < kv_valid) {
            dK_acc[dk_base + r1 * head_dim + c0] = dK_frag[n][2];
            dK_acc[dk_base + r1 * head_dim + c1] = dK_frag[n][3];
            dV_acc[dk_base + r1 * head_dim + c0] = dV_frag[n][2];
            dV_acc[dk_base + r1 * head_dim + c1] = dV_frag[n][3];
        }
    }
}

void attention_v6_backward(
    const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    const nv_bfloat16 *O,
    const nv_bfloat16 *L_bf16,
    const nv_bfloat16 *dO,
    nv_bfloat16 *dQ,
    nv_bfloat16 *dK,
    nv_bfloat16 *dV,
    int batch_size,
    int q_head,
    int kv_head,
    int q_len,
    int kv_len,
    int head_dim,
    bool is_causal,
    cudaStream_t stream)
{
    ASSERT_NOT_NULL(Q, K, V, O, dO, dQ, dK, dV);
    if (q_head % kv_head)
        ERROR("q_head(%d) %% kv_head(%d) != 0", q_head, kv_head);

    int q_kv_ratio = q_head / kv_head;

    // Convert L from bf16 to float
    int L_size = batch_size * q_head * q_len;
    float *L_float;
    CUDA_CHECK(cudaMalloc(&L_float, L_size * sizeof(float)));
    {
        int block = 256;
        int grid = cdiv(L_size, block);
        bf16_to_float_kernel<<<grid, block, 0, stream>>>(L_bf16, L_float, L_size);
    }

    // Compute D = rowsum(dO * O)
    float *D;
    CUDA_CHECK(cudaMalloc(&D, L_size * sizeof(float)));
    {
        int block = 256;
        dim3 grid(cdiv(q_len, block), batch_size * q_head);
        compute_D_kernel<<<grid, block, 0, stream>>>(O, dO, D, q_head, q_len, head_dim);
    }

    // Allocate float accumulators
    int dQ_size = batch_size * q_head * q_len * head_dim;
    int dKV_size = batch_size * kv_head * kv_len * head_dim;
    float *dQ_acc, *dK_acc, *dV_acc;
    CUDA_CHECK(cudaMalloc(&dQ_acc, dQ_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK_acc, dKV_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_acc, dKV_size * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_acc, 0, dQ_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_acc, 0, dKV_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_acc, 0, dKV_size * sizeof(float), stream));

    // Launch backward kernel
    constexpr int BLOCK_Q = 64, BLOCK_KV = 64, DIM = 64, NUM_WARPS = 4;
    int num_kv_blocks = cdiv(kv_len, BLOCK_KV);
    int num_blocks = num_kv_blocks * batch_size * kv_head;
    int block_size = NUM_WARPS * WARP_SIZE;
    int smem_size = 4 * BLOCK_KV * DIM * sizeof(nv_bfloat16); // 32KB

    if (is_causal) {
        launch_kernel(
            flash_attention_backward_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, true>,
            num_blocks, block_size, smem_size, stream,
            Q, K, V, L_float, dO, D, dQ_acc, dK_acc, dV_acc,
            batch_size, q_head, kv_head, q_len, kv_len, head_dim,
            q_kv_ratio, 1.0f / sqrtf((float)head_dim));
    } else {
        launch_kernel(
            flash_attention_backward_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, false>,
            num_blocks, block_size, smem_size, stream,
            Q, K, V, L_float, dO, D, dQ_acc, dK_acc, dV_acc,
            batch_size, q_head, kv_head, q_len, kv_len, head_dim,
            q_kv_ratio, 1.0f / sqrtf((float)head_dim));
    }

    // Convert float accumulators to bf16
    {
        int block = 256;
        float_to_bf16_kernel<<<cdiv(dQ_size, block), block, 0, stream>>>(dQ_acc, dQ, dQ_size);
        float_to_bf16_kernel<<<cdiv(dKV_size, block), block, 0, stream>>>(dK_acc, dK, dKV_size);
        float_to_bf16_kernel<<<cdiv(dKV_size, block), block, 0, stream>>>(dV_acc, dV, dKV_size);
    }

    // Synchronize before freeing temporary buffers to ensure all kernels
    // on the stream have completed using these buffers.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(L_float));
    CUDA_CHECK(cudaFree(D));
    CUDA_CHECK(cudaFree(dQ_acc));
    CUDA_CHECK(cudaFree(dK_acc));
    CUDA_CHECK(cudaFree(dV_acc));
}

// Flash Attention – self-contained CUDA kernel
// (merged from my-flash-attention/common.h, attention_v6.cu, attention_v6_bp.cu)

#include <cfloat>
#include <cstdint>
#include <iostream>
#include <random>

#include <cuda_bf16.h>

// ============================================================
// Section 1: common utilities (from common.h)
// ============================================================

#define ANSI_GREEN "\033[32m"
#define ANSI_RED "\033[31m"
#define ANSI_RESET "\033[0m"
#define FA_CUDA_CHECK(x)                                                                                               \
    {                                                                                                                  \
        auto error = x;                                                                                                \
        if (error != cudaSuccess) {                                                                                    \
            std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;               \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

#define FA_ERROR(fmt, ...)                                                                                             \
    do {                                                                                                               \
        fprintf(stderr, ANSI_RED "[ERROR]: [%s:%d] " fmt ANSI_RESET "\n", __FILE__, __LINE__, ##__VA_ARGS__);          \
        exit(1);                                                                                                       \
    } while (0)

#define FA_ASSERT_NOT_NULL(...)                                                                                        \
    do {                                                                                                               \
        const void *_ptrs[] = {__VA_ARGS__};                                                                           \
        const char *_names = #__VA_ARGS__;                                                                             \
        int _n = sizeof(_ptrs) / sizeof(_ptrs[0]);                                                                     \
        char _buf[256];                                                                                                \
        strncpy(_buf, _names, sizeof(_buf));                                                                           \
        char *_tok = strtok(_buf, ",");                                                                                \
        for (int _i = 0; _i < _n; _i++) {                                                                              \
            if (!_ptrs[_i]) {                                                                                          \
                FA_ERROR("assertion failed: '%s' is a nullptr", _tok ? _tok : "?");                                    \
            }                                                                                                          \
            _tok = strtok(NULL, " ,");                                                                                 \
        }                                                                                                              \
    } while (0)

inline constexpr int FA_WARP_SIZE = 32;

__device__ __host__ constexpr int fa_cdiv(int a, int b) { return (a + b - 1) / b; }

// NOTE: stride in bytes
template <int STRIDE> __device__ uint32_t fa_swizzle(uint32_t index) {
    if constexpr (STRIDE == 16) {
        return index;
    }
    uint32_t row_idx = (index / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void fa_global_to_shared_swizzle(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
    constexpr int num_elems = 16 / sizeof(nv_bfloat16);
    constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
    for (int iter = 0; iter < num_iters; iter++) {
        const int idx = (iter * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        const uint32_t dst_addr
            = fa_swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
        const nv_bfloat16 *src_addr = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_addr), "l"(src_addr));
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void fa_global_to_shared_swizzle_padded(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid,
                                                          int valid_height) {
    constexpr int num_elems = 16 / sizeof(nv_bfloat16);
    constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
    for (int iter = 0; iter < num_iters; iter++) {
        const int idx = (iter * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        const uint32_t dst_addr
            = fa_swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
        if (row < valid_height) {
            const nv_bfloat16 *src_addr = src + (row * src_stride + col);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_addr), "l"(src_addr));
        } else {
            asm volatile("{\n"
                         ".reg .v4 .b32 zeros;\n"
                         "mov.v4.b32 zeros, {0, 0, 0, 0};\n"
                         "st.shared.v4.b32 [%0], zeros;\n"
                         "}\n" ::"r"(dst_addr));
        }
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void fa_global_to_shared_swizzle_float2bfloat16(uint32_t dst, const float *src, int stride, int tid) {
    constexpr int num_elems = 16 / sizeof(float);
    constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
#pragma unroll
    for (int iter = 0; iter < num_iters; iter++) {
        const int idx = (iter * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        const uint32_t dst_addr
            = fa_swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
        const float *src_addr = src + (row * stride + col);
        float4 reg_f4 = *reinterpret_cast<const float4 *>(src_addr);
        __nv_bfloat162 bf_01 = __floats2bfloat162_rn(reg_f4.x, reg_f4.y);
        __nv_bfloat162 bf_23 = __floats2bfloat162_rn(reg_f4.z, reg_f4.w);
        uint2 reg_store;
        reg_store.x = reinterpret_cast<uint32_t &>(bf_01);
        reg_store.y = reinterpret_cast<uint32_t &>(bf_23);
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(dst_addr), "r"(reg_store.x), "r"(reg_store.y));
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void fa_global_to_shared_swizzle_float2bfloat16_padded(uint32_t dst, const float *src, int stride,
                                                                         int tid, int valid_height) {
    constexpr int num_elems = 16 / sizeof(float);
    constexpr int num_iters = (HEIGHT * WIDTH) / (TB_SIZE * num_elems);
#pragma unroll
    for (int iter = 0; iter < num_iters; iter++) {
        const int idx = (iter * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        float4 reg_f4;
        if (row < valid_height) {
            const float *src_addr = src + (row * stride + col);
            reg_f4 = *reinterpret_cast<const float4 *>(src_addr);
        } else {
            reg_f4 = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        __nv_bfloat162 bf_01 = __floats2bfloat162_rn(reg_f4.x, reg_f4.y);
        __nv_bfloat162 bf_23 = __floats2bfloat162_rn(reg_f4.z, reg_f4.w);
        const uint32_t dst_addr
            = fa_swizzle<WIDTH * sizeof(__nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(__nv_bfloat16));
        uint2 reg_store;
        reg_store.x = reinterpret_cast<uint32_t &>(bf_01);
        reg_store.y = reinterpret_cast<uint32_t &>(bf_23);
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(dst_addr), "r"(reg_store.x), "r"(reg_store.y));
    }
}

__device__ inline void fa_ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(regs[0]), "=r"(regs[1]) : "r"(addr));
}

__device__ inline void fa_ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ inline void fa_ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                 : "=r"(regs[0]), "=r"(regs[1])
                 : "r"(addr));
}

__device__ inline void fa_ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ inline void fa_mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(D[0]), "f"(D[1]), "f"(D[2]),
                   "f"(D[3]));
}

// ============================================================
// Section 2: Forward kernels (from attention_v6.cu)
// ============================================================

template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__global__ void flash_atten_kernel(const nv_bfloat16 *Q, const nv_bfloat16 *K, const nv_bfloat16 *V, nv_bfloat16 *O,
                                   float *L_out, // [bs * q_head * q_len], logsumexp per row
                                   const float scale, int q_len, int kv_len, int bs, int q_head, int kv_head,
                                   int q_kv_ratio = 1) {
    constexpr int TB_SIZE = NUM_WARPS * FA_WARP_SIZE;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / FA_WARP_SIZE;
    const int lane_id = tid % FA_WARP_SIZE;

    const int num_q_blocks = fa_cdiv(q_len, BLOCK_Q);
    const int bs_id = bid / num_q_blocks;
    const int batch_id = bs_id / q_head;
    const int q_head_id = bs_id % q_head;
    const int kv_head_id = q_head_id / q_kv_ratio;
    const int q_block_id = bid % num_q_blocks;

    Q += (bs_id * q_len * DIM + q_block_id * BLOCK_Q * DIM);
    K += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM);
    V += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM);
    O += (bs_id * q_len * DIM + q_block_id * BLOCK_Q * DIM);
    L_out += (bs_id * q_len + q_block_id * BLOCK_Q);

    extern __shared__ nv_bfloat16 smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem;
    const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);

    constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = fa_swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = fa_swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = fa_swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    const int valid_rows = min(BLOCK_Q, q_len - q_block_id * BLOCK_Q);
    if (valid_rows == BLOCK_Q) {
        fa_global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
    } else {
        fa_global_to_shared_swizzle_padded<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, valid_rows);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
            uint32_t addr = Q_smem_thread;
            addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
            addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
            fa_ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }
    }
    __syncthreads();

    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }
    for (int off_kv = 0; off_kv < kv_len; off_kv += BLOCK_KV) {
        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

        int valid_kv_rows = min(BLOCK_KV, kv_len - off_kv);
        if (valid_kv_rows == BLOCK_KV) {
            fa_global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid);
        } else {
            fa_global_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid, valid_kv_rows);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                uint32_t addr = K_smem_thread;
                addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                fa_ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                    fa_mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d],
                                    S_rmem[mma_id_q][mma_id_kv]);
                }
            }
        }
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                for (int reg_id = 0; reg_id < 4; reg_id++) { S_rmem[mma_id_q][mma_id_kv][reg_id] *= scale; }
            }

            float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_rmem[mma_id_q][mma_id_kv];
                this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
            }

            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

            this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

            float rescale[2];
            rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
            rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
            }

            rowmax[mma_id_q][0] = this_rowmax[0];
            rowmax[mma_id_q][1] = this_rowmax[1];

            float this_rowsumexp[2] = {};
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_rmem[mma_id_q][mma_id_kv];
                for (int i = 0; i < 4; i++) { regs[i] = __expf(regs[i] - rowmax[mma_id_q][i / 2]); }
                this_rowsumexp[0] += regs[0] + regs[1];
                this_rowsumexp[1] += regs[2] + regs[3];

                nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
                this_P_rmem[(mma_id_kv % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
                this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
            }
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

            rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
            rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
        }
        if (valid_kv_rows == BLOCK_KV) {
            fa_global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid);
        } else {
            fa_global_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid, valid_kv_rows);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                uint32_t addr = V_smem_thread;
                addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
                fa_ldmatrix_x2_trans(V_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                    fa_mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d],
                                    O_rmem[mma_id_q][mma_id_d]);
                }
            }
        }

        K += valid_kv_rows * DIM;
        V += valid_kv_rows * DIM;
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
            float *regs = O_rmem[mma_id_q][mma_id_d];
            regs[0] /= rowsumexp[mma_id_q][0];
            regs[1] /= rowsumexp[mma_id_q][0];
            regs[2] /= rowsumexp[mma_id_q][1];
            regs[3] /= rowsumexp[mma_id_q][1];
            const int local_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int global_row = q_block_id * BLOCK_Q + local_row;

            if (global_row < q_len) {
                reinterpret_cast<nv_bfloat162 *>(O + local_row * DIM + col)[0]
                    = __float22bfloat162_rn({regs[0], regs[1]});
            }
            if (global_row + 8 < q_len) {
                reinterpret_cast<nv_bfloat162 *>(O + (local_row + 8) * DIM + col)[0]
                    = __float22bfloat162_rn({regs[2], regs[3]});
            }
        }
        if ((lane_id & 3) == 0) {
            const int local_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int global_row = q_block_id * BLOCK_Q + local_row;
            if (global_row < q_len) {
                L_out[local_row] = rowmax[mma_id_q][0] + logf(rowsumexp[mma_id_q][0]);
            }
            if (global_row + 8 < q_len) {
                L_out[local_row + 8] = rowmax[mma_id_q][1] + logf(rowsumexp[mma_id_q][1]);
            }
        }
    }
}

template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__global__ void flash_atten_kernel_causal(const nv_bfloat16 *Q, const nv_bfloat16 *K, const nv_bfloat16 *V,
                                          nv_bfloat16 *O, float *L_out, const float scale, int q_len, int kv_len,
                                          int bs, int q_head, int kv_head, int q_kv_ratio = 1) {
    constexpr int TB_SIZE = NUM_WARPS * FA_WARP_SIZE;
    const int WARP_Q = BLOCK_Q / NUM_WARPS;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / FA_WARP_SIZE;
    const int lane_id = tid % FA_WARP_SIZE;

    const int num_q_blocks = fa_cdiv(q_len, BLOCK_Q);
    const int bs_id = bid / num_q_blocks;
    const int batch_id = bs_id / q_head;
    const int q_head_id = bs_id % q_head;
    const int kv_head_id = q_head_id / q_kv_ratio;
    const int q_block_id = bid % num_q_blocks;

    Q += (bs_id * q_len * DIM + q_block_id * BLOCK_Q * DIM);
    K += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM);
    V += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM);
    O += (bs_id * q_len * DIM + q_block_id * BLOCK_Q * DIM);
    L_out += (bs_id * q_len + q_block_id * BLOCK_Q);

    const int MMA_M = 16;
    const int MMA_N = 8;
    const int MMA_K = 16;

    extern __shared__ nv_bfloat16 smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem;
    const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);

    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][4];
    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

    int q_valid_rows = min(BLOCK_Q, q_len - BLOCK_Q * q_block_id);
    if (q_valid_rows < BLOCK_Q) {
        fa_global_to_shared_swizzle_padded<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, q_valid_rows);
    } else {
        fa_global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = fa_swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = fa_swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = fa_swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
            uint32_t addr = Q_smem_thread;
            addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
            addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
            fa_ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }
    }
    __syncthreads();

    int causal_offset = 0;
    int q_start = q_block_id * BLOCK_Q + warp_id * WARP_Q;
    int q_end = q_start + min(WARP_Q, q_len - q_start);
    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};
    for (int i = 0; i < WARP_Q / MMA_M; i++) {
        rowmax[i][0] = -FLT_MAX;
        rowmax[i][1] = -FLT_MAX;
    }
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
    for (int off_kv = 0; off_kv < kv_len; off_kv += BLOCK_KV) {
        if (off_kv > q_end - 1 + causal_offset) {
            break;
        }
        int valid_kv_rows = min(BLOCK_KV, kv_len - off_kv);
        int end_kv = off_kv + valid_kv_rows;

        if (valid_kv_rows < BLOCK_KV) {
            fa_global_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid, valid_kv_rows);
            fa_global_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid, valid_kv_rows);
        } else {
            fa_global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid);
            fa_global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
        if (end_kv - 1 <= q_start + causal_offset) {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                    uint32_t addr = K_smem_thread;
                    addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                    fa_ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
                }
            }

            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                        fa_mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d],
                                        S_rmem[mma_id_q][mma_id_kv]);
                    }
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    for (int i = 0; i < 4; i++) {
                        regs[i] *= scale;
                        int row_idx_mma = (lane_id >> 2) + 8 * (i >> 1);
                        int row_idx_global = q_start + mma_id_q * MMA_M + row_idx_mma;
                        int col_idx_mma = (lane_id % 4) * 2 + (i & 0x1);
                        int col_idx_global = off_kv + mma_id_kv * MMA_N + col_idx_mma;
                        if (col_idx_global > row_idx_global + causal_offset || col_idx_global >= kv_len) {
                            regs[i] = -FLT_MAX;
                        }
                    }
                    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
                }

                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

                this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
                this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

                float rescale[2];
                rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
                rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                    for (int j = 0; j < 4; j++) { O_rmem[mma_id_q][mma_id_d][j] *= rescale[j / 2]; }
                }

                rowmax[mma_id_q][0] = this_rowmax[0];
                rowmax[mma_id_q][1] = this_rowmax[1];

                float this_rowsumexp[2] = {};
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
                    regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
                    regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
                    regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);
                    this_rowsumexp[0] += regs[0] + regs[1];
                    this_rowsumexp[1] += regs[2] + regs[3];
                    nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
                    this_P_rmem[(mma_id_kv % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
                    this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
                }

                this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

                rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
                rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
            }

            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                    uint32_t addr = V_smem_thread;
                    addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
                    fa_ldmatrix_x2_trans(V_rmem[mma_id_kv][mma_id_d], addr);
                }
            }
            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                        fa_mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d],
                                        O_rmem[mma_id_q][mma_id_d]);
                    }
                }
            }
            K += valid_kv_rows * DIM;
            V += valid_kv_rows * DIM;
        } else {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                    uint32_t addr = K_smem_thread;
                    addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                    fa_ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
                }
            }

            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
                        fa_mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d],
                                        S_rmem[mma_id_q][mma_id_kv]);
                    }
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    for (int i = 0; i < 4; i++) {
                        regs[i] *= scale;
                        int row_idx_mma = (lane_id >> 2) + 8 * (i >> 1);
                        int row_idx_global = q_start + mma_id_q * MMA_M + row_idx_mma;
                        int col_idx_mma = (lane_id % 4) * 2 + (i & 0x1);
                        int col_idx_global = off_kv + mma_id_kv * MMA_N + col_idx_mma;
                        if (col_idx_global > row_idx_global || col_idx_global >= kv_len) {
                            regs[i] = -FLT_MAX;
                        }
                    }
                    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
                }
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

                this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
                this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);
                float rescale[2];
                rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
                rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                    O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                    O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                    O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                    O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
                }
                rowmax[mma_id_q][0] = this_rowmax[0];
                rowmax[mma_id_q][1] = this_rowmax[1];

                float this_rowsumexp[2] = {};
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
                    regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
                    regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
                    regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);
                    this_rowsumexp[0] += (regs[0] + regs[1]);
                    this_rowsumexp[1] += (regs[2] + regs[3]);
                    nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
                    this_P_rmem[(mma_id_kv % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
                    this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
                }

                this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

                rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
                rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
            }
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                    uint32_t addr = V_smem_thread;
                    addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
                    addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
                    fa_ldmatrix_x2_trans(V_rmem[mma_id_kv][mma_id_d], addr);
                }
            }
            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                        fa_mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d],
                                        O_rmem[mma_id_q][mma_id_d]);
                    }
                }
            }
            K += valid_kv_rows * DIM;
            V += valid_kv_rows * DIM;
        }
    }
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
            float *regs = O_rmem[mma_id_q][mma_id_d];
            regs[0] /= rowsumexp[mma_id_q][0];
            regs[1] /= rowsumexp[mma_id_q][0];
            regs[2] /= rowsumexp[mma_id_q][1];
            regs[3] /= rowsumexp[mma_id_q][1];
            const int local_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int global_row = q_block_id * BLOCK_Q + local_row;

            if (global_row < q_len) {
                reinterpret_cast<nv_bfloat162 *>(O + local_row * DIM + col)[0]
                    = __float22bfloat162_rn({regs[0], regs[1]});
            }
            if (global_row + 8 < q_len) {
                reinterpret_cast<nv_bfloat162 *>(O + (local_row + 8) * DIM + col)[0]
                    = __float22bfloat162_rn({regs[2], regs[3]});
            }
        }
        if ((lane_id & 3) == 0) {
            const int local_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int global_row = q_block_id * BLOCK_Q + local_row;
            if (global_row < q_len) {
                L_out[local_row] = rowmax[mma_id_q][0] + logf(rowsumexp[mma_id_q][0]);
            }
            if (global_row + 8 < q_len) {
                L_out[local_row + 8] = rowmax[mma_id_q][1] + logf(rowsumexp[mma_id_q][1]);
            }
        }
    }
}

static void attention_v6_impl(const nv_bfloat16 *Q, const nv_bfloat16 *K, const nv_bfloat16 *V, nv_bfloat16 *O,
                              float *L_out, int bs, int q_head, int kv_head, int q_len, int kv_len, int head_dim,
                              bool is_causal, float scale, cudaStream_t stream) {
    FA_ASSERT_NOT_NULL(Q, K, V, O);
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int TB_SIZE = 128;
    constexpr int NUM_WARPS = 4;
    constexpr int DIM = 64;

    if (head_dim != 64) {
        FA_ERROR("current only support head_dim=64");
    }
    const int num_blocks = bs * q_head * fa_cdiv(q_len, BLOCK_Q);
    const int smem_size = max(BLOCK_Q, BLOCK_KV * 2) * DIM * sizeof(nv_bfloat16);
    if (scale < 0.0f) {
        scale = 1.0f / sqrtf((float(DIM)));
    }
    if (!is_causal) {
        cudaFuncSetAttribute(flash_atten_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        flash_atten_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS><<<num_blocks, TB_SIZE, smem_size, stream>>>(
            Q, K, V, O, L_out, scale, q_len, kv_len, bs, q_head, kv_head, q_head / kv_head);
    } else {
        cudaFuncSetAttribute(flash_atten_kernel_causal<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        flash_atten_kernel_causal<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS><<<num_blocks, TB_SIZE, smem_size, stream>>>(
            Q, K, V, O, L_out, scale, q_len, kv_len, bs, q_head, kv_head, q_head / kv_head);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================
// Section 3: Backward kernels (from attention_v6_bp.cu)
// ============================================================

const int BP_MMA_M = 16;
const int BP_MMA_K = 16;
const int BP_MMA_N = 8;

__device__ __forceinline__ void fa_ABT(const int a_height, const int b_height, const int dim, auto &a_rmem,
                                       auto &b_rmem, auto &c_rmem) {
    for (int mma_id_a = 0; mma_id_a < a_height / BP_MMA_M; mma_id_a++) {
        for (int mma_id_b = 0; mma_id_b < b_height / BP_MMA_N; mma_id_b++) {
            for (int mma_id_d = 0; mma_id_d < dim / BP_MMA_K; mma_id_d++) {
                fa_mma_m16n8k16(a_rmem[mma_id_a][mma_id_d], b_rmem[mma_id_b][mma_id_d], c_rmem[mma_id_a][mma_id_b]);
            }
        }
    }
}

__device__ __forceinline__ void fa_ATB(const int height, const int a_dim, const int b_dim, auto &a_rmem, auto &b_rmem,
                                       auto &c_rmem) {
    for (int mma_id_a_dim = 0; mma_id_a_dim < a_dim / BP_MMA_M; mma_id_a_dim++) {
        for (int mma_id_b_dim = 0; mma_id_b_dim < b_dim / BP_MMA_N; mma_id_b_dim++) {
            for (int mma_id_h = 0; mma_id_h < height / BP_MMA_K; mma_id_h++) {
                fa_mma_m16n8k16(a_rmem[mma_id_h][mma_id_a_dim], b_rmem[mma_id_h][mma_id_b_dim],
                                c_rmem[mma_id_a_dim][mma_id_b_dim]);
            }
        }
    }
}

__device__ __forceinline__ void fa_AB(const int m, const int k, const int n, auto &a_rmem, auto &b_rmem, auto &c_rmem) {
    for (int mma_id_m = 0; mma_id_m < m / BP_MMA_M; mma_id_m++) {
        for (int mma_id_n = 0; mma_id_n < n / BP_MMA_N; mma_id_n++) {
            for (int mma_id_k = 0; mma_id_k < k / BP_MMA_K; mma_id_k++) {
                fa_mma_m16n8k16(a_rmem[mma_id_m][mma_id_k], b_rmem[mma_id_k][mma_id_n], c_rmem[mma_id_m][mma_id_n]);
            }
        }
    }
}

__device__ __forceinline__ void fa_get_dS(auto &p_rmem, auto &dp_rmem, auto &d, int height, int width, int lane_id) {
    for (int mma_id_h = 0; mma_id_h < height / BP_MMA_M; mma_id_h++) {
        for (int mma_id_w = 0; mma_id_w < width / BP_MMA_N; mma_id_w++) {
            float *dp_regs = dp_rmem[mma_id_h][mma_id_w];
            nv_bfloat162 *this_p_rmem = reinterpret_cast<nv_bfloat162 *>(p_rmem[mma_id_h][mma_id_w / 2]);
            int row_idx1 = mma_id_h * BP_MMA_M + (lane_id >> 2);
            int row_idx2 = row_idx1 + 8;
            this_p_rmem[(mma_id_w % 2) * 2]
                = this_p_rmem[(mma_id_w % 2) * 2]
                * __float22bfloat162_rn({dp_regs[0] - d[row_idx1], dp_regs[1] - d[row_idx1]});
            this_p_rmem[(mma_id_w % 2) * 2 + 1]
                = this_p_rmem[(mma_id_w % 2) * 2 + 1]
                * __float22bfloat162_rn({dp_regs[2] - d[row_idx2], dp_regs[3] - d[row_idx2]});
        }
    }
}

__device__ __forceinline__ void fa_compute_P(const int q_height, const int kv_height, const int dim, auto &q_rmem,
                                             auto &kv_rmem, auto &s_rmem, auto &P_smem, auto &L, bool is_causal,
                                             int q_start, int kv_start, int kv_len, int lane_id, int warp_id,
                                             int BLOCK_KV, float scale) {
    for (int mma_id_q = 0; mma_id_q < q_height / BP_MMA_M; mma_id_q++) {
        for (int mma_id_kv = 0; mma_id_kv < kv_height / BP_MMA_N; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < dim / BP_MMA_K; mma_id_d++) {
                fa_mma_m16n8k16(q_rmem[mma_id_q][mma_id_d], kv_rmem[mma_id_kv][mma_id_d], s_rmem[mma_id_q][mma_id_kv]);
            }
            for (int i = 0; i < 4; i++) {
                int q_local_idx = mma_id_q * BP_MMA_M + (lane_id >> 2) + 8 * (i >= 2);
                int kv_local_idx = mma_id_kv * BP_MMA_N + (lane_id % 4) * 2 + (i & 0x1);
                int q_global_idx = q_start + q_local_idx;
                int kv_global_idx = kv_start + kv_local_idx;
                bool mask = (kv_global_idx >= kv_len) || (is_causal && kv_global_idx > q_global_idx);
                s_rmem[mma_id_q][mma_id_kv][i]
                    = mask ? 0 : expf(s_rmem[mma_id_q][mma_id_kv][i] * scale - L[q_local_idx]);
                uint32_t byte_off = (q_local_idx * BLOCK_KV + warp_id * kv_height + kv_local_idx) * sizeof(nv_bfloat16);
                uint32_t swz_off = fa_swizzle<128>(P_smem + byte_off);
                nv_bfloat16 *dst = reinterpret_cast<nv_bfloat16 *>(__cvta_shared_to_generic(swz_off));
                *dst = __float2bfloat16(s_rmem[mma_id_q][mma_id_kv][i]);
            }
        }
    }
}

__device__ __forceinline__ void fa_write_dQ(auto &dq_rmem, auto &d_q, int height, int width, int lane_id, int q_start,
                                            int q_len, float scale) {
    for (int mma_id_q = 0; mma_id_q < height / BP_MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < width / BP_MMA_N; mma_id_d++) {
            int q_local_idx = mma_id_q * BP_MMA_M + (lane_id >> 2);
            int q_global_idx = q_local_idx + q_start;
            int d_local_idx = (lane_id % 4) * 2;
            if (q_global_idx < q_len) {
                float *this_dq = &d_q[q_global_idx * width + mma_id_d * BP_MMA_N + d_local_idx];
                atomicAdd(this_dq, dq_rmem[mma_id_q][mma_id_d][0] * scale);
                atomicAdd(this_dq + 1, dq_rmem[mma_id_q][mma_id_d][1] * scale);
            }
            if (q_global_idx + 8 < q_len) {
                float *this_dq = &d_q[(q_global_idx + 8) * width + mma_id_d * BP_MMA_N + d_local_idx];
                atomicAdd(this_dq, dq_rmem[mma_id_q][mma_id_d][2] * scale);
                atomicAdd(this_dq + 1, dq_rmem[mma_id_q][mma_id_d][3] * scale);
            }
        }
    }
}

__device__ __forceinline__ void fa_write_dkv(auto &d_kv_rmem, auto &d_kv, int height, int width, int lane_id,
                                             int kv_start, int kv_len, float scale) {
    for (int mma_id_kv = 0; mma_id_kv < height / BP_MMA_M; mma_id_kv++) {
        for (int mma_id_d = 0; mma_id_d < width / BP_MMA_N; mma_id_d++) {
            int kv_local_idx = mma_id_kv * BP_MMA_M + (lane_id >> 2);
            int kv_global_idx = kv_start + kv_local_idx;
            int d_local_idx = (lane_id % 4) * 2;
            if (kv_global_idx < kv_len) {
                float2 vals = {d_kv_rmem[mma_id_kv][mma_id_d][0] * scale, d_kv_rmem[mma_id_kv][mma_id_d][1] * scale};
                float2 *this_dkv
                    = reinterpret_cast<float2 *>(&d_kv[kv_global_idx * width + mma_id_d * BP_MMA_N + d_local_idx]);
                *this_dkv = vals;
            }
            if (kv_global_idx + 8 < kv_len) {
                float2 vals = {d_kv_rmem[mma_id_kv][mma_id_d][2] * scale, d_kv_rmem[mma_id_kv][mma_id_d][3] * scale};
                float2 *this_dkv = reinterpret_cast<float2 *>(
                    &d_kv[(kv_global_idx + 8) * width + mma_id_d * BP_MMA_N + d_local_idx]);
                *this_dkv = vals;
            }
        }
    }
}

template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__global__ void flash_atten_bakward_1(const nv_bfloat16 *Q, const nv_bfloat16 *K, const nv_bfloat16 *V,
                                      const nv_bfloat16 *O, const float *L, const float *D, const float *dO, float *dQ,
                                      float *d_temp_K, // [batch_size, q_head, kv_len, dim]
                                      float *d_temp_V, // [batch_size, q_head, kv_len, dim]
                                      int bs, int q_head, int kv_head, int q_len, int kv_len, int head_dim, float scale,
                                      bool is_causal = false, int q_kv_ratio = 1) {
    constexpr int TB_SIZE = NUM_WARPS * FA_WARP_SIZE;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / FA_WARP_SIZE;
    const int lane_id = tid % FA_WARP_SIZE;
    const int num_kv_blocks = fa_cdiv(kv_len, BLOCK_KV);
    const int bs_id = bid / num_kv_blocks;
    const int batch_id = bs_id / q_head;
    const int q_head_id = bs_id % q_head;
    const int kv_head_id = q_head_id / q_kv_ratio;
    const int kv_block_id = bid % num_kv_blocks;
    const int WARP_KV = BLOCK_KV / NUM_WARPS;

    Q += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM);
    dQ += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM);
    K += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM + kv_block_id * BLOCK_KV * DIM);
    V += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM + kv_block_id * BLOCK_KV * DIM);
    O += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM);
    dO += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM);
    d_temp_K += (batch_id * q_head * kv_len * DIM + q_head_id * kv_len * DIM);
    d_temp_V += (batch_id * q_head * kv_len * DIM + q_head_id * kv_len * DIM);
    L += (batch_id * q_head * q_len + q_head_id * q_len);
    D += (batch_id * q_head * q_len + q_head_id * q_len);

    extern __shared__ nv_bfloat16 smem[];
    const uint32_t K_smem = __cvta_generic_to_shared(smem);
    const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    const uint32_t Q_smem = V_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    const uint32_t L_smem = Q_smem + BLOCK_Q * DIM * sizeof(nv_bfloat16);
    const uint32_t D_smem = L_smem + BLOCK_Q * sizeof(float);
    const uint32_t dO_smem = D_smem + BLOCK_Q * sizeof(float);
    const uint32_t P_smem = dO_smem + BLOCK_Q * DIM * sizeof(nv_bfloat16);

    uint32_t K_smem_thread, V_smem_thread;
    {
        const int row_off = warp_id * WARP_KV + lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = fa_swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        V_smem_thread = fa_swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }

    const int kv_valid_rows = min(BLOCK_KV, kv_len - kv_block_id * BLOCK_KV);
    if (kv_valid_rows == BLOCK_KV) {
        fa_global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid);
        fa_global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid);
    } else {
        fa_global_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid, kv_valid_rows);
        fa_global_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid, kv_valid_rows);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    uint32_t K_rmem[WARP_KV / BP_MMA_N][DIM / BP_MMA_K][2];
    uint32_t V_rmem[WARP_KV / BP_MMA_N][DIM / BP_MMA_K][2];

    for (int mma_id_kv = 0; mma_id_kv < WARP_KV / BP_MMA_N; mma_id_kv++) {
        for (int mma_id_d = 0; mma_id_d < DIM / BP_MMA_K; mma_id_d++) {
            uint32_t k_addr = K_smem_thread;
            uint32_t v_addr = V_smem_thread;
            k_addr += mma_id_kv * BP_MMA_N * DIM * sizeof(nv_bfloat16);
            k_addr ^= mma_id_d * BP_MMA_K * sizeof(nv_bfloat16);
            v_addr += mma_id_kv * BP_MMA_N * DIM * sizeof(nv_bfloat16);
            v_addr ^= mma_id_d * BP_MMA_K * sizeof(nv_bfloat16);
            fa_ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], k_addr);
            fa_ldmatrix_x2(V_rmem[mma_id_kv][mma_id_d], v_addr);
        }
    }

    uint32_t Q_rmem[BLOCK_Q / BP_MMA_M][DIM / BP_MMA_K][4];
    uint32_t dO_right_rmem[BLOCK_Q / BP_MMA_K][DIM / BP_MMA_N][2];
    uint32_t dO_left_rmem[BLOCK_Q / BP_MMA_M][DIM / BP_MMA_K][4];

    float dK_rmem[WARP_KV / BP_MMA_M][DIM / BP_MMA_N][4] = {};
    float dV_rmem[WARP_KV / BP_MMA_M][DIM / BP_MMA_N][4] = {};
    int kv_start = kv_block_id * BLOCK_KV + WARP_KV * warp_id;

    for (int off_q = 0; off_q < q_len; off_q += BLOCK_Q) {
        float S_rmem[BLOCK_Q / BP_MMA_M][WARP_KV / BP_MMA_N][4] = {};
        float dP_rmem[BLOCK_Q / BP_MMA_M][WARP_KV / BP_MMA_N][4] = {};
        float dQ_rmem[BLOCK_Q / BP_MMA_M][DIM / BP_MMA_N][4] = {};

        int q_valid_rows = min(BLOCK_Q, q_len - off_q);
        if (q_valid_rows == BLOCK_Q) {
            fa_global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
            fa_global_to_shared_swizzle_float2bfloat16<BLOCK_Q, DIM, TB_SIZE>(dO_smem, dO, DIM, tid);
        } else {
            fa_global_to_shared_swizzle_padded<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, q_valid_rows);
            fa_global_to_shared_swizzle_float2bfloat16_padded<BLOCK_Q, DIM, TB_SIZE>(dO_smem, dO, DIM, tid,
                                                                                     q_valid_rows);
        }
        for (int i = tid; i < BLOCK_Q; i += TB_SIZE) {
            int idx = i + off_q;
            if (idx < q_len) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((uint32_t)(L_smem + i * sizeof(float))),
                             "l"(&L[i]));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((uint32_t)(D_smem + i * sizeof(float))),
                             "l"(&D[i]));
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        uint32_t Q_smem_thread, dO_left_smem_thread, dO_right_smem_thread;
        {
            const int row_off = lane_id % 16;
            const int col_off = lane_id / 16 * 8;
            Q_smem_thread
                = fa_swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
            dO_left_smem_thread
                = fa_swizzle<DIM * sizeof(nv_bfloat16)>(dO_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        }
        {
            const int row_off = lane_id % 16;
            const int col_off = lane_id / 16 * 8;
            dO_right_smem_thread
                = fa_swizzle<DIM * sizeof(nv_bfloat16)>(dO_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        }

        for (int mma_id_q = 0; mma_id_q < BLOCK_Q / BP_MMA_M; mma_id_q++) {
            for (int mma_id_d = 0; mma_id_d < DIM / BP_MMA_K; mma_id_d++) {
                uint32_t q_addr = Q_smem_thread;
                uint32_t do_left_addr = dO_left_smem_thread;
                q_addr += mma_id_q * BP_MMA_M * DIM * sizeof(nv_bfloat16);
                q_addr ^= mma_id_d * BP_MMA_K * sizeof(nv_bfloat16);
                do_left_addr += mma_id_q * BP_MMA_M * DIM * sizeof(nv_bfloat16);
                do_left_addr ^= mma_id_d * BP_MMA_K * sizeof(nv_bfloat16);
                fa_ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], q_addr);
                fa_ldmatrix_x4(dO_left_rmem[mma_id_q][mma_id_d], do_left_addr);
            }
        }
        for (int mma_id_q = 0; mma_id_q < BLOCK_Q / BP_MMA_K; mma_id_q++) {
            for (int mma_id_d = 0; mma_id_d < DIM / BP_MMA_N; mma_id_d++) {
                uint32_t addr = dO_right_smem_thread;
                addr += mma_id_q * BP_MMA_K * DIM * sizeof(nv_bfloat16);
                addr ^= mma_id_d * BP_MMA_N * sizeof(nv_bfloat16);
                fa_ldmatrix_x2_trans(dO_right_rmem[mma_id_q][mma_id_d], addr);
            }
        }

        float *L_smem_ptr = reinterpret_cast<float *>(__cvta_shared_to_generic(L_smem));
        fa_compute_P(BLOCK_Q, WARP_KV, DIM, Q_rmem, K_rmem, S_rmem, P_smem, L_smem_ptr, is_causal, off_q, kv_start,
                     kv_len, lane_id, warp_id, BLOCK_KV, scale);

        uint32_t P_rmem[BLOCK_Q / BP_MMA_M][WARP_KV / BP_MMA_K][4];
        uint32_t p_smem_thread;
        {
            const int row_off = lane_id % 16;
            const int col_off = lane_id / 16 * 8;
            p_smem_thread = fa_swizzle<BLOCK_KV * sizeof(nv_bfloat16)>(
                P_smem + (row_off * BLOCK_KV + warp_id * WARP_KV + col_off) * sizeof(nv_bfloat16));
        }
        for (int mma_id_q = 0; mma_id_q < BLOCK_Q / BP_MMA_M; mma_id_q++) {
            for (int mma_id_kv = 0; mma_id_kv < WARP_KV / BP_MMA_K; mma_id_kv++) {
                uint32_t addr = p_smem_thread;
                addr += mma_id_q * BP_MMA_M * BLOCK_KV * sizeof(nv_bfloat16);
                addr ^= (mma_id_kv * BP_MMA_K) * sizeof(nv_bfloat16);
                fa_ldmatrix_x4_trans(P_rmem[mma_id_q][mma_id_kv], addr);
                uint32_t tem = P_rmem[mma_id_q][mma_id_kv][1];
                P_rmem[mma_id_q][mma_id_kv][1] = P_rmem[mma_id_q][mma_id_kv][2];
                P_rmem[mma_id_q][mma_id_kv][2] = tem;
            }
        }

        fa_ATB(BLOCK_Q, WARP_KV, DIM, P_rmem, dO_right_rmem, dV_rmem);
        fa_ABT(BLOCK_Q, WARP_KV, DIM, dO_left_rmem, V_rmem, dP_rmem);

        for (int mma_id_q = 0; mma_id_q < BLOCK_Q / BP_MMA_M; mma_id_q++) {
            for (int mma_id_kv = 0; mma_id_kv < WARP_KV / BP_MMA_K; mma_id_kv++) {
                uint32_t addr = p_smem_thread;
                addr += mma_id_q * BP_MMA_M * BLOCK_KV * sizeof(nv_bfloat16);
                addr ^= (mma_id_kv * BP_MMA_K) * sizeof(nv_bfloat16);
                fa_ldmatrix_x4(P_rmem[mma_id_q][mma_id_kv], addr);
            }
        }
        float *D_ptr = reinterpret_cast<float *>(__cvta_shared_to_generic(D_smem));
        fa_get_dS(P_rmem, dP_rmem, D_ptr, BLOCK_Q, WARP_KV, lane_id);

        {
            const int row_off = lane_id % 16 + warp_id * WARP_KV;
            const int col_off = lane_id / 16 * 8;
            K_smem_thread
                = fa_swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        }
        uint32_t new_K_rmem[WARP_KV / BP_MMA_K][DIM / BP_MMA_N][2];
        for (int mma_id_kv = 0; mma_id_kv < WARP_KV / BP_MMA_K; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < DIM / BP_MMA_N; mma_id_d++) {
                uint32_t k_addr = K_smem_thread;
                k_addr += mma_id_kv * BP_MMA_K * DIM * sizeof(nv_bfloat16);
                k_addr ^= mma_id_d * BP_MMA_N * sizeof(nv_bfloat16);
                fa_ldmatrix_x2_trans(new_K_rmem[mma_id_kv][mma_id_d], k_addr);
            }
        }
        fa_AB(BLOCK_Q, WARP_KV, DIM, P_rmem, new_K_rmem, dQ_rmem);

        for (int mma_id_q = 0; mma_id_q < BLOCK_Q / BP_MMA_M; mma_id_q++) {
            for (int mma_id_kv = 0; mma_id_kv < WARP_KV / BP_MMA_K; mma_id_kv++) {
                nv_bfloat162 *regs = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv]);
                int row0 = lane_id >> 2;
                int col0 = (lane_id % 4) * 2;
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < 2; i++) {
                        uint32_t byte_off
                            = (mma_id_q * BP_MMA_M + row0 + 8 * i) * BLOCK_KV * sizeof(nv_bfloat16)
                            + (warp_id * WARP_KV + mma_id_kv * BP_MMA_K + col0 + 8 * j) * sizeof(nv_bfloat16);
                        uint32_t swz_off = fa_swizzle<128>(P_smem + byte_off);
                        nv_bfloat162 *dst = reinterpret_cast<nv_bfloat162 *>(__cvta_shared_to_generic(swz_off));
                        *dst = regs[j * 2 + i];
                    }
                }
            }
        }

        for (int mma_id_q = 0; mma_id_q < BLOCK_Q / BP_MMA_M; mma_id_q++) {
            for (int mma_id_kv = 0; mma_id_kv < WARP_KV / BP_MMA_K; mma_id_kv++) {
                uint32_t addr = p_smem_thread;
                addr += mma_id_q * BP_MMA_M * BLOCK_KV * sizeof(nv_bfloat16);
                addr ^= (mma_id_kv * BP_MMA_K) * sizeof(nv_bfloat16);
                fa_ldmatrix_x4_trans(P_rmem[mma_id_q][mma_id_kv], addr);
                uint32_t tem = P_rmem[mma_id_q][mma_id_kv][1];
                P_rmem[mma_id_q][mma_id_kv][1] = P_rmem[mma_id_q][mma_id_kv][2];
                P_rmem[mma_id_q][mma_id_kv][2] = tem;
            }
        }
        uint32_t Q_rmem_thread_new;
        {
            const int row_off = lane_id % 16;
            const int col_off = lane_id / 16 * 8;
            Q_rmem_thread_new
                = fa_swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
        }
        uint32_t Q_new_rmem[BLOCK_Q / BP_MMA_K][DIM / BP_MMA_N][2];
        for (int mma_id_q = 0; mma_id_q < BLOCK_Q / BP_MMA_K; mma_id_q++) {
            for (int mma_id_d = 0; mma_id_d < DIM / BP_MMA_N; mma_id_d++) {
                uint32_t addr = Q_rmem_thread_new;
                addr += mma_id_q * BP_MMA_K * DIM * sizeof(nv_bfloat16);
                addr ^= mma_id_d * BP_MMA_N * sizeof(nv_bfloat16);
                fa_ldmatrix_x2_trans(Q_new_rmem[mma_id_q][mma_id_d], addr);
            }
        }
        fa_ATB(BLOCK_Q, WARP_KV, DIM, P_rmem, Q_new_rmem, dK_rmem);

        fa_write_dQ(dQ_rmem, dQ, BLOCK_Q, DIM, lane_id, off_q, q_len, scale);

        Q += q_valid_rows * DIM;
        O += q_valid_rows * DIM;
        dO += q_valid_rows * DIM;
        L += q_valid_rows;
        D += q_valid_rows;
    }

    fa_write_dkv(dK_rmem, d_temp_K, WARP_KV, DIM, lane_id, kv_start, kv_len, scale);
    fa_write_dkv(dV_rmem, d_temp_V, WARP_KV, DIM, lane_id, kv_start, kv_len, 1.0);
}

__global__ void fa_compute_D(const float *__restrict__ dO, const nv_bfloat16 *__restrict__ O, float *__restrict__ D,
                             int total_rows, int dim) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= total_rows) {
        return;
    }

    const float *dO_row = dO + row * dim;
    const nv_bfloat16 *O_row = O + row * dim;

    float acc = 0.f;
    for (int d = lane; d < dim; d += 32) { acc += dO_row[d] * __bfloat162float(O_row[d]); }

    for (int off = 16; off > 0; off >>= 1) { acc += __shfl_down_sync(0xffffffff, acc, off); }

    if (lane == 0) {
        D[row] = acc;
    }
}

__global__ void fa_reduce_dkv_grad(const float *__restrict__ d_temp_K, const float *__restrict__ d_temp_V,
                                   float *__restrict__ dK, float *__restrict__ dV, int bs, int q_head, int kv_head,
                                   int kv_len, int dim4, int q_kv_ratio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bs * kv_head * kv_len * dim4;
    if (idx >= total) {
        return;
    }

    int d4_idx = idx % dim4;
    int kl_idx = (idx / dim4) % kv_len;
    int kh_idx = (idx / dim4 / kv_len) % kv_head;
    int b_idx = idx / dim4 / kv_len / kv_head;

    int base = (b_idx * q_head + kh_idx * q_kv_ratio) * kv_len * dim4 + kl_idx * dim4 + d4_idx;
    int stride = kv_len * dim4;

    const float4 *srcK = reinterpret_cast<const float4 *>(d_temp_K);
    const float4 *srcV = reinterpret_cast<const float4 *>(d_temp_V);

    float4 accK = {0.f, 0.f, 0.f, 0.f};
    float4 accV = {0.f, 0.f, 0.f, 0.f};
    for (int r = 0; r < q_kv_ratio; r++) {
        float4 k = srcK[base + r * stride];
        float4 v = srcV[base + r * stride];
        accK.x += k.x;
        accK.y += k.y;
        accK.z += k.z;
        accK.w += k.w;
        accV.x += v.x;
        accV.y += v.y;
        accV.z += v.z;
        accV.w += v.w;
    }
    reinterpret_cast<float4 *>(dK)[idx] = accK;
    reinterpret_cast<float4 *>(dV)[idx] = accV;
}

static void attention_v6_backward_impl(const nv_bfloat16 *Q, const nv_bfloat16 *K, const nv_bfloat16 *V,
                                       const nv_bfloat16 *O, const float *L, const float *dO, float *dQ, float *dK,
                                       float *dV, int batch_size, int q_head, int kv_head, int q_len, int kv_len,
                                       int head_dim, bool is_causal, float scale, cudaStream_t stream) {
    const int q_kv_ratio = q_head / kv_head;
    const int total_rows = batch_size * q_head * q_len;

    float *D_buf = nullptr;
    float *temp_K = nullptr;
    float *temp_V = nullptr;
    cudaMallocAsync(&D_buf, (size_t)total_rows * sizeof(float), stream);
    cudaMallocAsync(&temp_K, (size_t)batch_size * q_head * kv_len * head_dim * sizeof(float), stream);
    cudaMallocAsync(&temp_V, (size_t)batch_size * q_head * kv_len * head_dim * sizeof(float), stream);

    cudaMemsetAsync(dQ, 0, (size_t)batch_size * q_head * q_len * head_dim * sizeof(float), stream);
    cudaMemsetAsync(temp_K, 0, (size_t)batch_size * q_head * kv_len * head_dim * sizeof(float), stream);
    cudaMemsetAsync(temp_V, 0, (size_t)batch_size * q_head * kv_len * head_dim * sizeof(float), stream);

    {
        constexpr int WARPS_PER_BLOCK = 8;
        int grid = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        fa_compute_D<<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(dO, O, D_buf, total_rows, head_dim);
    }

    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int DIM = 64;
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE = NUM_WARPS * 32;

    const int num_kv_blocks = (kv_len + BLOCK_KV - 1) / BLOCK_KV;
    const int num_blocks = batch_size * q_head * num_kv_blocks;

    constexpr size_t smem_size = BLOCK_KV * DIM * sizeof(nv_bfloat16) + BLOCK_KV * DIM * sizeof(nv_bfloat16)
                               + BLOCK_Q * DIM * sizeof(nv_bfloat16) + BLOCK_Q * sizeof(float) + BLOCK_Q * sizeof(float)
                               + BLOCK_Q * DIM * sizeof(nv_bfloat16) + BLOCK_Q * BLOCK_KV * sizeof(nv_bfloat16);

    cudaFuncSetAttribute(flash_atten_bakward_1<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    if (scale < 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }
    flash_atten_bakward_1<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>
        <<<num_blocks, TB_SIZE, smem_size, stream>>>(Q, K, V, O, L, D_buf, dO, dQ, temp_K, temp_V, batch_size, q_head,
                                                     kv_head, q_len, kv_len, head_dim, scale, is_causal, q_kv_ratio);

    {
        int dim4 = head_dim / 4;
        int total = batch_size * kv_head * kv_len * dim4;
        int block = 256;
        int grid = (total + block - 1) / block;
        fa_reduce_dkv_grad<<<grid, block, 0, stream>>>(temp_K, temp_V, dK, dV, batch_size, q_head, kv_head, kv_len,
                                                       dim4, q_kv_ratio);
    }

    cudaFreeAsync(D_buf, stream);
    cudaFreeAsync(temp_K, stream);
    cudaFreeAsync(temp_V, stream);
}

// ============================================================
// Section 4: InfiniTrain Framework Wrappers
// ============================================================

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

std::vector<std::shared_ptr<Tensor>> FlashAttentionForward(const std::shared_ptr<Tensor> &q,
                                                           const std::shared_ptr<Tensor> &k,
                                                           const std::shared_ptr<Tensor> &v, bool is_causal,
                                                           float scale = -1.0f) {
    CHECK(q->Dtype() == DataType::kBFLOAT16);
    const auto &q_dims = q->Dims();
    CHECK_EQ(static_cast<int>(q_dims.size()), 4);
    const int64_t head_dim = q_dims[3];
    CHECK_EQ(head_dim, static_cast<int64_t>(64));

    const int64_t B = q_dims[0];
    const int64_t q_head = q_dims[1];
    const int64_t q_len = q_dims[2];
    const int64_t kv_head = k->Dims()[1];
    const int64_t kv_len = k->Dims()[2];

    auto device = q->GetDevice();
    auto o = std::make_shared<Tensor>(std::vector<int64_t>{B, q_head, q_len, head_dim}, DataType::kBFLOAT16, device);
    auto l = std::make_shared<Tensor>(std::vector<int64_t>{B, q_head, q_len}, DataType::kFLOAT32, device);

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    attention_v6_impl(static_cast<const nv_bfloat16 *>(q->DataPtr()), static_cast<const nv_bfloat16 *>(k->DataPtr()),
                      static_cast<const nv_bfloat16 *>(v->DataPtr()), static_cast<nv_bfloat16 *>(o->DataPtr()),
                      static_cast<float *>(l->DataPtr()), static_cast<int>(B), static_cast<int>(q_head),
                      static_cast<int>(kv_head), static_cast<int>(q_len), static_cast<int>(kv_len),
                      static_cast<int>(head_dim), is_causal, scale, cuda_stream);

    return {o, l};
}

std::vector<std::shared_ptr<Tensor>>
FlashAttentionBackward(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                       const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &o,
                       const std::shared_ptr<Tensor> &l, const std::shared_ptr<Tensor> &do_, bool is_causal,
                       float scale = -1.0f) {
    CHECK(q->Dtype() == DataType::kBFLOAT16);
    CHECK(l->Dtype() == DataType::kFLOAT32);
    const auto &q_dims = q->Dims();
    CHECK_EQ(static_cast<int>(q_dims.size()), 4);
    const int64_t head_dim = q_dims[3];
    CHECK_EQ(head_dim, static_cast<int64_t>(64));

    const int64_t B = q_dims[0];
    const int64_t q_head = q_dims[1];
    const int64_t q_len = q_dims[2];
    const int64_t kv_head = k->Dims()[1];
    const int64_t kv_len = k->Dims()[2];

    auto device = q->GetDevice();

    auto do_f32 = do_->Dtype() == DataType::kFLOAT32 ? do_ : std::make_shared<Tensor>(do_->To(DataType::kFLOAT32));

    auto dq_f32 = std::make_shared<Tensor>(q->Dims(), DataType::kFLOAT32, device);
    auto dk_f32 = std::make_shared<Tensor>(k->Dims(), DataType::kFLOAT32, device);
    auto dv_f32 = std::make_shared<Tensor>(v->Dims(), DataType::kFLOAT32, device);

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    attention_v6_backward_impl(
        static_cast<const nv_bfloat16 *>(q->DataPtr()), static_cast<const nv_bfloat16 *>(k->DataPtr()),
        static_cast<const nv_bfloat16 *>(v->DataPtr()), static_cast<const nv_bfloat16 *>(o->DataPtr()),
        static_cast<const float *>(l->DataPtr()), static_cast<const float *>(do_f32->DataPtr()),
        static_cast<float *>(dq_f32->DataPtr()), static_cast<float *>(dk_f32->DataPtr()),
        static_cast<float *>(dv_f32->DataPtr()), static_cast<int>(B), static_cast<int>(q_head),
        static_cast<int>(kv_head), static_cast<int>(q_len), static_cast<int>(kv_len), static_cast<int>(head_dim),
        is_causal, scale, cuda_stream);

    auto dq = std::make_shared<Tensor>(dq_f32->To(DataType::kBFLOAT16));
    auto dk = std::make_shared<Tensor>(dk_f32->To(DataType::kBFLOAT16));
    auto dv = std::make_shared<Tensor>(dv_f32->To(DataType::kBFLOAT16));

    return {dq, dk, dv};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FLASH_ATTENTION_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionForward)
REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionBackward)

#undef REGISTER_CUDA_FLASH_ATTENTION_KERNEL

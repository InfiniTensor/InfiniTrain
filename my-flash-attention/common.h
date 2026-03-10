#pragma once

#include <iostream>
#include <cstdint>

#include <cuda_bf16.h>

#define ANSI_GREEN "\033[32m"
#define ANSI_RED  "\033[31m"
#define ANSI_RESET "\033[0m"
#define CUDA_CHECK(x)                                                                                                  \
  {                                                                                                                    \
    auto error = x;                                                                                                    \
    if (error != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }

#define LOG(fmt, ...) \
        printf(ANSI_GREEN "[LOG] " fmt ANSI_RESET "\n", ##__VA_ARGS__)

#define ERROR(fmt, ...) \
  do { \
    fprintf(stderr, ANSI_RED "[ERROR]: [%s:%d] " fmt ANSI_RESET "\n",__FILE__, __LINE__ ,##__VA_ARGS__); \
    exit(1); \
  }while(0)

#define ASSERT_NOT_NULL(...) \
    do { \
        const void *_ptrs[] = {__VA_ARGS__}; \
        const char *_names = #__VA_ARGS__; \
        int _n = sizeof(_ptrs) / sizeof(_ptrs[0]); \
        char _buf[256]; \
        strncpy(_buf, _names, sizeof(_buf)); \
        char *_tok = strtok(_buf, ","); \
        for (int _i = 0; _i < _n; _i++) { \
            if (!_ptrs[_i]) { \
                ERROR("assertion failed: '%s' is a nullptr", _tok ? _tok : "?"); \
            } \
            _tok = strtok(NULL, " ,"); \
        } \
    } while(0)
inline constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void gloabl_to_shared_swizzle_padded(
  uint32_t dst,
  const nv_bfloat16 *src,
  int src_stride,
  int tid,
  int valid_height
){
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
  for (int iter = 0; iter < num_iters; iter ++){
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    
    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(
      dst + (row * WIDTH + col) * sizeof(nv_bfloat16)
    );
    if(row < valid_height){
      //有效行，执行异步拷贝
      const nv_bfloat16 *src_addr = src + (row * src_stride + col);
      asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(dst_addr), "l"(src_addr)
      );
    }
    else{
      asm volatile(
        "{\n"
        ".reg .v4 .b32 zeros;\n"
        "mov.v4.b32 zeros, {0, 0, 0, 0};\n"
        "st.shared.v4.b32 [%0], zeros;\n"
        "}\n"
        :: "r"(dst_addr)
      );
    }
  }
}

// 初始化 shared memory 为 0，使用 swizzle 避免 bank conflict
template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void init_shared_zero(uint32_t dst, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);  // 16 bytes per store
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(
        dst + (row * WIDTH + col) * sizeof(nv_bfloat16));

    // 使用 st.shared.v4.b32 一次写入 16 bytes
    asm volatile(
      "{\n"
      ".reg .v4 .b32 zeros;\n"
      "mov.v4.b32 zeros, {0, 0, 0, 0};\n"
      "st.shared.v4.b32 [%0], zeros;\n"
      "}\n"
      :: "r"(dst_addr)
    );
  }
}

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

__device__ inline
void mma_m16n8k16_inverse(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.col.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}


template <typename T, typename... Args>
void launch_kernel(
  T *kernel,
  int num_blocks,
  int block_size,
  int smem_size,
  cudaStream_t stream,
  Args... args) {
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  kernel<<<num_blocks, block_size, smem_size, stream>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}

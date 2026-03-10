#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Forward declarations for flash attention kernel entry points.
// Implementations are in attention_v6.cu and attention_v6_bp.cu.

void attention_v6(const nv_bfloat16 *Q,
                  const nv_bfloat16 *K,
                  const nv_bfloat16 *V,
                  nv_bfloat16 *O,
                  nv_bfloat16 *L_out,
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
                  cudaStream_t stream = 0);

void attention_v6_backward(const nv_bfloat16 *Q,
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
                            cudaStream_t stream = 0);

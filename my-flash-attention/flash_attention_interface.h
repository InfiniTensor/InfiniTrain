#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Forward declarations for flash attention kernel entry points.
// Implementations are in attention_v6.cu and attention_v6_bp.cu.

// Forward pass.
// L_out is float32 (log-sum-exp per query row).
void attention_v6(const nv_bfloat16 *Q,
                  const nv_bfloat16 *K,
                  const nv_bfloat16 *V,
                  nv_bfloat16 *O,
                  float *L_out,          // [bs, q_head, q_len], float32
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
                  float scale = -1.0f,   // <0 means use default 1/sqrt(head_dim)
                  cudaStream_t stream = 0);

// Backward pass.
// L:  float32 logsumexp from forward.
// dO: float32 upstream gradient (caller converts from bf16 if needed).
// dQ/dK/dV: float32 output gradients (caller converts to bf16 if needed).
void attention_v6_backward(const nv_bfloat16 *Q,
                            const nv_bfloat16 *K,
                            const nv_bfloat16 *V,
                            const nv_bfloat16 *O,
                            const float       *L,
                            const float       *dO,
                            float             *dQ,
                            float             *dK,
                            float             *dV,
                            int batch_size,
                            int q_head,
                            int kv_head,
                            int q_len,
                            int kv_len,
                            int head_dim,
                            bool is_causal,
                            float scale = -1.0f,  // <0 means use default 1/sqrt(head_dim)
                            cudaStream_t stream = 0);

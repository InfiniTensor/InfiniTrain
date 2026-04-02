#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace infini_train::kernels::cuda {

// Reference FP32 implementation for verification
// Naive attention without tiling or shared memory optimization
// Computes O = Softmax(Q * K^T / sqrt(D) + Mask) * V

__global__ void FlashAttentionReferenceForwardKernel(
    const float* Q, const float* K, const float* V, float* O, float* L,
    int B, int H, int T, int D, float softmax_scale, bool is_causal) {
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z * blockDim.x + threadIdx.x; // Sequence index i (query)

    if (i >= T) return;

    // Offsets
    size_t bh_offset = (size_t)(b * H + h) * T * D;
    const float* q_ptr = Q + bh_offset + i * D;
    const float* k_base = K + bh_offset;
    const float* v_base = V + bh_offset;
    float* o_ptr = O + bh_offset + i * D;
    float* l_ptr = L + (b * H + h) * T;

    // 1. Compute scores S_ij = Q_i * K_j^T
    // We can't store all scores, so we use online softmax or 2-pass.
    // Reference implementation can use 2-pass for simplicity and precision.
    // Or just online softmax.
    
    float m_i = -1e20f; // Max score
    float l_i = 0.0f;   // Sum exp
    
    // Accumulator for O_i
    // Use double for precision verification if possible, but registers are 32-bit.
    // We simulate high precision by just being careful.
    // Actually, let's use float but carefully.
    float acc_o[256]; // Max D
    for(int d=0; d<D; ++d) acc_o[d] = 0.0f;

    for (int j = 0; j < T; ++j) {
        if (is_causal && j > i) continue;

        // Dot product Q_i . K_j
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += q_ptr[d] * k_base[j * D + d];
        }
        score *= softmax_scale;

        // Online Softmax
        float m_prev = m_i;
        if (score > m_i) {
            m_i = score;
        }
        
        float p = expf(score - m_i); // This is exp(x - m_new) if we update m_i first?
        // Standard online softmax:
        // m_new = max(m_prev, x)
        // d = exp(m_prev - m_new)
        // l_new = l_prev * d + exp(x - m_new)
        // o_new = o_prev * d + v * exp(x - m_new)
        
        float d_scale = expf(m_prev - m_i);
        l_i = l_i * d_scale + p;
        
        for(int d=0; d<D; ++d) {
            acc_o[d] = acc_o[d] * d_scale + v_base[j * D + d] * p;
        }
    }

    // Finalize
    for(int d=0; d<D; ++d) {
        o_ptr[d] = acc_o[d] / l_i;
    }
    
    // LSE = m_i + log(l_i)
    l_ptr[i] = m_i + logf(l_i);
}

void FlashAttentionReferenceForward(const float* Q, const float* K, const float* V, float* O, float* L,
                                  int B, int H, int T, int D, float softmax_scale, bool is_causal, cudaStream_t stream) {
    dim3 grid(B, H, (T + 256 - 1) / 256);
    dim3 block(256);
    FlashAttentionReferenceForwardKernel<<<grid, block, 0, stream>>>(Q, K, V, O, L, B, H, T, D, softmax_scale, is_causal);
}

} // namespace

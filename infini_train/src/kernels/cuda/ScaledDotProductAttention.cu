#include <cmath>
#include <cstddef>

#include "cub/block/block_reduce.cuh"
#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

template <int BLOCK_THREADS, typename T>
__global__ void FlashAttnForwardKernel(T *__restrict__ y,       // (B,H,Tq,D)
                                       const T *__restrict__ q, // (B,H,Tq,D)
                                       const T *__restrict__ k, // (B,H,Tk,D)
                                       const T *__restrict__ v, // (B,H,Tk,D)
                                       const float *__restrict__ add_mask, int B, int H, int Tq, int Tk, int D,
                                       float scale, bool causal) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tq = blockIdx.z;
    const int tid = threadIdx.x;

    const int64_t base_qy = (((int64_t)b * H + h) * Tq + tq) * D;

    const T *q_row = q + base_qy;
    T *y_row = y + base_qy;

    float m = -INFINITY;
    float l = 0.f;

    for (int d = tid; d < D; d += BLOCK_THREADS) { y_row[d] = common::cuda::Cast<T>(0.0f); }

    for (int tk = 0; tk < Tk; ++tk) {

        float thread_dot = 0.f;
        const int64_t base_kv = (((int64_t)b * H + h) * Tk + tk) * D;
        const T *k_vec = k + base_kv;

        for (int d = tid; d < D; d += BLOCK_THREADS) {
            float qv = common::cuda::Cast<float>(q_row[d]);
            float kv = common::cuda::Cast<float>(k_vec[d]);
            thread_dot += qv * kv;
        }

        using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float dot_sum = BlockReduce(temp_storage).Sum(thread_dot);

        __shared__ float s_shared;
        if (tid == 0) {
            float s = dot_sum * scale;

            if (add_mask) {
                const int64_t idx_mask = (((int64_t)b * H + h) * Tq + tq) * (int64_t)Tk + tk;
                s += add_mask[idx_mask];
            }

            if (causal && tk > tq) {
                s = -INFINITY;
            }
            s_shared = s;
        }
        __syncthreads();

        const float s_val = s_shared;

        float m_new = fmaxf(m, s_val);
        float alpha = __expf(m - m_new);
        float p = __expf(s_val - m_new);
        float l_new = alpha * l + p;

        float coeff_y = (l_new > 0.f ? (alpha * l) / l_new : 0.f);
        float coeff_v = (l_new > 0.f ? p / l_new : 0.f);

        const T *v_vec = v + base_kv;
        for (int d = tid; d < D; d += BLOCK_THREADS) {
            float y_old = common::cuda::Cast<float>(y_row[d]);
            float vv = common::cuda::Cast<float>(v_vec[d]);
            float y_new = y_old * coeff_y + coeff_v * vv;
            y_row[d] = common::cuda::Cast<T>(y_new);
        }

        if (tid == 0) {
            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }
}

template <int BLOCK_THREADS, typename T>
void LaunchFlashAttnForward(const std::shared_ptr<Tensor> &y, const std::shared_ptr<Tensor> &q,
                            const std::shared_ptr<Tensor> &k, const std::shared_ptr<Tensor> &v,
                            const std::shared_ptr<Tensor> &attn_mask, bool is_causal, double /*dropout_p*/,
                            bool scale_has, double scale_val, bool /*enable_gqa*/) {
    const auto &qdims = q->Dims(); // (B,H,Tq,D)
    const int B = (int)qdims[0];
    const int H = (int)qdims[1];
    const int Tq = (int)qdims[2];
    const int D = (int)qdims[3];

    const auto &kdims = k->Dims(); // (B,H,Tk,D)
    const int Tk = (int)kdims[2];

    float scale = scale_has ? static_cast<float>(scale_val) : (1.0f / sqrtf((float)D));

    dim3 grid(B, H, Tq);
    dim3 block(BLOCK_THREADS);

    const float *mask_ptr = nullptr;
    if (attn_mask) {
        if (attn_mask->Dtype() != DataType::kFLOAT32) {
            LOG_LOC(FATAL, "CUDA SDPA forward: attn_mask must be float32 additive mask (0/-inf)");
        }
        mask_ptr = static_cast<const float *>(attn_mask->DataPtr());
    }

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(y->GetDevice());

    FlashAttnForwardKernel<BLOCK_THREADS, T><<<grid, block, 0, cuda_device->Stream()>>>(
        static_cast<T *>(y->DataPtr()), static_cast<const T *>(q->DataPtr()), static_cast<const T *>(k->DataPtr()),
        static_cast<const T *>(v->DataPtr()), mask_ptr, B, H, Tq, Tk, D, scale, is_causal);
}

std::shared_ptr<Tensor>
ScaledDotProductAttentionForward(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                 const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &attn_mask,
                                 bool is_causal, double dropout_p, bool scale_has, double scale_val, bool enable_gqa) {
    const auto &qdims = q->Dims(); // (B,H,Tq,D)
    auto y = std::make_shared<Tensor>(qdims, q->Dtype(), q->GetDevice());

    switch (q->Dtype()) {
        DISPATCH_CASE(WRAP(LaunchFlashAttnForward<256, float>(y, q, k, v, attn_mask, is_causal, dropout_p, scale_has,
                                                              scale_val, enable_gqa);),
                      DataType::kFLOAT32)
        DISPATCH_CASE(WRAP(LaunchFlashAttnForward<256, nv_bfloat16>(y, q, k, v, attn_mask, is_causal, dropout_p,
                                                                    scale_has, scale_val, enable_gqa);),
                      DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "CUDA SDPA forward: unsupported dtype");
    }
    return y;
}

template <int BLOCK_THREADS, typename T>
__global__ void RowwiseMaxKernel(const T *__restrict__ x, float *__restrict__ row_max, int B, int H, int Tq, int Tk) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tq = blockIdx.z;
    int64_t base = (((int64_t)b * H + h) * Tq + tq) * (int64_t)Tk;

    float local_max = -INFINITY;
    for (int tk = threadIdx.x; tk < Tk; tk += BLOCK_THREADS) {
        float v = common::cuda::Cast<float>(x[base + tk]);
        local_max = fmaxf(local_max, v);
    }
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float m = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
    if (threadIdx.x == 0) {
        row_max[((b * H + h) * Tq + tq)] = m;
    }
}

template <int BLOCK_THREADS, typename T>
__global__ void RowwiseSumExpKernel(const T *__restrict__ x, const float *__restrict__ row_max,
                                    float *__restrict__ row_sumexp, int B, int H, int Tq, int Tk) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tq = blockIdx.z;
    int64_t base = (((int64_t)b * H + h) * Tq + tq) * (int64_t)Tk;
    float m = row_max[((b * H + h) * Tq + tq)];

    float local_sum = 0.f;
    for (int tk = threadIdx.x; tk < Tk; tk += BLOCK_THREADS) {
        float v = common::cuda::Cast<float>(x[base + tk]);
        local_sum += __expf(v - m);
    }
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float s = BlockReduce(temp_storage).Reduce(local_sum, cub::Sum());
    if (threadIdx.x == 0) {
        row_sumexp[((b * H + h) * Tq + tq)] = s;
    }
}

template <int BLOCK_THREADS, typename T>
__global__ void NormalizeKernel(T *__restrict__ P, const T *__restrict__ scores, const float *__restrict__ row_max,
                                const float *__restrict__ row_sumexp, int B, int H, int Tq, int Tk) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tq = blockIdx.z;
    int64_t base = (((int64_t)b * H + h) * Tq + tq) * (int64_t)Tk;
    float m = row_max[((b * H + h) * Tq + tq)];
    float se = fmaxf(row_sumexp[((b * H + h) * Tq + tq)], 1e-20f);

    for (int tk = threadIdx.x; tk < Tk; tk += BLOCK_THREADS) {
        float v = common::cuda::Cast<float>(scores[base + tk]);
        float p = __expf(v - m) / se;
        P[base + tk] = common::cuda::Cast<T>(p);
    }
}

template <typename T> __global__ void AddCausalNegInfKernel(T *__restrict__ s, int B, int H, int Tq, int Tk) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tq = blockIdx.z;
    int64_t base = (((int64_t)b * H + h) * Tq + tq) * (int64_t)Tk;
    for (int tk = threadIdx.x; tk < Tk; tk += blockDim.x) {
        if (tk > tq) {
            float v = common::cuda::Cast<float>(s[base + tk]);
            s[base + tk] = common::cuda::Cast<T>(v + (-INFINITY));
        }
    }
}

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttentionBackward(const std::shared_ptr<Tensor> &dY, // (B,H,Tq,D)
                                                                       const std::shared_ptr<Tensor> &Q,  // (B,H,Tq,D)
                                                                       const std::shared_ptr<Tensor> &K,  // (B,H,Tk,D)
                                                                       const std::shared_ptr<Tensor> &V,  // (B,H,Tk,D)
                                                                       const std::shared_ptr<Tensor> &attn_mask,
                                                                       bool is_causal, double /*dropout_p*/,
                                                                       bool scale_has, double scale_val,
                                                                       bool /*enable_gqa*/) {

    const auto &qdims = Q->Dims(); // (B,H,Tq,D)
    const auto &kdims = K->Dims(); // (B,H,Tk,D)
    const int B = (int)qdims[0];
    const int H = (int)qdims[1];
    const int Tq = (int)qdims[2];
    const int D = (int)qdims[3];
    const int Tk = (int)kdims[2];

    const double scale = scale_has ? scale_val : (1.0 / std::sqrt((double)D));

    // 1) scores = scale * (Q @ K^T)
    std::shared_ptr<Tensor> scores = Q->Matmul(K->Transpose(-2, -1)) * (float)scale; // (B,H,Tq,Tk)

    // 2) external mask and causal mask
    if (attn_mask) {
        scores = scores + attn_mask; // allow boardcast
    }
    if (is_causal) {
        const auto *dev = dynamic_cast<const CudaDevice *>(scores->GetDevice());
        dim3 grid(B, H, Tq), block(256);
        switch (scores->Dtype()) {
            DISPATCH_CASE(WRAP(AddCausalNegInfKernel<float><<<grid, block, 0, dev->Stream()>>>(
                                   static_cast<float *>(scores->DataPtr()), B, H, Tq, Tk);),
                          DataType::kFLOAT32)
            DISPATCH_CASE(WRAP(AddCausalNegInfKernel<nv_bfloat16><<<grid, block, 0, dev->Stream()>>>(
                                   static_cast<nv_bfloat16 *>(scores->DataPtr()), B, H, Tq, Tk);),
                          DataType::kBFLOAT16)
        default:
            LOG_LOC(FATAL, "SDPA backward: unsupported dtype for causal masking");
        }
    }

    // 3) P = softmax(scores)
    auto row_max = std::make_shared<Tensor>(std::vector<int64_t>{B, H, Tq, 1}, DataType::kFLOAT32, scores->GetDevice());
    auto row_sumexp
        = std::make_shared<Tensor>(std::vector<int64_t>{B, H, Tq, 1}, DataType::kFLOAT32, scores->GetDevice());
    auto P = std::make_shared<Tensor>(scores->Dims(), scores->Dtype(), scores->GetDevice());

    const auto *dev2 = dynamic_cast<const CudaDevice *>(scores->GetDevice());
    dim3 grid(B, H, Tq), block(256);

    switch (scores->Dtype()) {
        DISPATCH_CASE(
            WRAP(RowwiseMaxKernel<256, float>
                 <<<grid, block, 0, dev2->Stream()>>>(static_cast<const float *>(scores->DataPtr()),
                                                      static_cast<float *>(row_max->DataPtr()), B, H, Tq, Tk);
                 RowwiseSumExpKernel<256, float><<<grid, block, 0, dev2->Stream()>>>(
                     static_cast<const float *>(scores->DataPtr()), static_cast<const float *>(row_max->DataPtr()),
                     static_cast<float *>(row_sumexp->DataPtr()), B, H, Tq, Tk);
                 NormalizeKernel<256, float><<<grid, block, 0, dev2->Stream()>>>(
                     static_cast<float *>(P->DataPtr()), static_cast<const float *>(scores->DataPtr()),
                     static_cast<const float *>(row_max->DataPtr()), static_cast<const float *>(row_sumexp->DataPtr()),
                     B, H, Tq, Tk);),
            DataType::kFLOAT32)
        DISPATCH_CASE(
            WRAP(RowwiseMaxKernel<256, nv_bfloat16>
                 <<<grid, block, 0, dev2->Stream()>>>(static_cast<const nv_bfloat16 *>(scores->DataPtr()),
                                                      static_cast<float *>(row_max->DataPtr()), B, H, Tq, Tk);
                 RowwiseSumExpKernel<256, nv_bfloat16>
                 <<<grid, block, 0, dev2->Stream()>>>(static_cast<const nv_bfloat16 *>(scores->DataPtr()),
                                                      static_cast<const float *>(row_max->DataPtr()),
                                                      static_cast<float *>(row_sumexp->DataPtr()), B, H, Tq, Tk);
                 NormalizeKernel<256, nv_bfloat16><<<grid, block, 0, dev2->Stream()>>>(
                     static_cast<nv_bfloat16 *>(P->DataPtr()), static_cast<const nv_bfloat16 *>(scores->DataPtr()),
                     static_cast<const float *>(row_max->DataPtr()), static_cast<const float *>(row_sumexp->DataPtr()),
                     B, H, Tq, Tk);),
            DataType::kBFLOAT16)
    default:
        LOG_LOC(FATAL, "SDPA backward: unsupported dtype for softmax");
    }

    // 4) dV = P^T @ dY
    auto dV = P->Transpose(-2, -1)->Matmul(dY); // (B,H,Tk,D)

    // 5) dP = dY @ V^T
    auto dP = dY->Matmul(V->Transpose(-2, -1)); // (B,H,Tq,Tk)

    // 6) dS = (dP - (dP ⊙ P).sum(-1, keepdim)) ⊙ P
    auto Tmul = dP * P; // (B,H,Tq,Tk)

    auto ones = std::make_shared<Tensor>(std::vector<int64_t>{Tk, 1}, Tmul->Dtype(), Tmul->GetDevice());
    ones->Fill<float>(1.0f);
    auto T2d = Tmul->View({(int64_t)B * H * Tq, (int64_t)Tk});
    auto tmp2d = T2d->Matmul(ones);        // (B*H*Tq,1)
    auto tmp = tmp2d->View({B, H, Tq, 1}); // (B,H,Tq,1)
    auto dS = (dP - tmp)->Mul(P);          // (B,H,Tq,Tk)

    // 7) dQ = (dS @ K) * scale
    auto dQ = dS->Matmul(K) * (float)scale; // (B,H,Tq,D)
    // 8) dK = (dS^T @ Q) * scale
    auto dK = dS->Transpose(-2, -1)->Matmul(Q) * (float)scale; // (B,H,Tk,D)

    return {dQ, dK, dV};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SDPA_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SDPA_KERNEL(ScaledDotProductAttentionForward);
REGISTER_CUDA_SDPA_KERNEL(ScaledDotProductAttentionBackward);

#undef REGISTER_CUDA_SDPA_KERNEL

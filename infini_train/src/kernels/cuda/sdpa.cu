#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <cub/block/block_reduce.cuh>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

namespace {
struct MaskInfo {
    const float *ptr = nullptr;
    int64_t b = 1;
    int64_t h = 1;
    int64_t t1 = 1;
    int64_t t2 = 1;
};

__host__ inline MaskInfo MakeMaskInfo(const std::shared_ptr<Tensor> &mask) {
    MaskInfo info;
    if (mask == nullptr) {
        return info;
    }
    const auto &dims = mask->Dims();
    CHECK_EQ(dims.size(), 4) << "SDPA mask must be 4D (B,H,T,T) or broadcastable";
    CHECK(mask->Dtype() == DataType::kFLOAT32) << "SDPA mask currently expects float32 (0/1)";
    info.ptr = static_cast<const float *>(mask->DataPtr());
    info.b = dims[0];
    info.h = dims[1];
    info.t1 = dims[2];
    info.t2 = dims[3];
    return info;
}

__device__ __forceinline__ bool IsMasked(const MaskInfo &m, int64_t B, int64_t H, int64_t i, int64_t j) {
    if (m.ptr == nullptr) {
        return false;
    }
    const int64_t mb = (m.b == 1) ? 0 : B;
    const int64_t mh = (m.h == 1) ? 0 : H;
    const int64_t mi = (m.t1 == 1) ? 0 : i;
    const int64_t mj = (m.t2 == 1) ? 0 : j;
    const int64_t idx = (((mb * m.h + mh) * m.t1 + mi) * m.t2 + mj);
    return m.ptr[idx] != 0.0f;
}

__device__ __forceinline__ float WarpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float WarpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

} // namespace

// q,k,v: (B,H,T,D) float32 contiguous
// out:   (B,H,T,D) float32
__global__ void SdpaForwardKernel(const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
                                 MaskInfo mask, bool is_causal, float scale, float *__restrict__ out,
                                 int64_t B, int64_t H, int64_t T, int64_t D) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t b = row / (H * T);
    const int64_t rem = row % (H * T);
    const int64_t h = rem / T;
    const int64_t i = rem % T;

    if (b >= B) {
        return;
    }

    extern __shared__ float smem[];
    float *out_acc = smem; // size D

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();

    __shared__ float warp_max[8];
    __shared__ float row_max;

    float local_max = -INFINITY;

    // Pass 1: max
    for (int64_t j = warp_id; j < T; j += num_warps) {
        if (is_causal && j > i) {
            continue;
        }
        if (IsMasked(mask, b, h, i, j)) {
            continue;
        }

        const float *q_row = q + (((b * H + h) * T + i) * D);
        const float *k_row = k + (((b * H + h) * T + j) * D);

        float dot = 0.0f;
        for (int64_t d = lane; d < D; d += 32) {
            dot += q_row[d] * k_row[d];
        }
        dot = WarpReduceSum(dot);
        if (lane == 0) {
            const float score = dot * scale;
            local_max = fmaxf(local_max, score);
        }
    }

    if (lane == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float m = -INFINITY;
        if (lane < num_warps) {
            m = warp_max[lane];
        }
        m = WarpReduceMax(m);
        if (lane == 0) {
            row_max = m;
        }
    }
    __syncthreads();

    __shared__ float warp_sum[8];
    __shared__ float row_sum;

    float local_sum = 0.0f;

    // Pass 2: sumexp + output accumulation
    for (int64_t j = warp_id; j < T; j += num_warps) {
        if (is_causal && j > i) {
            continue;
        }
        if (IsMasked(mask, b, h, i, j)) {
            continue;
        }

        const float *q_row = q + (((b * H + h) * T + i) * D);
        const float *k_row = k + (((b * H + h) * T + j) * D);
        const float *v_row = v + (((b * H + h) * T + j) * D);

        float dot = 0.0f;
        for (int64_t d = lane; d < D; d += 32) {
            dot += q_row[d] * k_row[d];
        }
        dot = WarpReduceSum(dot);

        float p = 0.0f;
        if (lane == 0) {
            const float score = dot * scale;
            p = __expf(score - row_max);
            local_sum += p;
        }
        p = __shfl_sync(0xffffffff, p, 0);

        for (int64_t d = lane; d < D; d += 32) {
            atomicAdd(&out_acc[d], p * v_row[d]);
        }
    }

    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float s = 0.0f;
        if (lane < num_warps) {
            s = warp_sum[lane];
        }
        s = WarpReduceSum(s);
        if (lane == 0) {
            row_sum = s;
        }
    }
    __syncthreads();

    const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    float *out_row = out + (((b * H + h) * T + i) * D);
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        out_row[d] = out_acc[d] * inv;
    }
}

__global__ void SdpaBackwardKernel(const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
                                  const float *__restrict__ grad_out, MaskInfo mask, bool is_causal, float scale,
                                  float *__restrict__ grad_q, float *__restrict__ grad_k, float *__restrict__ grad_v,
                                  int64_t B, int64_t H, int64_t T, int64_t D) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t b = row / (H * T);
    const int64_t rem = row % (H * T);
    const int64_t h = rem / T;
    const int64_t i = rem % T;

    if (b >= B) {
        return;
    }

    // compute row_max
    __shared__ float warp_max[8];
    __shared__ float row_max;

    float local_max = -INFINITY;
    for (int64_t j = warp_id; j < T; j += num_warps) {
        if (is_causal && j > i) {
            continue;
        }
        if (IsMasked(mask, b, h, i, j)) {
            continue;
        }
        const float *q_row = q + (((b * H + h) * T + i) * D);
        const float *k_row = k + (((b * H + h) * T + j) * D);
        float dot = 0.0f;
        for (int64_t d = lane; d < D; d += 32) {
            dot += q_row[d] * k_row[d];
        }
        dot = WarpReduceSum(dot);
        if (lane == 0) {
            local_max = fmaxf(local_max, dot * scale);
        }
    }
    if (lane == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();
    if (warp_id == 0) {
        float m = -INFINITY;
        if (lane < num_warps) {
            m = warp_max[lane];
        }
        m = WarpReduceMax(m);
        if (lane == 0) {
            row_max = m;
        }
    }
    __syncthreads();

    // row_sumexp
    __shared__ float warp_sum[8];
    __shared__ float row_sum;
    float local_sum = 0.0f;
    for (int64_t j = warp_id; j < T; j += num_warps) {
        if (is_causal && j > i) {
            continue;
        }
        if (IsMasked(mask, b, h, i, j)) {
            continue;
        }
        const float *q_row = q + (((b * H + h) * T + i) * D);
        const float *k_row = k + (((b * H + h) * T + j) * D);
        float dot = 0.0f;
        for (int64_t d = lane; d < D; d += 32) {
            dot += q_row[d] * k_row[d];
        }
        dot = WarpReduceSum(dot);
        if (lane == 0) {
            local_sum += __expf(dot * scale - row_max);
        }
    }
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        float s = 0.0f;
        if (lane < num_warps) {
            s = warp_sum[lane];
        }
        s = WarpReduceSum(s);
        if (lane == 0) {
            row_sum = s;
        }
    }
    __syncthreads();

    const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // compute dot_scalar = sum_j p_ij * (dO_i dot V_j)
    __shared__ float warp_dot[8];
    __shared__ float row_dot;
    float local_dot = 0.0f;

    const float *q_row = q + (((b * H + h) * T + i) * D);
    const float *go_row = grad_out + (((b * H + h) * T + i) * D);

    for (int64_t j = warp_id; j < T; j += num_warps) {
        if (is_causal && j > i) {
            continue;
        }
        if (IsMasked(mask, b, h, i, j)) {
            continue;
        }

        const float *k_row = k + (((b * H + h) * T + j) * D);
        const float *v_row = v + (((b * H + h) * T + j) * D);

        float dot_qk = 0.0f;
        float dot_gv = 0.0f;
        for (int64_t d = lane; d < D; d += 32) {
            dot_qk += q_row[d] * k_row[d];
            dot_gv += go_row[d] * v_row[d];
        }
        dot_qk = WarpReduceSum(dot_qk);
        dot_gv = WarpReduceSum(dot_gv);

        if (lane == 0) {
            const float p = __expf(dot_qk * scale - row_max) * inv;
            local_dot += p * dot_gv;
        }
    }

    if (lane == 0) {
        warp_dot[warp_id] = local_dot;
    }
    __syncthreads();
    if (warp_id == 0) {
        float s = 0.0f;
        if (lane < num_warps) {
            s = warp_dot[lane];
        }
        s = WarpReduceSum(s);
        if (lane == 0) {
            row_dot = s;
        }
    }
    __syncthreads();

    // grad_q init
    float *gq_row = grad_q + (((b * H + h) * T + i) * D);
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        gq_row[d] = 0.0f;
    }
    __syncthreads();

    // second pass: accumulate grads
    for (int64_t j = warp_id; j < T; j += num_warps) {
        if (is_causal && j > i) {
            continue;
        }
        if (IsMasked(mask, b, h, i, j)) {
            continue;
        }

        const float *k_row = k + (((b * H + h) * T + j) * D);
        const float *v_row = v + (((b * H + h) * T + j) * D);

        float dot_qk = 0.0f;
        float dot_gv = 0.0f;
        for (int64_t d = lane; d < D; d += 32) {
            dot_qk += q_row[d] * k_row[d];
            dot_gv += go_row[d] * v_row[d];
        }
        dot_qk = WarpReduceSum(dot_qk);
        dot_gv = WarpReduceSum(dot_gv);

        float p = 0.0f;
        float ds = 0.0f;
        if (lane == 0) {
            p = __expf(dot_qk * scale - row_max) * inv;
            ds = p * (dot_gv - row_dot);
        }
        p = __shfl_sync(0xffffffff, p, 0);
        ds = __shfl_sync(0xffffffff, ds, 0);

        // grad_v[j] += p * grad_out[i]
        float *gv_row = grad_v + (((b * H + h) * T + j) * D);
        float *gk_row = grad_k + (((b * H + h) * T + j) * D);

        for (int64_t d = lane; d < D; d += 32) {
            atomicAdd(&gv_row[d], p * go_row[d]);
            atomicAdd(&gk_row[d], ds * q_row[d] * scale);
            atomicAdd(&gq_row[d], ds * k_row[d] * scale);
        }
    }
}

std::shared_ptr<Tensor> SdpaForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                   const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                                   bool is_causal, float scale) {
    CHECK(query->Dtype() == DataType::kFLOAT32);
    CHECK(key->Dtype() == DataType::kFLOAT32);
    CHECK(value->Dtype() == DataType::kFLOAT32);

    const auto &q_dims = query->Dims();
    CHECK_EQ(q_dims.size(), 4);
    const int64_t B = q_dims[0];
    const int64_t H = q_dims[1];
    const int64_t T = q_dims[2];
    const int64_t D = q_dims[3];

    auto out = std::make_shared<Tensor>(q_dims, DataType::kFLOAT32, query->GetDevice());

    auto device = query->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const int threads = 128;
    const int64_t blocks = B * H * T;
    const size_t shmem = static_cast<size_t>(D) * sizeof(float);

    auto mask_info = MakeMaskInfo(attn_mask);

    SdpaForwardKernel<<<blocks, threads, shmem, cuda_stream>>>(static_cast<const float *>(query->DataPtr()),
                                                              static_cast<const float *>(key->DataPtr()),
                                                              static_cast<const float *>(value->DataPtr()),
                                                              mask_info, is_causal, scale,
                                                              static_cast<float *>(out->DataPtr()), B, H, T, D);
    return out;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
SdpaBackward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key, const std::shared_ptr<Tensor> &value,
            const std::shared_ptr<Tensor> &grad_out, const std::shared_ptr<Tensor> &attn_mask,
            bool is_causal, float scale) {
    CHECK(query->Dtype() == DataType::kFLOAT32);
    CHECK(key->Dtype() == DataType::kFLOAT32);
    CHECK(value->Dtype() == DataType::kFLOAT32);
    CHECK(grad_out->Dtype() == DataType::kFLOAT32);

    const auto &q_dims = query->Dims();
    CHECK_EQ(q_dims.size(), 4);
    const int64_t B = q_dims[0];
    const int64_t H = q_dims[1];
    const int64_t T = q_dims[2];
    const int64_t D = q_dims[3];

    auto grad_q = std::make_shared<Tensor>(q_dims, DataType::kFLOAT32, query->GetDevice());
    auto grad_k = std::make_shared<Tensor>(key->Dims(), DataType::kFLOAT32, key->GetDevice());
    auto grad_v = std::make_shared<Tensor>(value->Dims(), DataType::kFLOAT32, value->GetDevice());

    grad_k->Fill<float>(0.0f);
    grad_v->Fill<float>(0.0f);

    auto device = query->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const int threads = 128;
    const int64_t blocks = B * H * T;

    auto mask_info = MakeMaskInfo(attn_mask);

    SdpaBackwardKernel<<<blocks, threads, 0, cuda_stream>>>(static_cast<const float *>(query->DataPtr()),
                                                           static_cast<const float *>(key->DataPtr()),
                                                           static_cast<const float *>(value->DataPtr()),
                                                           static_cast<const float *>(grad_out->DataPtr()),
                                                           mask_info, is_causal, scale,
                                                           static_cast<float *>(grad_q->DataPtr()),
                                                           static_cast<float *>(grad_k->DataPtr()),
                                                           static_cast<float *>(grad_v->DataPtr()),
                                                           B, H, T, D);

    return {grad_q, grad_k, grad_v};
}



// Wrappers to keep REGISTER_KERNEL lines short (some environments hard-wrap long lines when writing).
static std::shared_ptr<Tensor> SdpaForwardKernelEntry(const std::shared_ptr<Tensor> &query,
                                                     const std::shared_ptr<Tensor> &key,
                                                     const std::shared_ptr<Tensor> &value,
                                                     const std::shared_ptr<Tensor> &attn_mask,
                                                     bool is_causal, float scale) {
    return infini_train::kernels::cuda::SdpaForward(query, key, value, attn_mask, is_causal, scale);
}

static std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
SdpaBackwardKernelEntry(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                        const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &grad_out,
                        const std::shared_ptr<Tensor> &attn_mask, bool is_causal, float scale) {
    return infini_train::kernels::cuda::SdpaBackward(query, key, value, grad_out, attn_mask, is_causal, scale);
}
} // namespace infini_train::kernels::cuda

using SdpaFwdEntry = decltype(&infini_train::kernels::cuda::SdpaForwardKernelEntry);
using SdpaBwdEntry = decltype(&infini_train::kernels::cuda::SdpaBackwardKernelEntry);
static SdpaFwdEntry _sdpa_fwd_entry = &infini_train::kernels::cuda::SdpaForwardKernelEntry;
static SdpaBwdEntry _sdpa_bwd_entry = &infini_train::kernels::cuda::SdpaBackwardKernelEntry;
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, SdpaForward, _sdpa_fwd_entry);
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, SdpaBackward, _sdpa_bwd_entry);

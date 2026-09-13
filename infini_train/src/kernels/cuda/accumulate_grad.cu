#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

// Aligned vector type for vectorized loads/stores (up to 128-bit). The alignment matches the payload
// size so NVCC emits the widest legal access for it (16B -> .128, 8B -> .64, 4B -> .32).
template <typename T, int N> struct __align__(sizeof(T) * N) aligned_vector { T val[N]; };

// Elements per 128-bit access: float -> 4, bf16/half -> 8, double -> 2.
template <typename T> constexpr int kVecSize = 16 / sizeof(T);

// Vector width for kernels touching two dtypes (e.g. fp32 master weight + bf16 shadow). Sizing by the
// wider element keeps both payloads inside a single 128-bit access; the narrower one then uses a
// 64/32-bit access, still one instruction instead of VecSize scalar ones.
template <typename T, typename U>
constexpr int kMixedVecSize = 16 / (sizeof(T) > sizeof(U) ? sizeof(T) : sizeof(U));

// Vectorized access also needs the payload width to divide the pointer. Optimizer state is freshly
// allocated (hence 16B-aligned), but a param that is a view into a larger buffer may not be.
inline bool IsAlignedTo(const void *ptr, size_t bytes) { return (reinterpret_cast<uintptr_t>(ptr) % bytes) == 0; }

// SM count for a device ordinal, resolved once and cached the way cuda_guard_impl.cc caches streams
// and handles. Returns 0 if the query fails or the ordinal is out of range.
inline int MultiProcessorCount(int index) {
    constexpr int kMaxGpus = 8;
    static std::array<std::atomic<int>, kMaxGpus> cache{}; // 0 == not resolved yet

    if (index < 0 || index >= kMaxGpus) {
        return 0;
    }
    int sms = cache[index].load(std::memory_order_relaxed);
    if (sms != 0) {
        return sms;
    }
    // Threads racing here query and store the same value, so no locking is needed.
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, index) != cudaSuccess || sms <= 0) {
        return 0;
    }
    cache[index].store(sms, std::memory_order_relaxed);
    return sms;
}

// Smallest tensor worth handing to the vectorized kernel. That kernel launches vec_size-times fewer
// threads and needs 46-57 registers/thread against the scalar kernel's 17-19, because every lane of
// all four payloads stays live from load to store; that caps it near half occupancy and gives its
// longer straight-line body a fixed cost of ~0.3us (vec_size 4) to ~1.3us (vec_size 8) on A100.
// Measured there it only overtakes the scalar kernel once the grid gives every SM at least one
// block, i.e. sms * threads_per_block vectors. Falls back to "one full vector" if the query failed.
inline size_t MinVectorizedElements(int device_index, int vec_size, int threads_per_block) {
    const int sms = MultiProcessorCount(device_index);
    if (sms <= 0) {
        return static_cast<size_t>(vec_size);
    }
    return static_cast<size_t>(sms) * threads_per_block * vec_size;
}

// Loop-invariant Adam scalars. beta1/beta2 are cast to T once per thread instead of once per element;
// the cast is deterministic, so the values fed to the math are unchanged.
template <typename T> struct AdamParams {
    T beta1;
    T one_minus_beta1;
    T beta2;
    T one_minus_beta2;
    float learning_rate;
    float bias_correction_m;
    float bias_correction_v;
    float eps;
};

template <typename T>
__device__ __forceinline__ AdamParams<T> MakeAdamParams(float learning_rate, float beta1, float beta2, float eps,
                                                        float bias_correction_m, float bias_correction_v) {
    return AdamParams<T>{common::cuda::Cast<T>(beta1), common::cuda::Cast<T>(1 - beta1), common::cuda::Cast<T>(beta2),
                         common::cuda::Cast<T>(1 - beta2), learning_rate, bias_correction_m, bias_correction_v, eps};
}

// One Adam update step for a single element: m/v EMAs, bias correction, then the param update.
// Shared by the scalar and vectorized kernels so both paths stay bit-identical.
template <typename T>
__device__ __forceinline__ void AdamUpdateElement(const T &grad, T &param, T &m, T &v, const AdamParams<T> &p) {
    m = common::cuda::Fma(p.beta1, m, p.one_minus_beta1 * grad);
    v = common::cuda::Fma(p.beta2, v, p.one_minus_beta2 * grad * grad);

    const float m_hat = common::cuda::Cast<float>(m) / p.bias_correction_m;
    const float v_hat = common::cuda::Cast<float>(v) / p.bias_correction_v;
    const float rcp = __frcp_rn(__fsqrt_rn(v_hat) + p.eps);

    if constexpr (std::is_same_v<T, float>) {
        // Spell out the contraction of "param - (lr * m_hat) * rcp". Left implicit, nvcc fuses it into
        // one FFMA in the scalar kernel but emits FMUL+FADD in the unrolled vectorized one, which costs
        // 1 ULP and would make the result depend on which of the two paths the dispatch picked.
        param = std::fma(-(p.learning_rate * m_hat), rcp, param);
    } else {
        param = common::cuda::Sub(param, common::cuda::Cast<T>((p.learning_rate * m_hat) * rcp));
    }
}

} // namespace

template <typename T>
__global__ void AccumulateGradKernel(const T *grad_ptr, float rate, T *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += common::cuda::Mul(grad_ptr[idx], common::cuda::Cast<T>(rate));
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    auto device = tensor->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        gradient->Dtype(),
        [=]<typename T>() {
            AccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                static_cast<const T *>(gradient->DataPtr()), rate, static_cast<T *>(tensor->DataPtr()), num_elements);
        },
        "CUDA AccumulateGrad");
}

template <typename T>
__global__ void AdamAccumulateGradKernel(const T *grad_data, T *param_data, size_t num_elements, T *m_data, T *v_data,
                                         float learning_rate, float beta1, float beta2, float eps,
                                         const float bias_correction_m, const float bias_correction_v) {
    const AdamParams<T> p = MakeAdamParams<T>(learning_rate, beta1, beta2, eps, bias_correction_m, bias_correction_v);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        AdamUpdateElement<T>(grad_data[idx], param_data[idx], m_data[idx], v_data[idx], p);
    }
}

// Vectorized Adam update: each thread handles VecSize contiguous elements through wide accesses,
// cutting load/store instruction count (and hence Long Scoreboard stalls) by VecSize on this
// DRAM-bound kernel. Numerically identical to AdamAccumulateGradKernel via AdamUpdateElement.
// The trailing num_elements % VecSize elements are finished by a scalar pass in the same launch.
template <typename T, int VecSize>
__global__ void AdamAccumulateGradKernelVectorized(const T *__restrict__ grad_data, T *__restrict__ param_data,
                                                   size_t num_elements, T *__restrict__ m_data, T *__restrict__ v_data,
                                                   float learning_rate, float beta1, float beta2, float eps,
                                                   const float bias_correction_m, const float bias_correction_v) {
    using VecT = aligned_vector<T, VecSize>;
    const AdamParams<T> p = MakeAdamParams<T>(learning_rate, beta1, beta2, eps, bias_correction_m, bias_correction_v);
    const size_t num_vecs = num_elements / VecSize;
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t vid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; vid < num_vecs; vid += grid_stride) {
        const size_t base = vid * VecSize;

        // Vectorized loads
        VecT grad_vec = *reinterpret_cast<const VecT *>(&grad_data[base]);
        VecT param_vec = *reinterpret_cast<const VecT *>(&param_data[base]);
        VecT m_vec = *reinterpret_cast<const VecT *>(&m_data[base]);
        VecT v_vec = *reinterpret_cast<const VecT *>(&v_data[base]);

#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            AdamUpdateElement<T>(grad_vec.val[i], param_vec.val[i], m_vec.val[i], v_vec.val[i], p);
        }

        // Vectorized stores
        *reinterpret_cast<VecT *>(&param_data[base]) = param_vec;
        *reinterpret_cast<VecT *>(&m_data[base]) = m_vec;
        *reinterpret_cast<VecT *>(&v_data[base]) = v_vec;
    }

    // Tail: numel % VecSize != 0
    const size_t tail_start = num_vecs * VecSize;
    for (size_t idx = tail_start + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < num_elements;
         idx += grid_stride) {
        AdamUpdateElement<T>(grad_data[idx], param_data[idx], m_data[idx], v_data[idx], p);
    }
}

template <typename T, typename TShadow>
__global__ void AdamAccumulateGradShadowKernel(const T *grad_data, T *param_data, TShadow *shadow_data,
                                               size_t num_elements, T *m_data, T *v_data, float learning_rate,
                                               float beta1, float beta2, float eps, const float bias_correction_m,
                                               const float bias_correction_v) {
    const AdamParams<T> p = MakeAdamParams<T>(learning_rate, beta1, beta2, eps, bias_correction_m, bias_correction_v);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        AdamUpdateElement<T>(grad_data[idx], param_data[idx], m_data[idx], v_data[idx], p);
        shadow_data[idx] = common::cuda::Cast<TShadow>(param_data[idx]);
    }
}

// Vectorized counterpart of AdamAccumulateGradShadowKernel. T and TShadow may differ in width, so
// VecSize is sized by the wider of the two (see kMixedVecSize): the wider dtype gets a 128-bit access
// and the narrower one a proportionally smaller but still single vector access.
template <typename T, typename TShadow, int VecSize>
__global__ void AdamAccumulateGradShadowKernelVectorized(const T *__restrict__ grad_data, T *__restrict__ param_data,
                                                         TShadow *__restrict__ shadow_data, size_t num_elements,
                                                         T *__restrict__ m_data, T *__restrict__ v_data,
                                                         float learning_rate, float beta1, float beta2, float eps,
                                                         const float bias_correction_m, const float bias_correction_v) {
    using VecT = aligned_vector<T, VecSize>;
    using VecTShadow = aligned_vector<TShadow, VecSize>;
    const AdamParams<T> p = MakeAdamParams<T>(learning_rate, beta1, beta2, eps, bias_correction_m, bias_correction_v);
    const size_t num_vecs = num_elements / VecSize;
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t vid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; vid < num_vecs; vid += grid_stride) {
        const size_t base = vid * VecSize;

        // Vectorized loads
        VecT grad_vec = *reinterpret_cast<const VecT *>(&grad_data[base]);
        VecT param_vec = *reinterpret_cast<const VecT *>(&param_data[base]);
        VecT m_vec = *reinterpret_cast<const VecT *>(&m_data[base]);
        VecT v_vec = *reinterpret_cast<const VecT *>(&v_data[base]);

        VecTShadow shadow_vec;
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            AdamUpdateElement<T>(grad_vec.val[i], param_vec.val[i], m_vec.val[i], v_vec.val[i], p);
            shadow_vec.val[i] = common::cuda::Cast<TShadow>(param_vec.val[i]);
        }

        // Vectorized stores
        *reinterpret_cast<VecT *>(&param_data[base]) = param_vec;
        *reinterpret_cast<VecT *>(&m_data[base]) = m_vec;
        *reinterpret_cast<VecT *>(&v_data[base]) = v_vec;
        *reinterpret_cast<VecTShadow *>(&shadow_data[base]) = shadow_vec;
    }

    // Tail: numel % VecSize != 0
    const size_t tail_start = num_vecs * VecSize;
    for (size_t idx = tail_start + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < num_elements;
         idx += grid_stride) {
        AdamUpdateElement<T>(grad_data[idx], param_data[idx], m_data[idx], v_data[idx], p);
        shadow_data[idx] = common::cuda::Cast<TShadow>(param_data[idx]);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    size_t num_elements = grad->NumElements();

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    int threads_per_block = 256;

    auto device = grad->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            const T *grad_ptr = static_cast<const T *>(grad->DataPtr());
            T *param_ptr = static_cast<T *>(param->DataPtr());
            T *m_ptr = static_cast<T *>(m->DataPtr());
            T *v_ptr = static_cast<T *>(v->DataPtr());

            constexpr int vec_size = kVecSize<T>;
            // Take the vectorized path only when every operand can legally serve a wide access and the
            // tensor is big enough for the vec_size-times narrower grid to still fill the device.
            // Anything else falls back to the scalar kernel, which produces identical values.
            const size_t min_elements = MinVectorizedElements(device.index(), vec_size, threads_per_block);
            const bool can_vectorize = num_elements >= min_elements
                                    && IsAlignedTo(grad_ptr, sizeof(T) * vec_size)
                                    && IsAlignedTo(param_ptr, sizeof(T) * vec_size)
                                    && IsAlignedTo(m_ptr, sizeof(T) * vec_size)
                                    && IsAlignedTo(v_ptr, sizeof(T) * vec_size);

            if (can_vectorize) {
                const size_t num_vecs = num_elements / vec_size;
                int num_blocks = (num_vecs + threads_per_block - 1) / threads_per_block;
                AdamAccumulateGradKernelVectorized<T, vec_size><<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                    grad_ptr, param_ptr, num_elements, m_ptr, v_ptr, learning_rate, beta1, beta2, eps,
                    bias_correction_m, bias_correction_v);
            } else {
                int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
                AdamAccumulateGradKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                    grad_ptr, param_ptr, num_elements, m_ptr, v_ptr, learning_rate, beta1, beta2, eps,
                    bias_correction_m, bias_correction_v);
            }
        },
        "CUDA AdamAccumulateGrad");
}

void AdamAccumulateGradShadow(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                              const std::shared_ptr<Tensor> &shadow, const std::shared_ptr<Tensor> &m,
                              const std::shared_ptr<Tensor> &v, float learning_rate, float beta1, float beta2,
                              float eps, int64_t t) {
    size_t num_elements = grad->NumElements();

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    int threads_per_block = 256;

    auto device = grad->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
                shadow->Dtype(),
                [=]<typename TShadow>() {
                    const T *grad_ptr = static_cast<const T *>(grad->DataPtr());
                    T *param_ptr = static_cast<T *>(param->DataPtr());
                    TShadow *shadow_ptr = static_cast<TShadow *>(shadow->DataPtr());
                    T *m_ptr = static_cast<T *>(m->DataPtr());
                    T *v_ptr = static_cast<T *>(v->DataPtr());

                    // T and TShadow may differ in width, so each operand is checked against its own
                    // payload width; the shadow check is what makes a narrower TShadow legal here.
                    constexpr int vec_size = kMixedVecSize<T, TShadow>;
                    const size_t min_elements = MinVectorizedElements(device.index(), vec_size, threads_per_block);
                    const bool can_vectorize = num_elements >= min_elements
                                            && IsAlignedTo(grad_ptr, sizeof(T) * vec_size)
                                            && IsAlignedTo(param_ptr, sizeof(T) * vec_size)
                                            && IsAlignedTo(m_ptr, sizeof(T) * vec_size)
                                            && IsAlignedTo(v_ptr, sizeof(T) * vec_size)
                                            && IsAlignedTo(shadow_ptr, sizeof(TShadow) * vec_size);

                    if (can_vectorize) {
                        const size_t num_vecs = num_elements / vec_size;
                        int num_blocks = (num_vecs + threads_per_block - 1) / threads_per_block;
                        AdamAccumulateGradShadowKernelVectorized<T, TShadow, vec_size>
                            <<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                                grad_ptr, param_ptr, shadow_ptr, num_elements, m_ptr, v_ptr, learning_rate, beta1,
                                beta2, eps, bias_correction_m, bias_correction_v);
                    } else {
                        int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
                        AdamAccumulateGradShadowKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(
                            grad_ptr, param_ptr, shadow_ptr, num_elements, m_ptr, v_ptr, learning_rate, beta1, beta2,
                            eps, bias_correction_m, bias_correction_v);
                    }
                },
                "CUDA AdamAccumulateGradShadow");
        },
        "CUDA AdamAccumulateGradShadow");
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGradShadow)
#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL

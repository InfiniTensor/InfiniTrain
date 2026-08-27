#include <cstdint>
#include <memory>
#include <vector>

#include "flash.h"
#include "glog/logging.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "infini_train/include/common/common.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kSequenceAlignment = 128;

struct FlashAttentionContext {
    std::shared_ptr<Tensor> q;
    std::shared_ptr<Tensor> k;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> out;
    std::shared_ptr<Tensor> softmax_lse;
    std::shared_ptr<Tensor> rng_state;
    DataType q_original_dtype = DataType::kFLOAT32;
    DataType k_original_dtype = DataType::kFLOAT32;
    DataType v_original_dtype = DataType::kFLOAT32;
    DataType flash_dtype = DataType::kBFLOAT16;
};

std::shared_ptr<Tensor> CastIfNeeded(const std::shared_ptr<Tensor> &tensor, DataType dtype) {
    return tensor->Dtype() == dtype ? tensor : std::make_shared<Tensor>(tensor->To(dtype));
}

std::shared_ptr<Tensor> CastGrad(const std::shared_ptr<Tensor> &grad, DataType original_dtype, DataType flash_dtype) {
    const auto grad_dtype = flash_dtype == DataType::kBFLOAT16 ? DataType::kFLOAT32 : original_dtype;
    return CastIfNeeded(grad, grad_dtype);
}

cudaStream_t GetCudaStream(Device device) {
    auto *stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
        infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device));
    CHECK(stream != nullptr);
    return stream->cuda_stream();
}

void CheckQKV(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k, const std::shared_ptr<Tensor> &v,
              DataType flash_dtype) {
    CHECK(q->GetDevice() == k->GetDevice());
    CHECK(q->GetDevice() == v->GetDevice());
    CHECK(q->GetDevice().IsCUDA()) << "FlashAttention backend requires CUDA tensors";
    CHECK(k->Dims() == v->Dims());
    CHECK_EQ(q->Dims().size(), 4);
    CHECK_EQ(k->Dims().size(), 4);
    CHECK(flash_dtype == DataType::kBFLOAT16 || flash_dtype == DataType::kFLOAT16)
        << "FlashAttention supports fp16/bf16 only";

    const auto &q_dims = q->Dims();
    const auto &kv_dims = k->Dims();
    CHECK_GT(q_dims[0], 0);
    CHECK_GT(q_dims[1], 0);
    CHECK_GT(q_dims[2], 0);
    CHECK_EQ(q_dims[0], kv_dims[0]);
    CHECK_EQ(q_dims[2], kv_dims[2]);
    CHECK_EQ(q_dims[3], kv_dims[3]);
    // FIXME(zbl): Extend supported head dimensions together with the AOT kernel instances in CMakeLists.txt;
    //             the runtime checks/dispatch and FLASH_ATTN_CUDA_SOURCES must remain in sync.
    CHECK(q_dims[3] == 64 || q_dims[3] == 128) << "Native FlashAttention currently supports head_dim 64/128 only";
    CHECK_GT(kv_dims[1], 0);
    CHECK_EQ(q_dims[1] % kv_dims[1], 0) << "Q heads must be divisible by KV heads for GQA/MQA";
}

void SetForwardParams(flash::Flash_fwd_params *params, const Tensor &q, const Tensor &k, const Tensor &v, Tensor *out,
                      Tensor *softmax_lse, float scale) {
    CHECK(params != nullptr);
    const auto &q_dims = q.Dims();
    const auto &kv_dims = k.Dims();
    const int64_t batch = q_dims[0];
    const int64_t q_heads = q_dims[1];
    const int64_t kv_heads = kv_dims[1];
    const int64_t seqlen = q_dims[2];
    const int64_t head_dim = q_dims[3];
    const int64_t q_batch_stride = q_heads * seqlen * head_dim;
    const int64_t kv_batch_stride = kv_heads * seqlen * head_dim;
    const int64_t head_stride = seqlen * head_dim;

    *params = {};
    params->q_ptr = const_cast<void *>(q.DataPtr());
    params->k_ptr = const_cast<void *>(k.DataPtr());
    params->v_ptr = const_cast<void *>(v.DataPtr());
    params->o_ptr = out->DataPtr();
    params->softmax_lse_ptr = softmax_lse->DataPtr();

    params->q_batch_stride = q_batch_stride;
    params->k_batch_stride = params->v_batch_stride = kv_batch_stride;
    params->o_batch_stride = q_batch_stride;
    params->q_row_stride = params->k_row_stride = params->v_row_stride = head_dim;
    params->o_row_stride = head_dim;
    params->q_head_stride = params->k_head_stride = params->v_head_stride = head_stride;
    params->o_head_stride = head_stride;

    params->b = batch;
    params->h = q_heads;
    params->h_k = kv_heads;
    params->h_h_k_ratio = q_heads / kv_heads;
    params->seqlen_q = seqlen;
    params->seqlen_k = seqlen;
    params->seqlen_q_rounded = ROUND_UP(seqlen, kSequenceAlignment);
    params->seqlen_k_rounded = ROUND_UP(seqlen, kSequenceAlignment);
    params->d = head_dim;
    params->d_rounded = head_dim;
    params->total_q = batch * seqlen;

    params->scale_softmax = scale;
    params->scale_softmax_log2 = scale * 1.4426950408889634F;
    params->p_dropout = 1.0F;
    params->p_dropout_in_uint8_t = 255;
    params->rp_dropout = 1.0F;
    params->scale_softmax_rp_dropout = scale;
    params->is_bf16 = q.Dtype() == DataType::kBFLOAT16;
    // TODO(zbl): Plumb mask type and local-window configuration through ScaledDotProductAttention.
    params->window_size_left = -1;
    params->window_size_right = 0;
    params->is_causal = true;
    params->is_seqlens_k_cumulative = true;
    params->num_splits = 1;
}

template <typename T> void RunForward(flash::Flash_fwd_params &params, cudaStream_t stream) {
    switch (params.d) {
    case 64:
        flash::run_mha_fwd_<T, 64, true>(params, stream);
        return;
    case 128:
        flash::run_mha_fwd_<T, 128, true>(params, stream);
        return;
    default:
        LOG(FATAL) << "Unsupported FlashAttention head_dim=" << params.d;
    }
}

template <typename T> void RunBackward(flash::Flash_bwd_params &params, cudaStream_t stream) {
    switch (params.d) {
    case 64:
        flash::run_mha_bwd_<T, 64, true>(params, stream);
        return;
    case 128:
        flash::run_mha_bwd_<T, 128, true>(params, stream);
        return;
    default:
        LOG(FATAL) << "Unsupported FlashAttention head_dim=" << params.d;
    }
}

void DispatchForward(flash::Flash_fwd_params &params, cudaStream_t stream) {
    if (params.is_bf16) {
        RunForward<cutlass::bfloat16_t>(params, stream);
    } else {
        RunForward<cutlass::half_t>(params, stream);
    }
}

void DispatchBackward(flash::Flash_bwd_params &params, cudaStream_t stream) {
    if (params.is_bf16) {
        RunBackward<cutlass::bfloat16_t>(params, stream);
    } else {
        RunBackward<cutlass::half_t>(params, stream);
    }
}

template <typename T> struct GqaGradConvert;

template <> struct GqaGradConvert<__half> {
    __device__ static float ToFloat(__half value) { return __half2float(value); }
    __device__ static __half FromFloat(float value) { return __float2half_rn(value); }
};

template <> struct GqaGradConvert<__nv_bfloat16> {
    __device__ static float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }
    __device__ static __nv_bfloat16 FromFloat(float value) { return __float2bfloat16_rn(value); }
};

template <typename T>
__global__ void ReduceGqaGradKernel(T *output, const T *expanded, int64_t num_elements, int64_t kv_heads,
                                    int64_t query_heads, int64_t seqlen, int64_t head_dim) {
    const int64_t output_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_idx >= num_elements) {
        return;
    }

    int64_t index = output_idx;
    const int64_t dim_idx = index % head_dim;
    index /= head_dim;
    const int64_t sequence_idx = index % seqlen;
    index /= seqlen;
    const int64_t kv_head_idx = index % kv_heads;
    const int64_t batch_idx = index / kv_heads;
    const int64_t repeats = query_heads / kv_heads;

    float sum = 0.0F;
    for (int64_t group = 0; group < repeats; ++group) {
        const int64_t query_head_idx = kv_head_idx * repeats + group;
        const int64_t expanded_idx
            = ((batch_idx * query_heads + query_head_idx) * seqlen + sequence_idx) * head_dim + dim_idx;
        sum += GqaGradConvert<T>::ToFloat(expanded[expanded_idx]);
    }
    output[output_idx] = GqaGradConvert<T>::FromFloat(sum);
}

void ReduceGqaGrad(Tensor *output, const Tensor &expanded, cudaStream_t stream) {
    CHECK(output != nullptr);
    CHECK(output->Dtype() == expanded.Dtype());
    const auto &output_dims = output->Dims();
    const auto &expanded_dims = expanded.Dims();
    CHECK_EQ(output_dims.size(), 4);
    CHECK_EQ(expanded_dims.size(), 4);
    CHECK_EQ(output_dims[0], expanded_dims[0]);
    CHECK_EQ(output_dims[2], expanded_dims[2]);
    CHECK_EQ(output_dims[3], expanded_dims[3]);
    CHECK_EQ(expanded_dims[1] % output_dims[1], 0);

    constexpr int threads = 256;
    const int64_t num_elements = output->NumElements();
    const int blocks = static_cast<int>((num_elements + threads - 1) / threads);
    if (output->Dtype() == DataType::kBFLOAT16) {
        ReduceGqaGradKernel<<<blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(output->DataPtr()), static_cast<const __nv_bfloat16 *>(expanded.DataPtr()),
            num_elements, output_dims[1], expanded_dims[1], output_dims[2], output_dims[3]);
    } else {
        ReduceGqaGradKernel<<<blocks, threads, 0, stream>>>(
            static_cast<__half *>(output->DataPtr()), static_cast<const __half *>(expanded.DataPtr()), num_elements,
            output_dims[1], expanded_dims[1], output_dims[2], output_dims[3]);
    }
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

} // namespace

std::shared_ptr<Tensor> ScaledDotProductAttentionForward(const std::shared_ptr<Tensor> &q,
                                                         const std::shared_ptr<Tensor> &k,
                                                         const std::shared_ptr<Tensor> &v, float scale,
                                                         DataType flash_dtype, std::shared_ptr<void> *opaque_ctx) {
    CHECK(opaque_ctx != nullptr);
    CheckQKV(q, k, v, flash_dtype);

    const auto device = q->GetDevice();
    infini_train::core::DeviceGuard guard(device);
    auto ctx = std::make_shared<FlashAttentionContext>();
    ctx->q_original_dtype = q->Dtype();
    ctx->k_original_dtype = k->Dtype();
    ctx->v_original_dtype = v->Dtype();
    ctx->flash_dtype = flash_dtype;
    ctx->q = CastIfNeeded(q, flash_dtype);
    ctx->k = CastIfNeeded(k, flash_dtype);
    ctx->v = CastIfNeeded(v, flash_dtype);
    auto output = std::make_shared<Tensor>(q->Dims(), flash_dtype, device);
    // Keep the forward buffer for backward without retaining the autograd output and
    // forming Function -> flash_ctx -> output -> Function.
    ctx->out = output->Detach();

    const auto &dims = q->Dims();
    ctx->softmax_lse
        = std::make_shared<Tensor>(std::vector<int64_t>{dims[0], dims[1], dims[2]}, DataType::kFLOAT32, device);
    ctx->rng_state = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kUINT64, device);
    const auto stream = GetCudaStream(device);
    CHECK_EQ(cudaMemsetAsync(ctx->rng_state->DataPtr(), 0, ctx->rng_state->SizeInBytes(), stream), cudaSuccess);

    flash::Flash_fwd_params params{};
    SetForwardParams(&params, *ctx->q, *ctx->k, *ctx->v, ctx->out.get(), ctx->softmax_lse.get(), scale);
    params.rng_state = static_cast<uint64_t *>(ctx->rng_state->DataPtr());
    DispatchForward(params, stream);

    *opaque_ctx = ctx;
    return output;
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttentionBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &,
                                  const std::shared_ptr<Tensor> &, const std::shared_ptr<Tensor> &,
                                  const std::shared_ptr<Tensor> &, float scale, std::shared_ptr<void> opaque_ctx) {
    auto ctx = std::static_pointer_cast<FlashAttentionContext>(opaque_ctx);
    CHECK(ctx != nullptr) << "Missing FlashAttention forward context";

    const auto device = ctx->q->GetDevice();
    infini_train::core::DeviceGuard guard(device);
    auto dout = CastIfNeeded(grad_output, ctx->flash_dtype);
    CHECK(dout->Dims() == ctx->out->Dims());

    auto dq = std::make_shared<Tensor>(ctx->q->Dims(), ctx->flash_dtype, device);
    auto dk = std::make_shared<Tensor>(ctx->k->Dims(), ctx->flash_dtype, device);
    auto dv = std::make_shared<Tensor>(ctx->v->Dims(), ctx->flash_dtype, device);
    const auto &dims = ctx->q->Dims();
    const int64_t batch = dims[0];
    const int64_t heads = dims[1];
    const int64_t seqlen = dims[2];
    const int64_t head_dim = dims[3];
    const int64_t seqlen_rounded = ROUND_UP(seqlen, kSequenceAlignment);

    auto softmax_d
        = std::make_shared<Tensor>(std::vector<int64_t>{batch, heads, seqlen_rounded}, DataType::kFLOAT32, device);
    auto dq_accum = std::make_shared<Tensor>(std::vector<int64_t>{batch, seqlen_rounded, heads, head_dim},
                                             DataType::kFLOAT32, device);
    const bool use_gqa = ctx->q->Dims()[1] != ctx->k->Dims()[1];
    auto dk_kernel = use_gqa ? std::make_shared<Tensor>(ctx->q->Dims(), ctx->flash_dtype, device) : dk;
    auto dv_kernel = use_gqa ? std::make_shared<Tensor>(ctx->q->Dims(), ctx->flash_dtype, device) : dv;

    flash::Flash_bwd_params params{};
    SetForwardParams(&params, *ctx->q, *ctx->k, *ctx->v, ctx->out.get(), ctx->softmax_lse.get(), scale);
    const int64_t batch_stride = heads * seqlen * head_dim;
    const int64_t head_stride = seqlen * head_dim;
    params.do_ptr = dout->DataPtr();
    params.dq_ptr = dq->DataPtr();
    params.dk_ptr = dk_kernel->DataPtr();
    params.dv_ptr = dv_kernel->DataPtr();
    params.do_batch_stride = params.dq_batch_stride = params.dk_batch_stride = params.dv_batch_stride = batch_stride;
    params.do_row_stride = params.dq_row_stride = params.dk_row_stride = params.dv_row_stride = head_dim;
    params.do_head_stride = params.dq_head_stride = params.dk_head_stride = params.dv_head_stride = head_stride;
    params.dq_accum_ptr = dq_accum->DataPtr();
    params.dsoftmax_sum = softmax_d->DataPtr();
    params.rng_state = static_cast<uint64_t *>(ctx->rng_state->DataPtr());
    params.deterministic = false;
    params.dq_accum_split_stride = 0;

    const auto stream = GetCudaStream(device);
    DispatchBackward(params, stream);
    if (use_gqa) {
        ReduceGqaGrad(dk.get(), *dk_kernel, stream);
        ReduceGqaGrad(dv.get(), *dv_kernel, stream);
    }

    // FIXME(zbl): Forward autocast currently uses raw Tensor::To conversions and wires the Function directly
    //             to the original autograd graph. Without an autograd cast-backward edge or generic mixed-dtype
    //             grad normalization, native BF16 FlashAttention grads can be accumulated together with the FP32
    //             grads propagated by Matmul/Linear backward paths. FlashAttention backward kernel performs in
    //             BF16, so we manually promote them at the end of kernel to keep backward consistent with the
    //             current forward autocast behavior. The proper fix belongs in autograd: once autocast and grad
    //             accumulation preserve type semantics centrally, return grads in its native input dtype.
    return {CastGrad(dq, ctx->q_original_dtype, ctx->flash_dtype),
            CastGrad(dk, ctx->k_original_dtype, ctx->flash_dtype),
            CastGrad(dv, ctx->v_original_dtype, ctx->flash_dtype)};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FLASH_ATTENTION_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FLASH_ATTENTION_KERNEL(ScaledDotProductAttentionForward)
REGISTER_CUDA_FLASH_ATTENTION_KERNEL(ScaledDotProductAttentionBackward)

#undef REGISTER_CUDA_FLASH_ATTENTION_KERNEL

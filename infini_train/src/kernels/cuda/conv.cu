#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/cuda/common/gemm.cuh"

namespace infini_train::kernels::cuda {
namespace {

constexpr int64_t kMaxWorkspaceBytes = 64LL * 1024 * 1024;  // 64 MiB

cudaStream_t GetCudaStream(const Device &device) {
  return dynamic_cast<infini_train::core::cuda::CudaStream *>(
             infini_train::core::GetDeviceGuardImpl(device.type())
                 ->GetStream(device))
      ->cuda_stream();
}

// Per-thread, per-device reusable CUDA workspace buffer to avoid repeated
// cudaMallocAsync/cudaFreeAsync overhead on every Conv2d invocation.
void *GetReusableWorkspace(const Device &device, int slot, size_t bytes) {
  struct SlotBuf {
    void *ptr = nullptr;
    size_t capacity = 0;
    int dev_idx = -1;
    ~SlotBuf() {
      if (ptr != nullptr) {
        int prev_dev = 0;
        cudaGetDevice(&prev_dev);
        cudaSetDevice(dev_idx);
        cudaFree(ptr);
        cudaSetDevice(prev_dev);
      }
    }
  };
  thread_local std::unordered_map<int64_t, SlotBuf> cache;
  const int64_t key =
      (static_cast<int64_t>(device.index()) << 8) | static_cast<int64_t>(slot);
  auto &entry = cache[key];
  if (bytes > entry.capacity) {
    if (entry.ptr != nullptr) {
      CUDA_CHECK(cudaFree(entry.ptr));
      entry.ptr = nullptr;
      entry.capacity = 0;
    }
    const size_t alloc_bytes = std::max<size_t>(bytes, 1024 * 1024);
    CUDA_CHECK(cudaMalloc(&entry.ptr, alloc_bytes));
    entry.capacity = alloc_bytes;
    entry.dev_idx = device.index();
  }
  return entry.ptr;
}

#ifdef USE_CUDNN

#define CUDNN_CHECK(call)                                  \
  do {                                                     \
    cudnnStatus_t status = (call);                         \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS)                 \
        << "cuDNN error: " << cudnnGetErrorString(status); \
  } while (0)

cudnnHandle_t GetCudnnHandle(const Device &device, cudaStream_t stream) {
  struct HandleHolder {
    cudnnHandle_t handle = nullptr;
    int dev_idx = -1;
    ~HandleHolder() {
      if (handle != nullptr) {
        int prev_dev = 0;
        cudaGetDevice(&prev_dev);
        cudaSetDevice(dev_idx);
        cudnnDestroy(handle);
        cudaSetDevice(prev_dev);
      }
    }
  };
  thread_local std::unordered_map<int, HandleHolder> handles;
  auto &holder = handles[device.index()];
  if (holder.handle == nullptr) {
    CUDNN_CHECK(cudnnCreate(&holder.handle));
    holder.dev_idx = device.index();
  }
  CUDNN_CHECK(cudnnSetStream(holder.handle, stream));
  return holder.handle;
}

cudnnDataType_t ToCudnnDataType(DataType dt) {
  switch (dt) {
    case DataType::kFLOAT32:
      return CUDNN_DATA_FLOAT;
    case DataType::kFLOAT16:
      return CUDNN_DATA_HALF;
    case DataType::kBFLOAT16:
      return CUDNN_DATA_BFLOAT16;
    default:
      LOG(FATAL) << "Unsupported cuDNN DataType: " << static_cast<int>(dt);
      return CUDNN_DATA_FLOAT;
  }
}

struct CudnnConvKey {
  int dev_idx;
  int dtype;
  int N, C_in, H_in, W_in, C_out, kH, kW, sH, sW, pH, pW, groups;

  bool operator==(const CudnnConvKey &o) const {
    return dev_idx == o.dev_idx && dtype == o.dtype && N == o.N &&
           C_in == o.C_in && H_in == o.H_in && W_in == o.W_in &&
           C_out == o.C_out && kH == o.kH && kW == o.kW && sH == o.sH &&
           sW == o.sW && pH == o.pH && pW == o.pW && groups == o.groups;
  }
};

struct CudnnConvKeyHash {
  size_t operator()(const CudnnConvKey &k) const {
    size_t h = 1469598103934665603ULL;
    auto mix = [&](int v) {
      h ^= static_cast<size_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(k.dev_idx);
    mix(k.dtype);
    mix(k.N);
    mix(k.C_in);
    mix(k.H_in);
    mix(k.W_in);
    mix(k.C_out);
    mix(k.kH);
    mix(k.kW);
    mix(k.sH);
    mix(k.sW);
    mix(k.pH);
    mix(k.pW);
    mix(k.groups);
    return h;
  }
};

struct CudnnConvPlan {
  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnFilterDescriptor_t w_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnnConvolutionFwdAlgo_t fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t fwd_ws = 0;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo =
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  size_t bwd_data_ws = 0;
  cudnnConvolutionBwdFilterAlgo_t bwd_filt_algo =
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  size_t bwd_filt_ws = 0;

  ~CudnnConvPlan() {
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
    if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
    if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
  }
};

const CudnnConvPlan &GetOrCreateCudnnPlan(cudnnHandle_t handle,
                                          const Device &device, DataType dtype,
                                          int N, int C_in, int H_in, int W_in,
                                          int C_out, int H_out, int W_out,
                                          int kH, int kW, int sH, int sW,
                                          int pH, int pW, int groups) {
  thread_local std::unordered_map<CudnnConvKey, std::unique_ptr<CudnnConvPlan>,
                                  CudnnConvKeyHash>
      plan_cache;
  const CudnnConvKey key{device.index(),
                         static_cast<int>(dtype),
                         N,
                         C_in,
                         H_in,
                         W_in,
                         C_out,
                         kH,
                         kW,
                         sH,
                         sW,
                         pH,
                         pW,
                         groups};
  auto it = plan_cache.find(key);
  if (it != plan_cache.end()) {
    return *it->second;
  }

  auto plan = std::make_unique<CudnnConvPlan>();
  const cudnnDataType_t cdt = ToCudnnDataType(dtype);
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&plan->x_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(plan->x_desc, CUDNN_TENSOR_NCHW, cdt,
                                         N, C_in, H_in, W_in));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&plan->y_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(plan->y_desc, CUDNN_TENSOR_NCHW, cdt,
                                         N, C_out, H_out, W_out));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&plan->w_desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(plan->w_desc, cdt, CUDNN_TENSOR_NCHW,
                                         C_out, C_in / groups, kH, kW));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&plan->conv_desc));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(plan->conv_desc, pH, pW, sH, sW,
                                              1, 1, CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(plan->conv_desc, groups));
  // Strict FP32 FMA (no TF32 truncation) for exact numerical alignment.
  CUDNN_CHECK(cudnnSetConvolutionMathType(
      plan->conv_desc,
      dtype == DataType::kFLOAT32 ? CUDNN_FMA_MATH : CUDNN_TENSOR_OP_MATH));

  constexpr size_t kMaxCudnnWs = 128ULL * 1024 * 1024;
  {
    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t perfs[8];
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, plan->x_desc, plan->w_desc, plan->conv_desc, plan->y_desc, 8,
        &returned, perfs));
    for (int i = 0; i < returned; ++i) {
      if (perfs[i].status == CUDNN_STATUS_SUCCESS &&
          perfs[i].memory <= kMaxCudnnWs) {
        plan->fwd_algo = perfs[i].algo;
        plan->fwd_ws = perfs[i].memory;
        break;
      }
    }
  }
  {
    int returned = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perfs[8];
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, plan->w_desc, plan->y_desc, plan->conv_desc, plan->x_desc, 8,
        &returned, perfs));
    bool found = false;
    for (int pass = 0; pass < 2 && !found; ++pass) {
      for (int i = 0; i < returned; ++i) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS &&
            perfs[i].memory <= kMaxCudnnWs &&
            (pass == 1 || perfs[i].determinism == CUDNN_DETERMINISTIC)) {
          plan->bwd_data_algo = perfs[i].algo;
          plan->bwd_data_ws = perfs[i].memory;
          found = true;
          break;
        }
      }
    }
  }
  {
    int returned = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perfs[8];
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle, plan->x_desc, plan->y_desc, plan->conv_desc, plan->w_desc, 8,
        &returned, perfs));
    bool found = false;
    for (int pass = 0; pass < 2 && !found; ++pass) {
      for (int i = 0; i < returned; ++i) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS &&
            perfs[i].memory <= kMaxCudnnWs &&
            (pass == 1 || perfs[i].determinism == CUDNN_DETERMINISTIC)) {
          plan->bwd_filt_algo = perfs[i].algo;
          plan->bwd_filt_ws = perfs[i].memory;
          found = true;
          break;
        }
      }
    }
  }

  auto &ref = *plan;
  plan_cache.emplace(key, std::move(plan));
  return ref;
}

#endif  // USE_CUDNN

// ============================================================================
// Direct Depthwise Conv2d Kernels (32-bit fast indexing)
// ============================================================================

template <typename T>
__global__ void DepthwiseConv2dForwardKernel(
    const T *__restrict__ input, const T *__restrict__ weight,
    const T *__restrict__ bias, T *__restrict__ output, int N, int C, int H_in,
    int W_in, int H_out, int W_out, int kH, int kW, int sH, int sW, int pH,
    int pW) {
  const int total = N * C * H_out * W_out;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int ow = idx % W_out;
    int tmp = idx / W_out;
    const int oh = tmp % H_out;
    tmp /= H_out;
    const int c = tmp % C;
    const int n = tmp / C;

    const T *x_nc = input + (n * C + c) * (H_in * W_in);
    const T *w_c = weight + c * (kH * kW);
    float acc = bias ? common::cuda::Cast<float>(bias[c]) : 0.0f;

    const int ih0 = oh * sH - pH;
    const int iw0 = ow * sW - pW;
    for (int kh = 0; kh < kH; ++kh) {
      const int ih = ih0 + kh;
      if (static_cast<unsigned>(ih) < static_cast<unsigned>(H_in)) {
        const T *x_row = x_nc + ih * W_in;
        const T *w_row = w_c + kh * kW;
        for (int kw = 0; kw < kW; ++kw) {
          const int iw = iw0 + kw;
          if (static_cast<unsigned>(iw) < static_cast<unsigned>(W_in)) {
            acc += common::cuda::Cast<float>(x_row[iw]) *
                   common::cuda::Cast<float>(w_row[kw]);
          }
        }
      }
    }
    output[idx] = common::cuda::Cast<T>(acc);
  }
}

template <typename T>
__global__ void DepthwiseConv2dBackwardInputKernel(
    const T *__restrict__ weight, const T *__restrict__ grad_out,
    T *__restrict__ grad_in, int N, int C, int H_in, int W_in, int H_out,
    int W_out, int kH, int kW, int sH, int sW, int pH, int pW) {
  const int total = N * C * H_in * W_in;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int iw = idx % W_in;
    int tmp = idx / W_in;
    const int ih = tmp % H_in;
    tmp /= H_in;
    const int c = tmp % C;
    const int n = tmp / C;

    const T *go_nc = grad_out + (n * C + c) * (H_out * W_out);
    const T *w_c = weight + c * (kH * kW);
    float acc = 0.0f;

    for (int kh = 0; kh < kH; ++kh) {
      const int num_h = ih + pH - kh;
      if (num_h < 0 || (sH != 1 && (num_h % sH != 0))) {
        continue;
      }
      const int oh = (sH == 1) ? num_h : (num_h / sH);
      if (oh >= H_out) {
        continue;
      }
      const T *go_row = go_nc + oh * W_out;
      const T *w_row = w_c + kh * kW;
      for (int kw = 0; kw < kW; ++kw) {
        const int num_w = iw + pW - kw;
        if (num_w < 0 || (sW != 1 && (num_w % sW != 0))) {
          continue;
        }
        const int ow = (sW == 1) ? num_w : (num_w / sW);
        if (ow >= W_out) {
          continue;
        }
        acc += common::cuda::Cast<float>(go_row[ow]) *
               common::cuda::Cast<float>(w_row[kw]);
      }
    }
    grad_in[idx] = common::cuda::Cast<T>(acc);
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void DepthwiseConv2dBackwardWeightKernel(
    const T *__restrict__ input, const T *__restrict__ grad_out,
    T *__restrict__ grad_weight, int N, int C, int H_in, int W_in, int H_out,
    int W_out, int kH, int kW, int sH, int sW, int pH, int pW) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int w_idx = blockIdx.x;
  const int kw = w_idx % kW;
  const int tmp = w_idx / kW;
  const int kh = tmp % kH;
  const int c = tmp / kH;

  const int HW_out = H_out * W_out;
  const int M = N * HW_out;
  float sum = 0.0f;

  for (int i = threadIdx.x; i < M; i += BLOCK_SIZE) {
    const int n = i / HW_out;
    const int hw = i % HW_out;
    const int oh = hw / W_out;
    const int ow = hw % W_out;
    const int ih = oh * sH - pH + kh;
    const int iw = ow * sW - pW + kw;
    if (static_cast<unsigned>(ih) < static_cast<unsigned>(H_in) &&
        static_cast<unsigned>(iw) < static_cast<unsigned>(W_in)) {
      const float x_val = common::cuda::Cast<float>(
          input[((n * C + c) * H_in + ih) * W_in + iw]);
      const float go_val =
          common::cuda::Cast<float>(grad_out[(n * C + c) * HW_out + hw]);
      sum += x_val * go_val;
    }
  }
  const float reduced = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    grad_weight[w_idx] = common::cuda::Cast<T>(reduced);
  }
}

// ============================================================================
// Batched Chunked Im2Col / Col2Im & Pack / Unpack Kernels (32-bit fast math)
// ============================================================================

template <typename T>
__global__ void BatchedIm2ColKernel(const T *__restrict__ input_chunk,
                                    T *__restrict__ col, int B, int C_in,
                                    int C_in_g, int groups, int H_in, int W_in,
                                    int H_out, int W_out, int kH, int kW,
                                    int sH, int sW, int pH, int pW) {
  const int K = C_in_g * kH * kW;
  const int HW_out = H_out * W_out;
  const int B_HW = B * HW_out;
  const int total = groups * K * B_HW;
  const int stride = gridDim.x * blockDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int col_x = idx % B_HW;
    int tmp = idx / B_HW;
    const int k_idx = tmp % K;
    const int g = tmp / K;

    const int b = col_x / HW_out;
    const int hw = col_x % HW_out;
    const int oh = hw / W_out;
    const int ow = hw % W_out;

    const int kw = k_idx % kW;
    const int k_tmp = k_idx / kW;
    const int kh = k_tmp % kH;
    const int c_local = k_tmp / kH;
    const int c = g * C_in_g + c_local;

    const int ih = oh * sH - pH + kh;
    const int iw = ow * sW - pW + kw;

    T val = common::cuda::Cast<T>(0.0f);
    if (static_cast<unsigned>(ih) < static_cast<unsigned>(H_in) &&
        static_cast<unsigned>(iw) < static_cast<unsigned>(W_in)) {
      val = input_chunk[((b * C_in + c) * H_in + ih) * W_in + iw];
    }
    col[idx] = val;
  }
}

template <typename T>
__global__ void BatchedCol2ImGatherKernel(const T *__restrict__ dcol,
                                          T *__restrict__ grad_input_chunk,
                                          int B, int C_in, int C_in_g, int H_in,
                                          int W_in, int H_out, int W_out,
                                          int kH, int kW, int sH, int sW,
                                          int pH, int pW) {
  const int K = C_in_g * kH * kW;
  const int HW_out = H_out * W_out;
  const int B_HW = B * HW_out;
  const int total = B * C_in * H_in * W_in;
  const int stride = gridDim.x * blockDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int iw = idx % W_in;
    int tmp = idx / W_in;
    const int ih = tmp % H_in;
    tmp /= H_in;
    const int c = tmp % C_in;
    const int b = tmp / C_in;

    const int g = c / C_in_g;
    const int c_local = c % C_in_g;
    const T *dcol_g = dcol + static_cast<int64_t>(g) * K * B_HW;

    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
      const int num_h = ih + pH - kh;
      if (num_h < 0 || (sH != 1 && (num_h % sH != 0))) {
        continue;
      }
      const int oh = (sH == 1) ? num_h : (num_h / sH);
      if (oh >= H_out) {
        continue;
      }
      for (int kw = 0; kw < kW; ++kw) {
        const int num_w = iw + pW - kw;
        if (num_w < 0 || (sW != 1 && (num_w % sW != 0))) {
          continue;
        }
        const int ow = (sW == 1) ? num_w : (num_w / sW);
        if (ow >= W_out) {
          continue;
        }
        const int k_idx = (c_local * kH + kh) * kW + kw;
        const int col_x = b * HW_out + oh * W_out + ow;
        acc += common::cuda::Cast<float>(
            dcol_g[static_cast<int64_t>(k_idx) * B_HW + col_x]);
      }
    }
    grad_input_chunk[idx] = common::cuda::Cast<T>(acc);
  }
}

template <typename T>
__global__ void UnpackOutAndAddBiasKernel(const T *__restrict__ out_packed,
                                          const T *__restrict__ bias,
                                          T *__restrict__ out_chunk, int B,
                                          int C_out, int HW_out) {
  const int B_HW = B * HW_out;
  const int total = B * C_out * HW_out;
  const int stride = gridDim.x * blockDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int hw = idx % HW_out;
    const int tmp = idx / HW_out;
    const int c = tmp % C_out;
    const int b = tmp / C_out;
    float v = common::cuda::Cast<float>(
        out_packed[static_cast<int64_t>(c) * B_HW + b * HW_out + hw]);
    if (bias != nullptr) {
      v += common::cuda::Cast<float>(bias[c]);
    }
    out_chunk[idx] = common::cuda::Cast<T>(v);
  }
}

template <typename T>
__global__ void PackNCHWToCNHWKernel(const T *__restrict__ src,
                                     T *__restrict__ dst, int B, int C,
                                     int HW) {
  const int B_HW = B * HW;
  const int total = B * C * HW;
  const int stride = gridDim.x * blockDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int hw = idx % HW;
    const int tmp = idx / HW;
    const int c = tmp % C;
    const int b = tmp / C;
    dst[static_cast<int64_t>(c) * B_HW + b * HW + hw] = src[idx];
  }
}

template <typename T>
__global__ void AddBiasNCHWKernel(T *__restrict__ output,
                                  const T *__restrict__ bias, int N, int C,
                                  int HW) {
  const int total = N * C * HW;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int c = (idx / HW) % C;
    output[idx] = common::cuda::Cast<T>(common::cuda::Cast<float>(output[idx]) +
                                        common::cuda::Cast<float>(bias[c]));
  }
}

template <int BLOCK_SIZE, typename T>
__global__ void Conv2dBackwardBiasKernel(const T *__restrict__ grad_out,
                                         T *__restrict__ grad_bias, int N,
                                         int C_out, int HW_out) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int c = blockIdx.x;
  if (c >= C_out) {
    return;
  }
  const int M = N * HW_out;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < M; i += BLOCK_SIZE) {
    const int n = i / HW_out;
    const int hw = i % HW_out;
    sum += common::cuda::Cast<float>(grad_out[(n * C_out + c) * HW_out + hw]);
  }
  const float reduced = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    grad_bias[c] = common::cuda::Cast<T>(reduced);
  }
}

int64_t ComputeChunkBatch(int64_t N, int64_t elems_per_sample,
                          size_t elem_size) {
  const int64_t bytes_per_sample =
      std::max<int64_t>(1, elems_per_sample * static_cast<int64_t>(elem_size));
  const int64_t max_b =
      std::max<int64_t>(1, kMaxWorkspaceBytes / bytes_per_sample);
  return std::min<int64_t>(N, max_b);
}

}  // namespace

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input,
                                      const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t pad_h, int64_t pad_w,
                                      int64_t groups) {
  CHECK_EQ(input->Dims().size(), 4);
  CHECK_EQ(weight->Dims().size(), 4);

  const int64_t N = input->Dims()[0];
  const int64_t C_in = input->Dims()[1];
  const int64_t H_in = input->Dims()[2];
  const int64_t W_in = input->Dims()[3];
  const int64_t C_out = weight->Dims()[0];
  const int64_t C_in_g = weight->Dims()[1];
  const int64_t kH = weight->Dims()[2];
  const int64_t kW = weight->Dims()[3];

  CHECK_EQ(C_in, C_in_g * groups);
  CHECK_EQ(C_out % groups, 0);
  const int64_t C_out_g = C_out / groups;
  const int64_t H_out = (H_in + 2 * pad_h - kH) / stride_h + 1;
  const int64_t W_out = (W_in + 2 * pad_w - kW) / stride_w + 1;
  CHECK_GT(H_out, 0);
  CHECK_GT(W_out, 0);

  auto device = input->GetDevice();
  auto dtype = input->Dtype();
  auto output = std::make_shared<Tensor>(
      std::vector<int64_t>{N, C_out, H_out, W_out}, dtype, device);
  const auto stream = GetCudaStream(device);
  const int threads = 256;
  const int64_t HW_out = H_out * W_out;

#ifdef USE_CUDNN
  {
    cudnnHandle_t handle = GetCudnnHandle(device, stream);
    const auto &plan = GetOrCreateCudnnPlan(
        handle, device, dtype, static_cast<int>(N), static_cast<int>(C_in),
        static_cast<int>(H_in), static_cast<int>(W_in), static_cast<int>(C_out),
        static_cast<int>(H_out), static_cast<int>(W_out), static_cast<int>(kH),
        static_cast<int>(kW), static_cast<int>(stride_h),
        static_cast<int>(stride_w), static_cast<int>(pad_h),
        static_cast<int>(pad_w), static_cast<int>(groups));
    void *ws = plan.fwd_ws > 0 ? GetReusableWorkspace(device, 2, plan.fwd_ws)
                               : nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        handle, &alpha, plan.x_desc, input->DataPtr(), plan.w_desc,
        weight->DataPtr(), plan.conv_desc, plan.fwd_algo, ws, plan.fwd_ws,
        &beta, plan.y_desc, output->DataPtr()));
    if (bias) {
      const int64_t total = N * C_out * HW_out;
      const int blocks = static_cast<int>(
          std::min<int64_t>((total + threads - 1) / threads, 65535));
      core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
          dtype,
          [=]<typename T>() {
            AddBiasNCHWKernel<<<blocks, threads, 0, stream>>>(
                static_cast<T *>(output->DataPtr()),
                static_cast<const T *>(bias->DataPtr()), static_cast<int>(N),
                static_cast<int>(C_out), static_cast<int>(HW_out));
          },
          "CUDA Conv2dAddBiasCudnn");
    }
    return output;
  }
#endif

  // Fast Path 1: Depthwise Conv2d (groups == C_in && C_out == C_in)
  if (groups == C_in && C_out == C_in) {
    const int64_t total = N * C_out * HW_out;
    const int blocks = static_cast<int>(
        std::min<int64_t>((total + threads - 1) / threads, 65535));
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
          const T *b_ptr =
              bias ? static_cast<const T *>(bias->DataPtr()) : nullptr;
          DepthwiseConv2dForwardKernel<<<blocks, threads, 0, stream>>>(
              static_cast<const T *>(input->DataPtr()),
              static_cast<const T *>(weight->DataPtr()), b_ptr,
              static_cast<T *>(output->DataPtr()), static_cast<int>(N),
              static_cast<int>(C_in), static_cast<int>(H_in),
              static_cast<int>(W_in), static_cast<int>(H_out),
              static_cast<int>(W_out), static_cast<int>(kH),
              static_cast<int>(kW), static_cast<int>(stride_h),
              static_cast<int>(stride_w), static_cast<int>(pad_h),
              static_cast<int>(pad_w));
        },
        "CUDA DepthwiseConv2dForward");
    return output;
  }

  // Fast Path 2: Pointwise 1x1 Conv2d (stride=1, pad=0, groups=1) -> zero
  // im2col!
  if (kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 &&
      pad_w == 0 && groups == 1) {
    Dispatcher::Instance().Call<void>(
        {device.type(), "Gemm"}, device,
        GemmParams{
            .trans_a = GemmTranspose::kNoTranspose,
            .trans_b = GemmTranspose::kNoTranspose,
            .m = static_cast<int>(HW_out),
            .n = static_cast<int>(C_out),
            .k = static_cast<int>(C_in),
            .A = input->DataPtr(),
            .lda = static_cast<int>(HW_out),
            .B = weight->DataPtr(),
            .ldb = static_cast<int>(C_in),
            .C = output->DataPtr(),
            .ldc = static_cast<int>(HW_out),
            .alpha = 1.0f,
            .beta = 0.0f,
            .batch_count = static_cast<int>(N),
            .stride_a = N == 1 ? 0LL : static_cast<long long>(C_in * HW_out),
            .stride_b = 0LL,
            .stride_c = N == 1 ? 0LL : static_cast<long long>(C_out * HW_out),
            .input_dtype = dtype,
            .output_dtype = dtype,
        });
    if (bias) {
      const int64_t total = N * C_out * HW_out;
      const int blocks = static_cast<int>(
          std::min<int64_t>((total + threads - 1) / threads, 65535));
      core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
          dtype,
          [=]<typename T>() {
            AddBiasNCHWKernel<<<blocks, threads, 0, stream>>>(
                static_cast<T *>(output->DataPtr()),
                static_cast<const T *>(bias->DataPtr()), static_cast<int>(N),
                static_cast<int>(C_out), static_cast<int>(HW_out));
          },
          "CUDA Conv2dAddBias");
    }
    return output;
  }

  // General Path: Chunked Batched Im2Col + cuBLAS Strided-Batched GEMM
  const int64_t K = C_in_g * kH * kW;
  const size_t elem_size = kDataTypeToSize.at(dtype);
  const int64_t chunk_b = ComputeChunkBatch(N, groups * K * HW_out, elem_size);

  void *col_ptr = GetReusableWorkspace(
      device, 0,
      static_cast<size_t>(groups * K * chunk_b * HW_out) * elem_size);
  void *out_packed_ptr = GetReusableWorkspace(
      device, 1, static_cast<size_t>(C_out * chunk_b * HW_out) * elem_size);

  const uint8_t *in_base = static_cast<const uint8_t *>(input->DataPtr());
  uint8_t *out_base = static_cast<uint8_t *>(output->DataPtr());
  const size_t in_sample_bytes =
      static_cast<size_t>(C_in * H_in * W_in) * elem_size;
  const size_t out_sample_bytes =
      static_cast<size_t>(C_out * HW_out) * elem_size;

  for (int64_t n0 = 0; n0 < N; n0 += chunk_b) {
    const int64_t B = std::min<int64_t>(chunk_b, N - n0);
    const int64_t B_HW = B * HW_out;
    const void *in_chunk_ptr =
        in_base + static_cast<size_t>(n0) * in_sample_bytes;
    void *out_chunk_ptr = out_base + static_cast<size_t>(n0) * out_sample_bytes;

    const int64_t col_total = groups * K * B_HW;
    const int col_blocks = static_cast<int>(
        std::min<int64_t>((col_total + threads - 1) / threads, 65535));
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
          BatchedIm2ColKernel<<<col_blocks, threads, 0, stream>>>(
              static_cast<const T *>(in_chunk_ptr), static_cast<T *>(col_ptr),
              static_cast<int>(B), static_cast<int>(C_in),
              static_cast<int>(C_in_g), static_cast<int>(groups),
              static_cast<int>(H_in), static_cast<int>(W_in),
              static_cast<int>(H_out), static_cast<int>(W_out),
              static_cast<int>(kH), static_cast<int>(kW),
              static_cast<int>(stride_h), static_cast<int>(stride_w),
              static_cast<int>(pad_h), static_cast<int>(pad_w));
        },
        "CUDA BatchedIm2Col");

    void *gemm_c_ptr = (B == 1) ? out_chunk_ptr : out_packed_ptr;
    Dispatcher::Instance().Call<void>(
        {device.type(), "Gemm"}, device,
        GemmParams{
            .trans_a = GemmTranspose::kNoTranspose,
            .trans_b = GemmTranspose::kNoTranspose,
            .m = static_cast<int>(B_HW),
            .n = static_cast<int>(C_out_g),
            .k = static_cast<int>(K),
            .A = col_ptr,
            .lda = static_cast<int>(B_HW),
            .B = weight->DataPtr(),
            .ldb = static_cast<int>(K),
            .C = gemm_c_ptr,
            .ldc = static_cast<int>(B_HW),
            .alpha = 1.0f,
            .beta = 0.0f,
            .batch_count = static_cast<int>(groups),
            .stride_a = groups == 1 ? 0LL : static_cast<long long>(K * B_HW),
            .stride_b = groups == 1 ? 0LL : static_cast<long long>(C_out_g * K),
            .stride_c =
                groups == 1 ? 0LL : static_cast<long long>(C_out_g * B_HW),
            .input_dtype = dtype,
            .output_dtype = dtype,
        });

    if (B == 1) {
      if (bias) {
        const int64_t out_total = C_out * HW_out;
        const int out_blocks = static_cast<int>(
            std::min<int64_t>((out_total + threads - 1) / threads, 65535));
        core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
            dtype,
            [=]<typename T>() {
              AddBiasNCHWKernel<<<out_blocks, threads, 0, stream>>>(
                  static_cast<T *>(out_chunk_ptr),
                  static_cast<const T *>(bias->DataPtr()), 1,
                  static_cast<int>(C_out), static_cast<int>(HW_out));
            },
            "CUDA Conv2dAddBiasSingle");
      }
    } else {
      const int64_t out_total = B * C_out * HW_out;
      const int out_blocks = static_cast<int>(
          std::min<int64_t>((out_total + threads - 1) / threads, 65535));
      core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
          dtype,
          [=]<typename T>() {
            const T *b_ptr =
                bias ? static_cast<const T *>(bias->DataPtr()) : nullptr;
            UnpackOutAndAddBiasKernel<<<out_blocks, threads, 0, stream>>>(
                static_cast<const T *>(out_packed_ptr), b_ptr,
                static_cast<T *>(out_chunk_ptr), static_cast<int>(B),
                static_cast<int>(C_out), static_cast<int>(HW_out));
          },
          "CUDA UnpackOutAndAddBias");
    }
  }
  return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(
    const std::shared_ptr<Tensor> &weight,
    const std::shared_ptr<Tensor> &grad_output,
    const std::vector<int64_t> &input_dims, int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w, int64_t groups) {
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(weight->Dims().size(), 4);

  const int64_t N = input_dims[0];
  const int64_t C_in = input_dims[1];
  const int64_t H_in = input_dims[2];
  const int64_t W_in = input_dims[3];
  const int64_t C_out = weight->Dims()[0];
  const int64_t C_in_g = weight->Dims()[1];
  const int64_t kH = weight->Dims()[2];
  const int64_t kW = weight->Dims()[3];
  const int64_t C_out_g = C_out / groups;
  const int64_t H_out = grad_output->Dims()[2];
  const int64_t W_out = grad_output->Dims()[3];
  const int64_t HW_out = H_out * W_out;

  auto device = grad_output->GetDevice();
  auto dtype = grad_output->Dtype();
  auto grad_input = std::make_shared<Tensor>(input_dims, dtype, device);
  const auto stream = GetCudaStream(device);
  const int threads = 256;

#ifdef USE_CUDNN
  {
    cudnnHandle_t handle = GetCudnnHandle(device, stream);
    const auto &plan = GetOrCreateCudnnPlan(
        handle, device, dtype, static_cast<int>(N), static_cast<int>(C_in),
        static_cast<int>(H_in), static_cast<int>(W_in), static_cast<int>(C_out),
        static_cast<int>(H_out), static_cast<int>(W_out), static_cast<int>(kH),
        static_cast<int>(kW), static_cast<int>(stride_h),
        static_cast<int>(stride_w), static_cast<int>(pad_h),
        static_cast<int>(pad_w), static_cast<int>(groups));
    void *ws = plan.bwd_data_ws > 0
                   ? GetReusableWorkspace(device, 2, plan.bwd_data_ws)
                   : nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle, &alpha, plan.w_desc, weight->DataPtr(), plan.y_desc,
        grad_output->DataPtr(), plan.conv_desc, plan.bwd_data_algo, ws,
        plan.bwd_data_ws, &beta, plan.x_desc, grad_input->DataPtr()));
    return grad_input;
  }
#endif

  // Fast Path 1: Depthwise Conv2d
  if (groups == C_in && C_out == C_in) {
    const int64_t total = N * C_in * H_in * W_in;
    const int blocks = static_cast<int>(
        std::min<int64_t>((total + threads - 1) / threads, 65535));
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
          DepthwiseConv2dBackwardInputKernel<<<blocks, threads, 0, stream>>>(
              static_cast<const T *>(weight->DataPtr()),
              static_cast<const T *>(grad_output->DataPtr()),
              static_cast<T *>(grad_input->DataPtr()), static_cast<int>(N),
              static_cast<int>(C_in), static_cast<int>(H_in),
              static_cast<int>(W_in), static_cast<int>(H_out),
              static_cast<int>(W_out), static_cast<int>(kH),
              static_cast<int>(kW), static_cast<int>(stride_h),
              static_cast<int>(stride_w), static_cast<int>(pad_h),
              static_cast<int>(pad_w));
        },
        "CUDA DepthwiseConv2dBackwardInput");
    return grad_input;
  }

  // Fast Path 2: Pointwise 1x1 Conv2d (stride=1, pad=0, groups=1)
  if (kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 &&
      pad_w == 0 && groups == 1) {
    Dispatcher::Instance().Call<void>(
        {device.type(), "Gemm"}, device,
        GemmParams{
            .trans_a = GemmTranspose::kNoTranspose,
            .trans_b = GemmTranspose::kTranspose,
            .m = static_cast<int>(HW_out),
            .n = static_cast<int>(C_in),
            .k = static_cast<int>(C_out),
            .A = grad_output->DataPtr(),
            .lda = static_cast<int>(HW_out),
            .B = weight->DataPtr(),
            .ldb = static_cast<int>(C_in),
            .C = grad_input->DataPtr(),
            .ldc = static_cast<int>(HW_out),
            .alpha = 1.0f,
            .beta = 0.0f,
            .batch_count = static_cast<int>(N),
            .stride_a = N == 1 ? 0LL : static_cast<long long>(C_out * HW_out),
            .stride_b = 0LL,
            .stride_c = N == 1 ? 0LL : static_cast<long long>(C_in * HW_out),
            .input_dtype = dtype,
            .output_dtype = dtype,
        });
    return grad_input;
  }

  // General Path: Chunked Batched GEMM + Deterministic Gather Col2Im
  const int64_t K = C_in_g * kH * kW;
  const size_t elem_size = kDataTypeToSize.at(dtype);
  const int64_t chunk_b = ComputeChunkBatch(N, groups * K * HW_out, elem_size);

  void *dcol_ptr = GetReusableWorkspace(
      device, 0,
      static_cast<size_t>(groups * K * chunk_b * HW_out) * elem_size);
  void *go_packed_ptr = GetReusableWorkspace(
      device, 1, static_cast<size_t>(C_out * chunk_b * HW_out) * elem_size);

  const uint8_t *go_base = static_cast<const uint8_t *>(grad_output->DataPtr());
  uint8_t *gi_base = static_cast<uint8_t *>(grad_input->DataPtr());
  const size_t go_sample_bytes =
      static_cast<size_t>(C_out * HW_out) * elem_size;
  const size_t gi_sample_bytes =
      static_cast<size_t>(C_in * H_in * W_in) * elem_size;

  for (int64_t n0 = 0; n0 < N; n0 += chunk_b) {
    const int64_t B = std::min<int64_t>(chunk_b, N - n0);
    const int64_t B_HW = B * HW_out;
    const void *go_chunk_ptr =
        go_base + static_cast<size_t>(n0) * go_sample_bytes;
    void *gi_chunk_ptr = gi_base + static_cast<size_t>(n0) * gi_sample_bytes;

    const void *gemm_a_ptr = go_chunk_ptr;
    if (B > 1) {
      const int64_t pack_total = B * C_out * HW_out;
      const int pack_blocks = static_cast<int>(
          std::min<int64_t>((pack_total + threads - 1) / threads, 65535));
      core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
          dtype,
          [=]<typename T>() {
            PackNCHWToCNHWKernel<<<pack_blocks, threads, 0, stream>>>(
                static_cast<const T *>(go_chunk_ptr),
                static_cast<T *>(go_packed_ptr), static_cast<int>(B),
                static_cast<int>(C_out), static_cast<int>(HW_out));
          },
          "CUDA PackGradOut");
      gemm_a_ptr = go_packed_ptr;
    }

    Dispatcher::Instance().Call<void>(
        {device.type(), "Gemm"}, device,
        GemmParams{
            .trans_a = GemmTranspose::kNoTranspose,
            .trans_b = GemmTranspose::kTranspose,
            .m = static_cast<int>(B_HW),
            .n = static_cast<int>(K),
            .k = static_cast<int>(C_out_g),
            .A = gemm_a_ptr,
            .lda = static_cast<int>(B_HW),
            .B = weight->DataPtr(),
            .ldb = static_cast<int>(K),
            .C = dcol_ptr,
            .ldc = static_cast<int>(B_HW),
            .alpha = 1.0f,
            .beta = 0.0f,
            .batch_count = static_cast<int>(groups),
            .stride_a =
                groups == 1 ? 0LL : static_cast<long long>(C_out_g * B_HW),
            .stride_b = groups == 1 ? 0LL : static_cast<long long>(C_out_g * K),
            .stride_c = groups == 1 ? 0LL : static_cast<long long>(K * B_HW),
            .input_dtype = dtype,
            .output_dtype = dtype,
        });

    const int64_t gi_total = B * C_in * H_in * W_in;
    const int gi_blocks = static_cast<int>(
        std::min<int64_t>((gi_total + threads - 1) / threads, 65535));
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
          BatchedCol2ImGatherKernel<<<gi_blocks, threads, 0, stream>>>(
              static_cast<const T *>(dcol_ptr), static_cast<T *>(gi_chunk_ptr),
              static_cast<int>(B), static_cast<int>(C_in),
              static_cast<int>(C_in_g), static_cast<int>(H_in),
              static_cast<int>(W_in), static_cast<int>(H_out),
              static_cast<int>(W_out), static_cast<int>(kH),
              static_cast<int>(kW), static_cast<int>(stride_h),
              static_cast<int>(stride_w), static_cast<int>(pad_h),
              static_cast<int>(pad_w));
        },
        "CUDA BatchedCol2ImGather");
  }
  return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(
    const std::shared_ptr<Tensor> &input,
    const std::shared_ptr<Tensor> &grad_output,
    const std::vector<int64_t> &weight_dims, int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w, int64_t groups) {
  CHECK_EQ(input->Dims().size(), 4);
  CHECK_EQ(weight_dims.size(), 4);

  const int64_t N = input->Dims()[0];
  const int64_t C_in = input->Dims()[1];
  const int64_t H_in = input->Dims()[2];
  const int64_t W_in = input->Dims()[3];
  const int64_t C_out = weight_dims[0];
  const int64_t C_in_g = weight_dims[1];
  const int64_t kH = weight_dims[2];
  const int64_t kW = weight_dims[3];
  const int64_t C_out_g = C_out / groups;
  const int64_t H_out = grad_output->Dims()[2];
  const int64_t W_out = grad_output->Dims()[3];
  const int64_t HW_out = H_out * W_out;

  auto device = grad_output->GetDevice();
  auto dtype = grad_output->Dtype();
  auto grad_weight = std::make_shared<Tensor>(weight_dims, dtype, device);
  const auto stream = GetCudaStream(device);
  const int threads = 256;

#ifdef USE_CUDNN
  {
    cudnnHandle_t handle = GetCudnnHandle(device, stream);
    const auto &plan = GetOrCreateCudnnPlan(
        handle, device, dtype, static_cast<int>(N), static_cast<int>(C_in),
        static_cast<int>(H_in), static_cast<int>(W_in), static_cast<int>(C_out),
        static_cast<int>(H_out), static_cast<int>(W_out), static_cast<int>(kH),
        static_cast<int>(kW), static_cast<int>(stride_h),
        static_cast<int>(stride_w), static_cast<int>(pad_h),
        static_cast<int>(pad_w), static_cast<int>(groups));
    void *ws = plan.bwd_filt_ws > 0
                   ? GetReusableWorkspace(device, 2, plan.bwd_filt_ws)
                   : nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        handle, &alpha, plan.x_desc, input->DataPtr(), plan.y_desc,
        grad_output->DataPtr(), plan.conv_desc, plan.bwd_filt_algo, ws,
        plan.bwd_filt_ws, &beta, plan.w_desc, grad_weight->DataPtr()));
    return grad_weight;
  }
#endif

  // Fast Path 1: Depthwise Conv2d
  if (groups == C_in && C_out == C_in) {
    constexpr int kBlockSize = 256;
    const int num_blocks = static_cast<int>(C_in * kH * kW);
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
          DepthwiseConv2dBackwardWeightKernel<kBlockSize>
              <<<num_blocks, kBlockSize, 0, stream>>>(
                  static_cast<const T *>(input->DataPtr()),
                  static_cast<const T *>(grad_output->DataPtr()),
                  static_cast<T *>(grad_weight->DataPtr()), static_cast<int>(N),
                  static_cast<int>(C_in), static_cast<int>(H_in),
                  static_cast<int>(W_in), static_cast<int>(H_out),
                  static_cast<int>(W_out), static_cast<int>(kH),
                  static_cast<int>(kW), static_cast<int>(stride_h),
                  static_cast<int>(stride_w), static_cast<int>(pad_h),
                  static_cast<int>(pad_w));
        },
        "CUDA DepthwiseConv2dBackwardWeight");
    return grad_weight;
  }

  // Fast Path 2: Pointwise 1x1 Conv2d (stride=1, pad=0, groups=1) -> single
  // GEMM
  if (kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 &&
      pad_w == 0 && groups == 1) {
    const int64_t N_HW = N * HW_out;
    const size_t elem_size = kDataTypeToSize.at(dtype);
    const void *in_packed_ptr = input->DataPtr();
    const void *go_packed_ptr = grad_output->DataPtr();
    if (N > 1) {
      void *in_ws = GetReusableWorkspace(
          device, 0, static_cast<size_t>(C_in * N_HW) * elem_size);
      void *go_ws = GetReusableWorkspace(
          device, 1, static_cast<size_t>(C_out * N_HW) * elem_size);
      const int in_blocks = static_cast<int>(std::min<int64_t>(
          (N * C_in * HW_out + threads - 1) / threads, 65535));
      const int go_blocks = static_cast<int>(std::min<int64_t>(
          (N * C_out * HW_out + threads - 1) / threads, 65535));
      core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
          dtype,
          [=]<typename T>() {
            PackNCHWToCNHWKernel<<<in_blocks, threads, 0, stream>>>(
                static_cast<const T *>(input->DataPtr()),
                static_cast<T *>(in_ws), static_cast<int>(N),
                static_cast<int>(C_in), static_cast<int>(HW_out));
            PackNCHWToCNHWKernel<<<go_blocks, threads, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()),
                static_cast<T *>(go_ws), static_cast<int>(N),
                static_cast<int>(C_out), static_cast<int>(HW_out));
          },
          "CUDA Pack1x1BackwardWeight");
      in_packed_ptr = in_ws;
      go_packed_ptr = go_ws;
    }
    Dispatcher::Instance().Call<void>(
        {device.type(), "Gemm"}, device,
        GemmParams{
            .trans_a = GemmTranspose::kTranspose,
            .trans_b = GemmTranspose::kNoTranspose,
            .m = static_cast<int>(C_in),
            .n = static_cast<int>(C_out),
            .k = static_cast<int>(N_HW),
            .A = in_packed_ptr,
            .lda = static_cast<int>(N_HW),
            .B = go_packed_ptr,
            .ldb = static_cast<int>(N_HW),
            .C = grad_weight->DataPtr(),
            .ldc = static_cast<int>(C_in),
            .alpha = 1.0f,
            .beta = 0.0f,
            .batch_count = 1,
            .stride_a = 0LL,
            .stride_b = 0LL,
            .stride_c = 0LL,
            .input_dtype = dtype,
            .output_dtype = dtype,
        });
    return grad_weight;
  }

  // General Path: Chunked Batched Im2Col + Accumulating cuBLAS GEMM
  const int64_t K = C_in_g * kH * kW;
  const size_t elem_size = kDataTypeToSize.at(dtype);
  const int64_t chunk_b = ComputeChunkBatch(N, groups * K * HW_out, elem_size);

  void *col_ptr = GetReusableWorkspace(
      device, 0,
      static_cast<size_t>(groups * K * chunk_b * HW_out) * elem_size);
  void *go_packed_ptr = GetReusableWorkspace(
      device, 1, static_cast<size_t>(C_out * chunk_b * HW_out) * elem_size);

  const uint8_t *in_base = static_cast<const uint8_t *>(input->DataPtr());
  const uint8_t *go_base = static_cast<const uint8_t *>(grad_output->DataPtr());
  const size_t in_sample_bytes =
      static_cast<size_t>(C_in * H_in * W_in) * elem_size;
  const size_t go_sample_bytes =
      static_cast<size_t>(C_out * HW_out) * elem_size;

  for (int64_t n0 = 0; n0 < N; n0 += chunk_b) {
    const int64_t B = std::min<int64_t>(chunk_b, N - n0);
    const int64_t B_HW = B * HW_out;
    const void *in_chunk_ptr =
        in_base + static_cast<size_t>(n0) * in_sample_bytes;
    const void *go_chunk_ptr =
        go_base + static_cast<size_t>(n0) * go_sample_bytes;

    const int64_t col_total = groups * K * B_HW;
    const int col_blocks = static_cast<int>(
        std::min<int64_t>((col_total + threads - 1) / threads, 65535));
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
          BatchedIm2ColKernel<<<col_blocks, threads, 0, stream>>>(
              static_cast<const T *>(in_chunk_ptr), static_cast<T *>(col_ptr),
              static_cast<int>(B), static_cast<int>(C_in),
              static_cast<int>(C_in_g), static_cast<int>(groups),
              static_cast<int>(H_in), static_cast<int>(W_in),
              static_cast<int>(H_out), static_cast<int>(W_out),
              static_cast<int>(kH), static_cast<int>(kW),
              static_cast<int>(stride_h), static_cast<int>(stride_w),
              static_cast<int>(pad_h), static_cast<int>(pad_w));
        },
        "CUDA BatchedIm2ColBackwardWeight");

    const void *gemm_b_ptr = go_chunk_ptr;
    if (B > 1) {
      const int64_t pack_total = B * C_out * HW_out;
      const int pack_blocks = static_cast<int>(
          std::min<int64_t>((pack_total + threads - 1) / threads, 65535));
      core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
          dtype,
          [=]<typename T>() {
            PackNCHWToCNHWKernel<<<pack_blocks, threads, 0, stream>>>(
                static_cast<const T *>(go_chunk_ptr),
                static_cast<T *>(go_packed_ptr), static_cast<int>(B),
                static_cast<int>(C_out), static_cast<int>(HW_out));
          },
          "CUDA PackGradOutWeight");
      gemm_b_ptr = go_packed_ptr;
    }

    Dispatcher::Instance().Call<void>(
        {device.type(), "Gemm"}, device,
        GemmParams{
            .trans_a = GemmTranspose::kTranspose,
            .trans_b = GemmTranspose::kNoTranspose,
            .m = static_cast<int>(K),
            .n = static_cast<int>(C_out_g),
            .k = static_cast<int>(B_HW),
            .A = col_ptr,
            .lda = static_cast<int>(B_HW),
            .B = gemm_b_ptr,
            .ldb = static_cast<int>(B_HW),
            .C = grad_weight->DataPtr(),
            .ldc = static_cast<int>(K),
            .alpha = 1.0f,
            .beta = (n0 == 0) ? 0.0f : 1.0f,
            .batch_count = static_cast<int>(groups),
            .stride_a = groups == 1 ? 0LL : static_cast<long long>(K * B_HW),
            .stride_b =
                groups == 1 ? 0LL : static_cast<long long>(C_out_g * B_HW),
            .stride_c = groups == 1 ? 0LL : static_cast<long long>(C_out_g * K),
            .input_dtype = dtype,
            .output_dtype = dtype,
        });
  }
  return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(
    const std::shared_ptr<Tensor> &grad_output) {
  CHECK_EQ(grad_output->Dims().size(), 4);
  const int64_t N = grad_output->Dims()[0];
  const int64_t C_out = grad_output->Dims()[1];
  const int64_t HW_out = grad_output->Dims()[2] * grad_output->Dims()[3];

  auto device = grad_output->GetDevice();
  auto dtype = grad_output->Dtype();
  auto grad_bias =
      std::make_shared<Tensor>(std::vector<int64_t>{C_out}, dtype, device);
  const auto stream = GetCudaStream(device);

  constexpr int kBlockSize = 256;
  core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
      dtype,
      [=]<typename T>() {
        Conv2dBackwardBiasKernel<kBlockSize>
            <<<static_cast<int>(C_out), kBlockSize, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()),
                static_cast<T *>(grad_bias->DataPtr()), static_cast<int>(N),
                static_cast<int>(C_out), static_cast<int>(HW_out));
      },
      "CUDA Conv2dBackwardBias");
  return grad_bias;
}

}  // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CONV_KERNEL(kernel_name)                          \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, \
                  infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CONV_KERNEL(Conv2dForward)
REGISTER_CUDA_CONV_KERNEL(Conv2dBackwardInput)
REGISTER_CUDA_CONV_KERNEL(Conv2dBackwardWeight)
REGISTER_CUDA_CONV_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CUDA_CONV_KERNEL

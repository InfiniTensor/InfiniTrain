#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {

inline int64_t AdaptiveStartIndex(int64_t out_idx, int64_t out_size,
                                  int64_t in_size) {
  return (out_idx * in_size) / out_size;
}

inline int64_t AdaptiveEndIndex(int64_t out_idx, int64_t out_size,
                                int64_t in_size) {
  return ((out_idx + 1) * in_size + out_size - 1) / out_size;
}

}  // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MaxPool2dForward(
    const std::shared_ptr<Tensor> &input, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w) {
  CHECK_EQ(input->Dims().size(), 4);
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  const int64_t N = input->Dims()[0];
  const int64_t C = input->Dims()[1];
  const int64_t H = input->Dims()[2];
  const int64_t W = input->Dims()[3];

  const int64_t H_out = (H + 2 * pad_h - kernel_h) / stride_h + 1;
  const int64_t W_out = (W + 2 * pad_w - kernel_w) / stride_w + 1;
  CHECK_GT(H_out, 0);
  CHECK_GT(W_out, 0);

  auto output = std::make_shared<Tensor>(
      std::vector<int64_t>{N, C, H_out, W_out}, DataType::kFLOAT32);
  auto indices = std::make_shared<Tensor>(
      std::vector<int64_t>{N, C, H_out, W_out}, DataType::kINT64);

  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  float *out_ptr = static_cast<float *>(output->DataPtr());
  int64_t *idx_ptr = static_cast<int64_t *>(indices->DataPtr());

#pragma omp parallel for collapse(2)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *in_nc = in_ptr + (n * C + c) * H * W;
      float *out_nc = out_ptr + (n * C + c) * H_out * W_out;
      int64_t *idx_nc = idx_ptr + (n * C + c) * H_out * W_out;
      for (int64_t oh = 0; oh < H_out; ++oh) {
        const int64_t h_start = std::max<int64_t>(0, oh * stride_h - pad_h);
        const int64_t h_end =
            std::min<int64_t>(H, oh * stride_h - pad_h + kernel_h);
        for (int64_t ow = 0; ow < W_out; ++ow) {
          const int64_t w_start = std::max<int64_t>(0, ow * stride_w - pad_w);
          const int64_t w_end =
              std::min<int64_t>(W, ow * stride_w - pad_w + kernel_w);
          float max_val = -std::numeric_limits<float>::infinity();
          int64_t max_idx = h_start * W + w_start;
          for (int64_t ih = h_start; ih < h_end; ++ih) {
            for (int64_t iw = w_start; iw < w_end; ++iw) {
              const float v = in_nc[ih * W + iw];
              if (v > max_val) {
                max_val = v;
                max_idx = ih * W + iw;
              }
            }
          }
          out_nc[oh * W_out + ow] = max_val;
          idx_nc[oh * W_out + ow] = max_idx;
        }
      }
    }
  }
  return {output, indices};
}

std::shared_ptr<Tensor> MaxPool2dBackward(
    const std::shared_ptr<Tensor> &grad_output,
    const std::shared_ptr<Tensor> &indices,
    const std::vector<int64_t> &input_dims) {
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(static_cast<int>(grad_output->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  const int64_t N = input_dims[0];
  const int64_t C = input_dims[1];
  const int64_t H = input_dims[2];
  const int64_t W = input_dims[3];
  const int64_t H_out = grad_output->Dims()[2];
  const int64_t W_out = grad_output->Dims()[3];

  auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
  grad_input->Fill(0.0f);

  const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
  const int64_t *idx_ptr = static_cast<const int64_t *>(indices->DataPtr());
  float *gi_ptr = static_cast<float *>(grad_input->DataPtr());

#pragma omp parallel for collapse(2)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *go_nc = go_ptr + (n * C + c) * H_out * W_out;
      const int64_t *idx_nc = idx_ptr + (n * C + c) * H_out * W_out;
      float *gi_nc = gi_ptr + (n * C + c) * H * W;
      for (int64_t i = 0; i < H_out * W_out; ++i) {
        gi_nc[idx_nc[i]] += go_nc[i];
      }
    }
  }
  return grad_input;
}

std::shared_ptr<Tensor> AdaptiveAvgPool2dForward(
    const std::shared_ptr<Tensor> &input, int64_t out_h, int64_t out_w) {
  CHECK_EQ(input->Dims().size(), 4);
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  const int64_t N = input->Dims()[0];
  const int64_t C = input->Dims()[1];
  const int64_t H = input->Dims()[2];
  const int64_t W = input->Dims()[3];
  CHECK_GT(out_h, 0);
  CHECK_GT(out_w, 0);

  auto output = std::make_shared<Tensor>(
      std::vector<int64_t>{N, C, out_h, out_w}, DataType::kFLOAT32);
  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  float *out_ptr = static_cast<float *>(output->DataPtr());

#pragma omp parallel for collapse(2)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *in_nc = in_ptr + (n * C + c) * H * W;
      float *out_nc = out_ptr + (n * C + c) * out_h * out_w;
      for (int64_t oh = 0; oh < out_h; ++oh) {
        const int64_t h0 = AdaptiveStartIndex(oh, out_h, H);
        const int64_t h1 = AdaptiveEndIndex(oh, out_h, H);
        for (int64_t ow = 0; ow < out_w; ++ow) {
          const int64_t w0 = AdaptiveStartIndex(ow, out_w, W);
          const int64_t w1 = AdaptiveEndIndex(ow, out_w, W);
          double sum = 0.0;
          for (int64_t ih = h0; ih < h1; ++ih) {
            for (int64_t iw = w0; iw < w1; ++iw) {
              sum += static_cast<double>(in_nc[ih * W + iw]);
            }
          }
          const int64_t count = (h1 - h0) * (w1 - w0);
          out_nc[oh * out_w + ow] = static_cast<float>(sum / count);
        }
      }
    }
  }
  return output;
}

std::shared_ptr<Tensor> AdaptiveAvgPool2dBackward(
    const std::shared_ptr<Tensor> &grad_output,
    const std::vector<int64_t> &input_dims) {
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(static_cast<int>(grad_output->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  const int64_t N = input_dims[0];
  const int64_t C = input_dims[1];
  const int64_t H = input_dims[2];
  const int64_t W = input_dims[3];
  const int64_t out_h = grad_output->Dims()[2];
  const int64_t out_w = grad_output->Dims()[3];

  auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
  grad_input->Fill(0.0f);

  const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
  float *gi_ptr = static_cast<float *>(grad_input->DataPtr());

#pragma omp parallel for collapse(2)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float *go_nc = go_ptr + (n * C + c) * out_h * out_w;
      float *gi_nc = gi_ptr + (n * C + c) * H * W;
      for (int64_t oh = 0; oh < out_h; ++oh) {
        const int64_t h0 = AdaptiveStartIndex(oh, out_h, H);
        const int64_t h1 = AdaptiveEndIndex(oh, out_h, H);
        for (int64_t ow = 0; ow < out_w; ++ow) {
          const int64_t w0 = AdaptiveStartIndex(ow, out_w, W);
          const int64_t w1 = AdaptiveEndIndex(ow, out_w, W);
          const float scale = go_nc[oh * out_w + ow] /
                              static_cast<float>((h1 - h0) * (w1 - w0));
          for (int64_t ih = h0; ih < h1; ++ih) {
            for (int64_t iw = w0; iw < w1; ++iw) {
              gi_nc[ih * W + iw] += scale;
            }
          }
        }
      }
    }
  }
  return grad_input;
}

}  // namespace infini_train::kernels::cpu

#define REGISTER_CPU_POOLING_KERNEL(kernel_name)                       \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, \
                  infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_POOLING_KERNEL(MaxPool2dForward)
REGISTER_CPU_POOLING_KERNEL(MaxPool2dBackward)
REGISTER_CPU_POOLING_KERNEL(AdaptiveAvgPool2dForward)
REGISTER_CPU_POOLING_KERNEL(AdaptiveAvgPool2dBackward)

#undef REGISTER_CPU_POOLING_KERNEL

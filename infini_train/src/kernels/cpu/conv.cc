#include <stdint.h>

#include <cstring>
#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {

using RowMajorMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void Im2ColSingleGroup(const float *img_g, int64_t C_in_g, int64_t H_in,
                       int64_t W_in, int64_t kH, int64_t kW, int64_t sH,
                       int64_t sW, int64_t pH, int64_t pW, int64_t H_out,
                       int64_t W_out, float *col) {
  const int64_t HW_out = H_out * W_out;
  for (int64_t c = 0; c < C_in_g; ++c) {
    const float *img_c = img_g + c * H_in * W_in;
    for (int64_t kh = 0; kh < kH; ++kh) {
      for (int64_t kw = 0; kw < kW; ++kw) {
        float *col_row = col + ((c * kH + kh) * kW + kw) * HW_out;
        for (int64_t oh = 0; oh < H_out; ++oh) {
          const int64_t ih = oh * sH - pH + kh;
          if (ih < 0 || ih >= H_in) {
            std::memset(col_row + oh * W_out, 0,
                        static_cast<size_t>(W_out) * sizeof(float));
            continue;
          }
          const float *img_row = img_c + ih * W_in;
          for (int64_t ow = 0; ow < W_out; ++ow) {
            const int64_t iw = ow * sW - pW + kw;
            col_row[oh * W_out + ow] =
                (iw >= 0 && iw < W_in) ? img_row[iw] : 0.0f;
          }
        }
      }
    }
  }
}

void Col2ImSingleGroup(const float *col, int64_t C_in_g, int64_t H_in,
                       int64_t W_in, int64_t kH, int64_t kW, int64_t sH,
                       int64_t sW, int64_t pH, int64_t pW, int64_t H_out,
                       int64_t W_out, float *img_g) {
  const int64_t HW_out = H_out * W_out;
  std::memset(img_g, 0,
              static_cast<size_t>(C_in_g * H_in * W_in) * sizeof(float));
  for (int64_t c = 0; c < C_in_g; ++c) {
    float *img_c = img_g + c * H_in * W_in;
    for (int64_t kh = 0; kh < kH; ++kh) {
      for (int64_t kw = 0; kw < kW; ++kw) {
        const float *col_row = col + ((c * kH + kh) * kW + kw) * HW_out;
        for (int64_t oh = 0; oh < H_out; ++oh) {
          const int64_t ih = oh * sH - pH + kh;
          if (ih < 0 || ih >= H_in) {
            continue;
          }
          float *img_row = img_c + ih * W_in;
          for (int64_t ow = 0; ow < W_out; ++ow) {
            const int64_t iw = ow * sW - pW + kw;
            if (iw >= 0 && iw < W_in) {
              img_row[iw] += col_row[oh * W_out + ow];
            }
          }
        }
      }
    }
  }
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
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));

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

  auto output = std::make_shared<Tensor>(
      std::vector<int64_t>{N, C_out, H_out, W_out}, DataType::kFLOAT32);

  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  const float *w_ptr = static_cast<const float *>(weight->DataPtr());
  const float *b_ptr =
      bias ? static_cast<const float *>(bias->DataPtr()) : nullptr;
  float *out_ptr = static_cast<float *>(output->DataPtr());

  // Direct fast path for depthwise convolution
  if (groups == C_in && C_out == C_in) {
#pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C_in; ++c) {
        const float *x_nc = in_ptr + (n * C_in + c) * H_in * W_in;
        const float *w_c = w_ptr + c * kH * kW;
        float *y_nc = out_ptr + (n * C_out + c) * H_out * W_out;
        const float b_val = b_ptr ? b_ptr[c] : 0.0f;
        for (int64_t oh = 0; oh < H_out; ++oh) {
          for (int64_t ow = 0; ow < W_out; ++ow) {
            float acc = b_val;
            for (int64_t kh = 0; kh < kH; ++kh) {
              const int64_t ih = oh * stride_h - pad_h + kh;
              if (ih < 0 || ih >= H_in) {
                continue;
              }
              for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t iw = ow * stride_w - pad_w + kw;
                if (iw >= 0 && iw < W_in) {
                  acc += x_nc[ih * W_in + iw] * w_c[kh * kW + kw];
                }
              }
            }
            y_nc[oh * W_out + ow] = acc;
          }
        }
      }
    }
    return output;
  }

  const int64_t K = C_in_g * kH * kW;
  const int64_t HW_out = H_out * W_out;

#pragma omp parallel for
  for (int64_t n = 0; n < N; ++n) {
    std::vector<float> col_buf(static_cast<size_t>(K * HW_out));
    for (int64_t g = 0; g < groups; ++g) {
      const float *img_ng = in_ptr + (n * C_in + g * C_in_g) * H_in * W_in;
      Im2ColSingleGroup(img_ng, C_in_g, H_in, W_in, kH, kW, stride_h, stride_w,
                        pad_h, pad_w, H_out, W_out, col_buf.data());

      Eigen::Map<const RowMajorMatrix> w_mat(w_ptr + g * C_out_g * K, C_out_g,
                                             K);
      Eigen::Map<const RowMajorMatrix> col_mat(col_buf.data(), K, HW_out);
      Eigen::Map<RowMajorMatrix> out_mat(
          out_ptr + (n * C_out + g * C_out_g) * HW_out, C_out_g, HW_out);
      out_mat.noalias() = w_mat * col_mat;

      if (b_ptr != nullptr) {
        Eigen::Map<const Eigen::VectorXf> b_vec(b_ptr + g * C_out_g, C_out_g);
        out_mat.colwise() += b_vec;
      }
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
  CHECK_EQ(static_cast<int>(grad_output->Dtype()),
           static_cast<int>(DataType::kFLOAT32));

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

  auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
  const float *w_ptr = static_cast<const float *>(weight->DataPtr());
  const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
  float *gi_ptr = static_cast<float *>(grad_input->DataPtr());

  if (groups == C_in && C_out == C_in) {
    grad_input->Fill(0.0f);
#pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C_in; ++c) {
        const float *go_nc = go_ptr + (n * C_out + c) * H_out * W_out;
        const float *w_c = w_ptr + c * kH * kW;
        float *gi_nc = gi_ptr + (n * C_in + c) * H_in * W_in;
        for (int64_t oh = 0; oh < H_out; ++oh) {
          for (int64_t ow = 0; ow < W_out; ++ow) {
            const float go_val = go_nc[oh * W_out + ow];
            for (int64_t kh = 0; kh < kH; ++kh) {
              const int64_t ih = oh * stride_h - pad_h + kh;
              if (ih < 0 || ih >= H_in) {
                continue;
              }
              for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t iw = ow * stride_w - pad_w + kw;
                if (iw >= 0 && iw < W_in) {
                  gi_nc[ih * W_in + iw] += go_val * w_c[kh * kW + kw];
                }
              }
            }
          }
        }
      }
    }
    return grad_input;
  }

  const int64_t K = C_in_g * kH * kW;
  const int64_t HW_out = H_out * W_out;

#pragma omp parallel for
  for (int64_t n = 0; n < N; ++n) {
    std::vector<float> col_buf(static_cast<size_t>(K * HW_out));
    for (int64_t g = 0; g < groups; ++g) {
      Eigen::Map<const RowMajorMatrix> w_mat(w_ptr + g * C_out_g * K, C_out_g,
                                             K);
      Eigen::Map<const RowMajorMatrix> go_mat(
          go_ptr + (n * C_out + g * C_out_g) * HW_out, C_out_g, HW_out);
      Eigen::Map<RowMajorMatrix> col_mat(col_buf.data(), K, HW_out);
      col_mat.noalias() = w_mat.transpose() * go_mat;

      float *gi_ng = gi_ptr + (n * C_in + g * C_in_g) * H_in * W_in;
      Col2ImSingleGroup(col_buf.data(), C_in_g, H_in, W_in, kH, kW, stride_h,
                        stride_w, pad_h, pad_w, H_out, W_out, gi_ng);
    }
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
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));

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

  auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
  grad_weight->Fill(0.0f);

  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
  float *gw_ptr = static_cast<float *>(grad_weight->DataPtr());

  if (groups == C_in && C_out == C_in) {
#pragma omp parallel for
    for (int64_t c = 0; c < C_in; ++c) {
      float *gw_c = gw_ptr + c * kH * kW;
      for (int64_t kh = 0; kh < kH; ++kh) {
        for (int64_t kw = 0; kw < kW; ++kw) {
          double acc = 0.0;
          for (int64_t n = 0; n < N; ++n) {
            const float *x_nc = in_ptr + (n * C_in + c) * H_in * W_in;
            const float *go_nc = go_ptr + (n * C_out + c) * H_out * W_out;
            for (int64_t oh = 0; oh < H_out; ++oh) {
              const int64_t ih = oh * stride_h - pad_h + kh;
              if (ih < 0 || ih >= H_in) {
                continue;
              }
              for (int64_t ow = 0; ow < W_out; ++ow) {
                const int64_t iw = ow * stride_w - pad_w + kw;
                if (iw >= 0 && iw < W_in) {
                  acc += static_cast<double>(x_nc[ih * W_in + iw]) *
                         static_cast<double>(go_nc[oh * W_out + ow]);
                }
              }
            }
          }
          gw_c[kh * kW + kw] = static_cast<float>(acc);
        }
      }
    }
    return grad_weight;
  }

  const int64_t K = C_in_g * kH * kW;
  const int64_t HW_out = H_out * W_out;

  std::vector<float> col_buf(static_cast<size_t>(K * HW_out));
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t g = 0; g < groups; ++g) {
      const float *img_ng = in_ptr + (n * C_in + g * C_in_g) * H_in * W_in;
      Im2ColSingleGroup(img_ng, C_in_g, H_in, W_in, kH, kW, stride_h, stride_w,
                        pad_h, pad_w, H_out, W_out, col_buf.data());

      Eigen::Map<const RowMajorMatrix> go_mat(
          go_ptr + (n * C_out + g * C_out_g) * HW_out, C_out_g, HW_out);
      Eigen::Map<const RowMajorMatrix> col_mat(col_buf.data(), K, HW_out);
      Eigen::Map<RowMajorMatrix> gw_mat(gw_ptr + g * C_out_g * K, C_out_g, K);
      gw_mat.noalias() += go_mat * col_mat.transpose();
    }
  }

  return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(
    const std::shared_ptr<Tensor> &grad_output) {
  CHECK_EQ(grad_output->Dims().size(), 4);
  CHECK_EQ(static_cast<int>(grad_output->Dtype()),
           static_cast<int>(DataType::kFLOAT32));

  const int64_t N = grad_output->Dims()[0];
  const int64_t C_out = grad_output->Dims()[1];
  const int64_t HW_out = grad_output->Dims()[2] * grad_output->Dims()[3];

  auto grad_bias =
      std::make_shared<Tensor>(std::vector<int64_t>{C_out}, DataType::kFLOAT32);
  const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
  float *gb_ptr = static_cast<float *>(grad_bias->DataPtr());

#pragma omp parallel for
  for (int64_t c = 0; c < C_out; ++c) {
    double sum = 0.0;
    for (int64_t n = 0; n < N; ++n) {
      const float *go_nc = go_ptr + (n * C_out + c) * HW_out;
      for (int64_t hw = 0; hw < HW_out; ++hw) {
        sum += static_cast<double>(go_nc[hw]);
      }
    }
    gb_ptr[c] = static_cast<float>(sum);
  }
  return grad_bias;
}

}  // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CONV_KERNEL(kernel_name)                          \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, \
                  infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CONV_KERNEL(Conv2dForward)
REGISTER_CPU_CONV_KERNEL(Conv2dBackwardInput)
REGISTER_CPU_CONV_KERNEL(Conv2dBackwardWeight)
REGISTER_CPU_CONV_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CPU_CONV_KERNEL

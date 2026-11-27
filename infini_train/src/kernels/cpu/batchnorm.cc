#include <stdint.h>

#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
BatchNorm2dForward(const std::shared_ptr<Tensor> &input,
                   const std::shared_ptr<Tensor> &weight,
                   const std::shared_ptr<Tensor> &bias,
                   const std::shared_ptr<Tensor> &running_mean,
                   const std::shared_ptr<Tensor> &running_var, bool training,
                   float momentum, float eps) {
  CHECK_EQ(input->Dims().size(), 4);
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  const int64_t N = input->Dims()[0];
  const int64_t C = input->Dims()[1];
  const int64_t H = input->Dims()[2];
  const int64_t W = input->Dims()[3];
  const int64_t HW = H * W;
  const int64_t M = N * HW;
  CHECK_EQ(weight->Dims()[0], C);
  CHECK_EQ(bias->Dims()[0], C);

  auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
  auto save_mean =
      std::make_shared<Tensor>(std::vector<int64_t>{C}, DataType::kFLOAT32);
  auto save_invstd =
      std::make_shared<Tensor>(std::vector<int64_t>{C}, DataType::kFLOAT32);

  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  const float *w_ptr = static_cast<const float *>(weight->DataPtr());
  const float *b_ptr = static_cast<const float *>(bias->DataPtr());
  float *rm_ptr =
      running_mean ? static_cast<float *>(running_mean->DataPtr()) : nullptr;
  float *rv_ptr =
      running_var ? static_cast<float *>(running_var->DataPtr()) : nullptr;
  float *out_ptr = static_cast<float *>(output->DataPtr());
  float *sm_ptr = static_cast<float *>(save_mean->DataPtr());
  float *si_ptr = static_cast<float *>(save_invstd->DataPtr());

#pragma omp parallel for
  for (int64_t c = 0; c < C; ++c) {
    double mean = 0.0;
    double invstd = 0.0;
    if (training) {
      double sum = 0.0;
      for (int64_t n = 0; n < N; ++n) {
        const float *x_nc = in_ptr + (n * C + c) * HW;
        for (int64_t hw = 0; hw < HW; ++hw) {
          sum += static_cast<double>(x_nc[hw]);
        }
      }
      mean = sum / static_cast<double>(M);

      double sq_sum = 0.0;
      for (int64_t n = 0; n < N; ++n) {
        const float *x_nc = in_ptr + (n * C + c) * HW;
        for (int64_t hw = 0; hw < HW; ++hw) {
          const double diff = static_cast<double>(x_nc[hw]) - mean;
          sq_sum += diff * diff;
        }
      }
      const double var_biased = sq_sum / static_cast<double>(M);
      invstd = 1.0 / std::sqrt(var_biased + static_cast<double>(eps));

      if (rm_ptr != nullptr && rv_ptr != nullptr) {
        const double var_unbiased =
            M > 1 ? sq_sum / static_cast<double>(M - 1) : var_biased;
        rm_ptr[c] =
            static_cast<float>((1.0 - momentum) * rm_ptr[c] + momentum * mean);
        rv_ptr[c] = static_cast<float>((1.0 - momentum) * rv_ptr[c] +
                                       momentum * var_unbiased);
      }
    } else {
      CHECK(rm_ptr != nullptr && rv_ptr != nullptr);
      mean = static_cast<double>(rm_ptr[c]);
      invstd = 1.0 / std::sqrt(static_cast<double>(rv_ptr[c]) +
                               static_cast<double>(eps));
    }

    sm_ptr[c] = static_cast<float>(mean);
    si_ptr[c] = static_cast<float>(invstd);

    const float mean_f = static_cast<float>(mean);
    const float scale = w_ptr[c] * static_cast<float>(invstd);
    const float shift = b_ptr[c];
    for (int64_t n = 0; n < N; ++n) {
      const float *x_nc = in_ptr + (n * C + c) * HW;
      float *y_nc = out_ptr + (n * C + c) * HW;
      for (int64_t hw = 0; hw < HW; ++hw) {
        y_nc[hw] = (x_nc[hw] - mean_f) * scale + shift;
      }
    }
  }
  return {output, save_mean, save_invstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
           std::shared_ptr<Tensor>>
BatchNorm2dBackward(const std::shared_ptr<Tensor> &grad_output,
                    const std::shared_ptr<Tensor> &input,
                    const std::shared_ptr<Tensor> &weight,
                    const std::shared_ptr<Tensor> &save_mean,
                    const std::shared_ptr<Tensor> &save_invstd, bool training) {
  CHECK_EQ(input->Dims().size(), 4);
  CHECK_EQ(static_cast<int>(input->Dtype()),
           static_cast<int>(DataType::kFLOAT32));
  const int64_t N = input->Dims()[0];
  const int64_t C = input->Dims()[1];
  const int64_t H = input->Dims()[2];
  const int64_t W = input->Dims()[3];
  const int64_t HW = H * W;
  const int64_t M = N * HW;

  auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
  auto grad_weight =
      std::make_shared<Tensor>(std::vector<int64_t>{C}, DataType::kFLOAT32);
  auto grad_bias =
      std::make_shared<Tensor>(std::vector<int64_t>{C}, DataType::kFLOAT32);

  const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
  const float *in_ptr = static_cast<const float *>(input->DataPtr());
  const float *w_ptr = static_cast<const float *>(weight->DataPtr());
  const float *sm_ptr = static_cast<const float *>(save_mean->DataPtr());
  const float *si_ptr = static_cast<const float *>(save_invstd->DataPtr());
  float *gi_ptr = static_cast<float *>(grad_input->DataPtr());
  float *gw_ptr = static_cast<float *>(grad_weight->DataPtr());
  float *gb_ptr = static_cast<float *>(grad_bias->DataPtr());

#pragma omp parallel for
  for (int64_t c = 0; c < C; ++c) {
    const double mean = static_cast<double>(sm_ptr[c]);
    const double invstd = static_cast<double>(si_ptr[c]);
    double sum_dy = 0.0;
    double sum_dy_xhat = 0.0;

    for (int64_t n = 0; n < N; ++n) {
      const float *go_nc = go_ptr + (n * C + c) * HW;
      const float *x_nc = in_ptr + (n * C + c) * HW;
      for (int64_t hw = 0; hw < HW; ++hw) {
        const double dy = static_cast<double>(go_nc[hw]);
        const double xhat = (static_cast<double>(x_nc[hw]) - mean) * invstd;
        sum_dy += dy;
        sum_dy_xhat += dy * xhat;
      }
    }

    gb_ptr[c] = static_cast<float>(sum_dy);
    gw_ptr[c] = static_cast<float>(sum_dy_xhat);

    const double w = static_cast<double>(w_ptr[c]);
    if (training) {
      const double coeff = w * invstd / static_cast<double>(M);
      for (int64_t n = 0; n < N; ++n) {
        const float *go_nc = go_ptr + (n * C + c) * HW;
        const float *x_nc = in_ptr + (n * C + c) * HW;
        float *gi_nc = gi_ptr + (n * C + c) * HW;
        for (int64_t hw = 0; hw < HW; ++hw) {
          const double dy = static_cast<double>(go_nc[hw]);
          const double xhat = (static_cast<double>(x_nc[hw]) - mean) * invstd;
          gi_nc[hw] = static_cast<float>(coeff * (static_cast<double>(M) * dy -
                                                  sum_dy - xhat * sum_dy_xhat));
        }
      }
    } else {
      const float scale = static_cast<float>(w * invstd);
      for (int64_t n = 0; n < N; ++n) {
        const float *go_nc = go_ptr + (n * C + c) * HW;
        float *gi_nc = gi_ptr + (n * C + c) * HW;
        for (int64_t hw = 0; hw < HW; ++hw) {
          gi_nc[hw] = go_nc[hw] * scale;
        }
      }
    }
  }
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace infini_train::kernels::cpu

#define REGISTER_CPU_BATCHNORM_KERNEL(kernel_name)                     \
  REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, \
                  infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_BATCHNORM_KERNEL(BatchNorm2dForward)
REGISTER_CPU_BATCHNORM_KERNEL(BatchNorm2dBackward)

#undef REGISTER_CPU_BATCHNORM_KERNEL

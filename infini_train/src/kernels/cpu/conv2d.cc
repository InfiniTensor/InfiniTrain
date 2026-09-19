#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input,
                                      const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias,
                                      int64_t stride, int64_t padding) {
    const auto &input_dims = input->Dims();
    const auto &weight_dims = weight->Dims();
    int64_t N = input_dims[0], C_in = input_dims[1], H = input_dims[2], W = input_dims[3];
    int64_t C_out = weight_dims[0], K = weight_dims[2];
    int64_t H_out = (H + 2 * padding - K) / stride + 1;
    int64_t W_out = (W + 2 * padding - K) / stride + 1;

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{N, C_out, H_out, W_out}, DataType::kFLOAT32);

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *weight_ptr = static_cast<const float *>(weight->DataPtr());
    const float *bias_ptr = bias ? static_cast<const float *>(bias->DataPtr()) : nullptr;
    float *output_ptr = static_cast<float *>(output->DataPtr());

    for (int64_t n = 0; n < N; ++n)
        for (int64_t co = 0; co < C_out; ++co)
            for (int64_t oh = 0; oh < H_out; ++oh)
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    float sum = bias_ptr ? bias_ptr[co] : 0.0f;
                    for (int64_t ci = 0; ci < C_in; ++ci)
                        for (int64_t kh = 0; kh < K; ++kh)
                            for (int64_t kw = 0; kw < K; ++kw) {
                                int64_t ih = oh * stride + kh - padding;
                                int64_t iw = ow * stride + kw - padding;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                sum += input_ptr[((n * C_in + ci) * H + ih) * W + iw]
                                     * weight_ptr[((co * C_in + ci) * K + kh) * K + kw];
                            }
                    output_ptr[((n * C_out + co) * H_out + oh) * W_out + ow] = sum;
                }
    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims,
                                            int64_t stride, int64_t padding) {
    int64_t N = input_dims[0], C_in = input_dims[1], H = input_dims[2], W = input_dims[3];
    const auto &weight_dims = weight->Dims();
    int64_t C_out = weight_dims[0], K = weight_dims[2];
    int64_t H_out = (H + 2 * padding - K) / stride + 1;
    int64_t W_out = (W + 2 * padding - K) / stride + 1;

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    const float *weight_ptr = static_cast<const float *>(weight->DataPtr());

    for (int64_t i = 0; i < N * C_in * H * W; ++i) grad_input_ptr[i] = 0.0f;

    for (int64_t n = 0; n < N; ++n)
        for (int64_t co = 0; co < C_out; ++co)
            for (int64_t oh = 0; oh < H_out; ++oh)
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    float g = grad_output_ptr[((n * C_out + co) * H_out + oh) * W_out + ow];
                    for (int64_t ci = 0; ci < C_in; ++ci)
                        for (int64_t kh = 0; kh < K; ++kh)
                            for (int64_t kw = 0; kw < K; ++kw) {
                                int64_t ih = oh * stride + kh - padding;
                                int64_t iw = ow * stride + kw - padding;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                grad_input_ptr[((n * C_in + ci) * H + ih) * W + iw]
                                    += g * weight_ptr[((co * C_in + ci) * K + kh) * K + kw];
                            }
                }
    return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output,
                                             const std::vector<int64_t> &weight_dims,
                                             int64_t stride, int64_t padding) {
    int64_t N = input->Dims()[0], C_in = input->Dims()[1], H = input->Dims()[2], W = input->Dims()[3];
    int64_t C_out = weight_dims[0], K = weight_dims[2];
    int64_t H_out = (H + 2 * padding - K) / stride + 1;
    int64_t W_out = (W + 2 * padding - K) / stride + 1;

    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    float *grad_weight_ptr = static_cast<float *>(grad_weight->DataPtr());
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());

    for (int64_t i = 0; i < C_out * C_in * K * K; ++i) grad_weight_ptr[i] = 0.0f;

    for (int64_t n = 0; n < N; ++n)
        for (int64_t co = 0; co < C_out; ++co)
            for (int64_t oh = 0; oh < H_out; ++oh)
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    float g = grad_output_ptr[((n * C_out + co) * H_out + oh) * W_out + ow];
                    for (int64_t ci = 0; ci < C_in; ++ci)
                        for (int64_t kh = 0; kh < K; ++kh)
                            for (int64_t kw = 0; kw < K; ++kw) {
                                int64_t ih = oh * stride + kh - padding;
                                int64_t iw = ow * stride + kw - padding;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                grad_weight_ptr[((co * C_in + ci) * K + kh) * K + kw]
                                    += g * input_ptr[((n * C_in + ci) * H + ih) * W + iw];
                            }
                }
    return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_channels) {
    const auto &grad_dims = grad_output->Dims();
    int64_t N = grad_dims[0], H_out = grad_dims[2], W_out = grad_dims[3];
    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32);
    float *grad_bias_ptr = static_cast<float *>(grad_bias->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());

    for (int64_t co = 0; co < out_channels; ++co) {
        float sum = 0.0f;
        for (int64_t n = 0; n < N; ++n)
            for (int64_t oh = 0; oh < H_out; ++oh)
                for (int64_t ow = 0; ow < W_out; ++ow)
                    sum += grad_output_ptr[((n * out_channels + co) * H_out + oh) * W_out + ow];
        grad_bias_ptr[co] = sum;
    }
    return grad_bias;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CONV2D_KERNEL(kernel_name) \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CONV2D_KERNEL(Conv2dForward)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardInput)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardWeight)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CPU_CONV2D_KERNEL
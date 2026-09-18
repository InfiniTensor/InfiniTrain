#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/kernels/common/conv.h"

namespace infini_train::kernels::cpu {
namespace {

inline int64_t NCHWOffset(int64_t n, int64_t c, int64_t h, int64_t w, int64_t channels, int64_t height, int64_t width) {
    return ((n * channels + c) * height + h) * width + w;
}

inline int64_t OIHWOffset(int64_t o, int64_t i, int64_t kh, int64_t kw, int64_t in_channels, int64_t kernel_h,
                          int64_t kernel_w) {
    return ((o * in_channels + i) * kernel_h + kh) * kernel_w + kw;
}

void CheckFP32(const std::shared_ptr<Tensor> &tensor, const char *name) {
    CHECK(tensor->Dtype() == DataType::kFLOAT32) << name << " currently supports float32 only";
}

} // namespace

// Direct cross-correlation (no kernel flip), scalar stride/padding.
std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias, int64_t stride, int64_t padding) {
    CheckFP32(input, "CPU Conv2dForward");
    CheckFP32(weight, "CPU Conv2dForward");
    if (bias) {
        CheckFP32(bias, "CPU Conv2dForward");
    }

    const Conv2dMeta meta = MakeConv2dMeta(input, weight, stride, padding);
    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], meta.out_channels);
    }

    auto output = std::make_shared<Tensor>(Conv2dOutputDims(meta), DataType::kFLOAT32);
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *weight_ptr = static_cast<const float *>(weight->DataPtr());
    const float *bias_ptr = bias ? static_cast<const float *>(bias->DataPtr()) : nullptr;
    float *output_ptr = static_cast<float *>(output->DataPtr());

    for (int64_t n = 0; n < meta.batch; ++n) {
        for (int64_t oc = 0; oc < meta.out_channels; ++oc) {
            for (int64_t oh = 0; oh < meta.output_h; ++oh) {
                for (int64_t ow = 0; ow < meta.output_w; ++ow) {
                    float acc = bias_ptr ? bias_ptr[oc] : 0.0f;
                    for (int64_t ic = 0; ic < meta.in_channels; ++ic) {
                        for (int64_t kh = 0; kh < meta.kernel_h; ++kh) {
                            const int64_t ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= meta.input_h) {
                                continue;
                            }
                            for (int64_t kw = 0; kw < meta.kernel_w; ++kw) {
                                const int64_t iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= meta.input_w) {
                                    continue;
                                }
                                acc += input_ptr[NCHWOffset(n, ic, ih, iw, meta.in_channels, meta.input_h,
                                                            meta.input_w)]
                                     * weight_ptr[OIHWOffset(oc, ic, kh, kw, meta.in_channels, meta.kernel_h,
                                                             meta.kernel_w)];
                            }
                        }
                    }
                    output_ptr[NCHWOffset(n, oc, oh, ow, meta.out_channels, meta.output_h, meta.output_w)] = acc;
                }
            }
        }
    }
    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output, int64_t stride, int64_t padding,
                                            const std::vector<int64_t> &input_dims) {
    CheckFP32(weight, "CPU Conv2dBackwardInput");
    CheckFP32(grad_output, "CPU Conv2dBackwardInput");
    CHECK_EQ(input_dims.size(), 4);

    const auto &go_dims = grad_output->Dims();
    CHECK_EQ(go_dims.size(), 4);
    const int64_t batch = input_dims[0];
    const int64_t in_channels = input_dims[1];
    const int64_t input_h = input_dims[2];
    const int64_t input_w = input_dims[3];
    const int64_t out_channels = go_dims[1];
    const int64_t output_h = go_dims[2];
    const int64_t output_w = go_dims[3];

    const auto &w_dims = weight->Dims();
    CHECK_EQ(w_dims.size(), 4);
    CHECK_EQ(w_dims[0], out_channels);
    CHECK_EQ(w_dims[1], in_channels);
    const int64_t kernel_h = w_dims[2];
    const int64_t kernel_w = w_dims[3];
    CHECK_EQ(output_h, (input_h + 2 * padding - kernel_h) / stride + 1);
    CHECK_EQ(output_w, (input_w + 2 * padding - kernel_w) / stride + 1);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    grad_input->Fill(0.0f);
    const float *weight_ptr = static_cast<const float *>(weight->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    const float go = grad_output_ptr[NCHWOffset(n, oc, oh, ow, out_channels, output_h, output_w)];
                    for (int64_t ic = 0; ic < in_channels; ++ic) {
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            const int64_t ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= input_h) {
                                continue;
                            }
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                const int64_t iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= input_w) {
                                    continue;
                                }
                                grad_input_ptr[NCHWOffset(n, ic, ih, iw, in_channels, input_h, input_w)]
                                    += go * weight_ptr[OIHWOffset(oc, ic, kh, kw, in_channels, kernel_h, kernel_w)];
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output, int64_t stride,
                                             int64_t padding, const std::vector<int64_t> &weight_dims) {
    CheckFP32(input, "CPU Conv2dBackwardWeight");
    CheckFP32(grad_output, "CPU Conv2dBackwardWeight");
    CHECK_EQ(weight_dims.size(), 4);

    const auto &in_dims = input->Dims();
    CHECK_EQ(in_dims.size(), 4);
    const int64_t batch = in_dims[0];
    const int64_t in_channels = in_dims[1];
    const int64_t input_h = in_dims[2];
    const int64_t input_w = in_dims[3];

    const auto &go_dims = grad_output->Dims();
    CHECK_EQ(go_dims.size(), 4);
    CHECK_EQ(go_dims[0], batch);
    const int64_t out_channels = go_dims[1];
    const int64_t output_h = go_dims[2];
    const int64_t output_w = go_dims[3];

    CHECK_EQ(weight_dims[0], out_channels);
    CHECK_EQ(weight_dims[1], in_channels);
    const int64_t kernel_h = weight_dims[2];
    const int64_t kernel_w = weight_dims[3];
    CHECK_EQ(output_h, (input_h + 2 * padding - kernel_h) / stride + 1);
    CHECK_EQ(output_w, (input_w + 2 * padding - kernel_w) / stride + 1);

    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    grad_weight->Fill(0.0f);
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_weight_ptr = static_cast<float *>(grad_weight->DataPtr());

    for (int64_t oc = 0; oc < out_channels; ++oc) {
        for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    float acc = 0.0f;
                    for (int64_t n = 0; n < batch; ++n) {
                        for (int64_t oh = 0; oh < output_h; ++oh) {
                            const int64_t ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= input_h) {
                                continue;
                            }
                            for (int64_t ow = 0; ow < output_w; ++ow) {
                                const int64_t iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= input_w) {
                                    continue;
                                }
                                acc += input_ptr[NCHWOffset(n, ic, ih, iw, in_channels, input_h, input_w)]
                                     * grad_output_ptr[NCHWOffset(n, oc, oh, ow, out_channels, output_h, output_w)];
                            }
                        }
                    }
                    grad_weight_ptr[OIHWOffset(oc, ic, kh, kw, in_channels, kernel_h, kernel_w)] = acc;
                }
            }
        }
    }
    return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output) {
    CheckFP32(grad_output, "CPU Conv2dBackwardBias");
    const auto &go_dims = grad_output->Dims();
    CHECK_EQ(go_dims.size(), 4);
    const int64_t batch = go_dims[0];
    const int64_t out_channels = go_dims[1];
    const int64_t output_h = go_dims[2];
    const int64_t output_w = go_dims[3];

    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32);
    grad_bias->Fill(0.0f);
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_bias_ptr = static_cast<float *>(grad_bias->DataPtr());

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    grad_bias_ptr[oc] += grad_output_ptr[NCHWOffset(n, oc, oh, ow, out_channels, output_h, output_w)];
                }
            }
        }
    }
    return grad_bias;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CONV_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CONV_KERNEL(Conv2dForward)
REGISTER_CPU_CONV_KERNEL(Conv2dBackwardInput)
REGISTER_CPU_CONV_KERNEL(Conv2dBackwardWeight)
REGISTER_CPU_CONV_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CPU_CONV_KERNEL

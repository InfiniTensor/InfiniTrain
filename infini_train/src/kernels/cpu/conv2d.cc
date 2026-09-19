// 每个函数块都只读取输入的tensor数据，算结果后，返回新的tensor
// 不在这里写autograd逻辑
// 用autograd层调用即可

#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
    // 公式与PyTorch一致: OH = (H + 2*padding - KH) / stride + 1
    inline int64_t ConvOutputSize(int64_t size, int64_t kernel, int64_t stride, int64_t padding) {
        return (size + 2 * padding - kernel) / stride + 1;
    }
}

// namespace

// Conv2dForward 前向
// Conv2dBackwardInput 反向
// Conv2dBackwardWeight 反向
// Conv2dBackwardBias 反向

// 前向
std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input, 
                                    const std::shared_ptr<Tensor> &weight, 
                                    const std::shared_ptr<Tensor> &bias,
                                    int64_t stride, int64_t padding) {                         
    const auto &input_dims = input->Dims();
    const auto &weight_dims = weight->Dims();
    CHECK_EQ(input_dims.size(), 4) << "Conv2d input must be 4-D (NCHW)";
    CHECK_EQ(weight_dims.size(), 4) << "Conv2d weight must be 4-D";
    CHECK_EQ(input_dims[1], weight_dims[1]) << "input channels must match weight channels";

    const int64_t n = input_dims[0], c = input_dims[1], h = input_dims[2], w = input_dims[3];
    const int64_t o = weight_dims[0], kh = weight_dims[2], kw = weight_dims[3];
    // oh ow： 输出高度， 输出宽度
    const int64_t oh = ConvOutputSize(h, kh, stride, padding);
    const int64_t ow = ConvOutputSize(w, kw, stride, padding);
    CHECK_GT(oh, 0) << "Conv2d output height must be positive, check kernel/stride/padding";
    CHECK_GT(ow, 0) << "Conv2d output width must be positive, check kernel/stride/padding";


    auto output = std::make_shared<Tensor>(std::vector<int64_t>{n, o, oh, ow}, DataType::kFLOAT32);
    output->Fill(0.0f);

    // NCHW布局
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *weight_ptr = static_cast<const float *>(weight->DataPtr());
    const float *bias_ptr = bias ? static_cast<const float *>(bias->DataPtr()) : nullptr;
    float *output_ptr = static_cast<float *>(output->DataPtr());

    // B->输出通道->输出高度->输出宽度
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t oi = 0; oi < o; ++oi) {
            for (int64_t oy = 0; oy < oh; ++oy) {
                for (int64_t ox = 0; ox < ow; ++ox) {
                    // 累加器
                    float acc = bias_ptr ? bias_ptr[oi] : 0.0f;

                    // 输入通道->卷积核高->卷积核宽
                    for (int64_t ci = 0; ci < c; ++ci) {
                        for (int64_t ky = 0; ky < kh; ++ky) {
                            // 输出坐标 (oy, ox)和卷积核位置(ky, kx) 对应输入坐标(iy, ix)
                            // iy = oy * stride + ky - padding
                            const int64_t iy = oy * stride + ky - padding;
                            if (iy < 0 || iy >= h) {
                                continue;
                            }

                            for (int64_t kx = 0; kx < kw; ++kx) {
                                const int64_t ix = ox * stride + kx - padding;
                                if (ix < 0 || ix >= w) {
                                    continue;
                                }
                                // input[ni, ci, iy, ix] * weight[oi, ci, ky, kx]
                                acc += input_ptr[( (ni*c + ci) * h + iy) * w + ix] * weight_ptr[( (oi*c + ci) * kh + ky) * kw + kx];
                            }
                        }
                    }

                    output_ptr[( (ni*o + oi) * oh + oy) * ow + ox] = acc;

                }
            }
        }
    }

    return output;                              
                                
}

// 反向1： 输入的梯度
std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims,
                                            int64_t stride, int64_t padding) {
    const auto &grad_dims = grad_output->Dims();
    const int64_t n = input_dims[0], c = input_dims[1], h = input_dims[2], w = input_dims[3];
    const int64_t o = weight->Dims()[0], kh = weight->Dims()[2], kw = weight->Dims()[3];
    const int64_t oh = grad_dims[2], ow = grad_dims[3];

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    grad_input->Fill(0.0f);

    const float *weight_ptr = static_cast<const float *>(weight->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());


    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t ci = 0; ci < c; ++ci) {
            for (int64_t iy = 0; iy < h; ++iy) {
                for (int64_t ix = 0; ix < w; ++ix) {
                    float acc = 0.0f;
                    for (int64_t oi = 0; oi < o; ++oi) {
                        for (int64_t ky = 0; ky < kh; ++ky) {
                            // 反解出：oy * stride = iy + padding - ky
                            const int64_t rem_y = iy + padding - ky;
                            // 如果除不尽说明(iy, ix)对卷积核这行没有贡献

                            if (rem_y < 0 || rem_y % stride != 0) {
                                continue;
                            }
                            const int64_t oy = rem_y / stride;
                            if (oy >= oh) {
                                continue;
                            }

                            for (int64_t kx = 0; kx < kw; ++kx) {
                                const int64_t rem_x = ix + padding - kx;
                                if (rem_x < 0 || rem_x % stride != 0){
                                    continue;
                                }
                                const int64_t ox = rem_x / stride;
                                if (ox >= ow) {
                                    continue;
                                }
                                acc += grad_output_ptr[( (ni*o + oi) * oh + oy) * ow + ox] * weight_ptr[( (oi*c + ci) * kh + ky) * kw + kx];
                            }

                        }
                    }

                    grad_input_ptr[( (ni*c + ci) * h + iy) * w + ix] = acc;
                }
            }
        }
    }
    return grad_input;
}


// 反向2： 权重的梯度
std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &weight_dims,
                                            int64_t stride, int64_t padding) {
    const auto &input_dims = input->Dims();
    const auto &grad_dims = grad_output->Dims();
    const int64_t n = input_dims[0], c = input_dims[1], h = input_dims[2], w = input_dims[3];
    const int64_t o = weight_dims[0], kh = weight_dims[2], kw = weight_dims[3];
    const int64_t oh = grad_dims[2], ow = grad_dims[3];


    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    grad_weight->Fill(0.0f);

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_weight_ptr = static_cast<float *>(grad_weight->DataPtr());

    for (int64_t oi = 0; oi < o; ++oi) {
        for (int64_t ci = 0; ci < c; ++ci) {
            for (int64_t ky = 0; ky < kh; ++ky) {
                for (int64_t kx = 0; kx < kw; ++kx) {
                    float acc = 0.0f;

                    for (int64_t ni = 0; ni < n; ++ni) {
                        for (int64_t oy = 0; oy < oh; ++oy) {
                            const int64_t iy = oy * stride + ky - padding;
                            if (iy < 0 || iy >= h) {
                                continue;
                            }
                            for (int64_t ox = 0; ox < ow; ++ox) {
                                const int64_t ix = ox * stride + kx - padding;
                                if (ix < 0 || ix >= w) {
                                    continue;
                                }
                                acc += input_ptr[( (ni*c + ci) * h + iy) * w + ix] * grad_output_ptr[( (ni*o + oi) * oh + oy) * ow + ox];

                            }
                        }
                    }

                    grad_weight_ptr[( (oi*c + ci) * kh + ky) * kw + kx] = acc;
                }
            }
        }
    }
    return grad_weight;
}


// 反向3： 偏置的梯度
std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output,
                                        int64_t out_channels) {
    const auto &grad_dims = grad_output->Dims();
    const int64_t n = grad_dims[0], oh = grad_dims[2], ow = grad_dims[3];

    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32);
    grad_bias->Fill(0.0f);


    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_bias_ptr = static_cast<float *>(grad_bias->DataPtr());

    for (int64_t oi = 0; oi < out_channels; ++oi) {
        float acc = 0.0f;
        for (int64_t ni = 0; ni < n; ++ni) {
            for (int64_t oy = 0; oy < oh; ++oy) {
                for (int64_t ox = 0; ox < ow; ++ox) {
                    acc += grad_output_ptr[( (ni * out_channels + oi) * oh + oy) * ow + ox];

                }
            }
        }
        grad_bias_ptr[oi] = acc;
    }
    return grad_bias;
}



}
// 注册表dispatcher
// CPU版本跟CUDA版本名字相同, device不同，Dispatcher按照device分发
#define REGISTER_CPU_CONV2D_KERNEL(kernel_name) \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CONV2D_KERNEL(Conv2dForward)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardInput)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardWeight)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CPU_CONV2D_KERNEL
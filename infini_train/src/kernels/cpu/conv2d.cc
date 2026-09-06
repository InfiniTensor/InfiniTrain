#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixMap = Eigen::Map<RowMajorMatrix>;
using ConstRowMajorMatrixMap = Eigen::Map<const RowMajorMatrix>;

// Conv kernels only support fp32, which covers the MNIST CNN training use case.
void CheckFloat32(const std::shared_ptr<Tensor> &tensor, const char *name) {
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "Conv2d kernel expects fp32 " << name;
    CHECK_EQ(static_cast<int>(tensor->GetDevice().type()), static_cast<int>(Device::DeviceType::kCPU))
        << "Conv2d cpu kernel expects a cpu tensor: " << name;
}

int64_t ConvOutSize(int64_t in_size, int64_t kernel_size, int64_t stride, int64_t padding) {
    CHECK_GT(stride, 0);
    CHECK_GE(padding, 0);
    const int64_t out_size = (in_size + 2 * padding - kernel_size) / stride + 1;
    CHECK_GT(out_size, 0) << "Non-positive convolution output size";
    return out_size;
}

struct Conv2dDims {
    int64_t N;
    int64_t C_in;
    int64_t H;
    int64_t W;
    int64_t C_out;
    int64_t kH;
    int64_t kW;
    int64_t H_out;
    int64_t W_out;
};

Conv2dDims ResolveConv2dDims(const std::vector<int64_t> &input_dims, const std::vector<int64_t> &weight_dims,
                             int64_t stride, int64_t padding) {
    CHECK_EQ(input_dims.size(), 4);
    CHECK_EQ(weight_dims.size(), 4);
    Conv2dDims d{};
    d.N = input_dims[0];
    d.C_in = input_dims[1];
    d.H = input_dims[2];
    d.W = input_dims[3];
    d.C_out = weight_dims[0];
    CHECK_EQ(d.C_in, weight_dims[1]) << "Conv2d input channel mismatch with weight";
    d.kH = weight_dims[2];
    d.kW = weight_dims[3];
    d.H_out = ConvOutSize(d.H, d.kH, stride, padding);
    d.W_out = ConvOutSize(d.W, d.kW, stride, padding);
    return d;
}

// Unfolds input (C_in, H, W) into col (C_in*kH*kW, H_out*W_out); padded regions contribute zeros.
// col row (c*kH+kh)*kW+kw holds the input values under the kernel tap (kh, kw) for every output position.
void Im2Col(const float *input, int64_t C_in, int64_t H, int64_t W, int64_t kH, int64_t kW, int64_t stride,
            int64_t padding, int64_t H_out, int64_t W_out, float *col) {
    const int64_t P = H_out * W_out;
    for (int64_t c = 0; c < C_in; ++c) {
        for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
                float *col_row = col + ((c * kH + kh) * kW + kw) * P;
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    const int64_t ih = oh * stride + kh - padding;
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        const int64_t iw = ow * stride + kw - padding;
                        const bool in_bounds = ih >= 0 && ih < H && iw >= 0 && iw < W;
                        col_row[oh * W_out + ow] = in_bounds ? input[(c * H + ih) * W + iw] : 0.0f;
                    }
                }
            }
        }
    }
}

// Scatter-adds grad_col (C_in*kH*kW, H_out*W_out) into grad_input (C_in, H, W) at the source positions.
void Col2ImAccumulate(const float *grad_col, int64_t C_in, int64_t H, int64_t W, int64_t kH, int64_t kW, int64_t stride,
                      int64_t padding, int64_t H_out, int64_t W_out, float *grad_input) {
    const int64_t P = H_out * W_out;
    for (int64_t c = 0; c < C_in; ++c) {
        for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
                const float *grad_col_row = grad_col + ((c * kH + kh) * kW + kw) * P;
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    const int64_t ih = oh * stride + kh - padding;
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        const int64_t iw = ow * stride + kw - padding;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            grad_input[(c * H + ih) * W + iw] += grad_col_row[oh * W_out + ow];
                        }
                    }
                }
            }
        }
    }
}
} // namespace

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias, int64_t stride, int64_t padding) {
    /*
    input: (N, C_in, H, W), weight: (C_out, C_in, kH, kW), bias: (C_out)
    output: (N, C_out, H_out, W_out), where H_out = (H + 2*padding - kH) / stride + 1
    */
    CheckFloat32(input, "input");
    CheckFloat32(weight, "weight");
    if (bias) {
        CheckFloat32(bias, "bias");
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], weight->Dims()[0]);
    }

    const Conv2dDims d = ResolveConv2dDims(input->Dims(), weight->Dims(), stride, padding);
    const int64_t K = d.C_in * d.kH * d.kW;
    const int64_t P = d.H_out * d.W_out;

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{d.N, d.C_out, d.H_out, d.W_out}, DataType::kFLOAT32);
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *weight_ptr = static_cast<const float *>(weight->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    const ConstRowMajorMatrixMap weight_mat(weight_ptr, d.C_out, K);
    RowMajorMatrix col(K, P);
    for (int64_t n = 0; n < d.N; ++n) {
        Im2Col(input_ptr + n * d.C_in * d.H * d.W, d.C_in, d.H, d.W, d.kH, d.kW, stride, padding, d.H_out, d.W_out,
               col.data());
        RowMajorMatrixMap out_mat(output_ptr + n * d.C_out * P, d.C_out, P);
        out_mat = weight_mat * col;
        if (bias) {
            const Eigen::Map<const Eigen::VectorXf> bias_vec(static_cast<const float *>(bias->DataPtr()), d.C_out);
            out_mat.colwise() += bias_vec;
        }
    }
    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output, int64_t stride, int64_t padding,
                                            const std::vector<int64_t> &input_dims) {
    /*
    grad_input = col2im(weight^T * grad_output)
    weight: (C_out, C_in, kH, kW), grad_output: (N, C_out, H_out, W_out), grad_input: (N, C_in, H, W)
    */
    CheckFloat32(weight, "weight");
    CheckFloat32(grad_output, "grad_output");

    const Conv2dDims d = ResolveConv2dDims(input_dims, weight->Dims(), stride, padding);
    const int64_t P = d.H_out * d.W_out;
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), 4);
    CHECK_EQ(grad_output_dims[0], d.N);
    CHECK_EQ(grad_output_dims[1], d.C_out);
    CHECK_EQ(grad_output_dims[2], d.H_out);
    CHECK_EQ(grad_output_dims[3], d.W_out);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    std::fill_n(grad_input_ptr, grad_input->NumElements(), 0.0f);

    const int64_t K = d.C_in * d.kH * d.kW;
    const ConstRowMajorMatrixMap weight_mat(static_cast<const float *>(weight->DataPtr()), d.C_out, K);
    RowMajorMatrix grad_col(K, P);
    for (int64_t n = 0; n < d.N; ++n) {
        const ConstRowMajorMatrixMap grad_out_mat(static_cast<const float *>(grad_output->DataPtr()) + n * d.C_out * P,
                                                  d.C_out, P);
        grad_col = weight_mat.transpose() * grad_out_mat;
        Col2ImAccumulate(grad_col.data(), d.C_in, d.H, d.W, d.kH, d.kW, stride, padding, d.H_out, d.W_out,
                         grad_input_ptr + n * d.C_in * d.H * d.W);
    }
    return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output, int64_t stride,
                                             int64_t padding, const std::vector<int64_t> &weight_dims) {
    /*
    grad_weight = sum_n grad_output[n] * col(input[n])^T
    input: (N, C_in, H, W), grad_output: (N, C_out, H_out, W_out), grad_weight: (C_out, C_in, kH, kW)
    */
    CheckFloat32(input, "input");
    CheckFloat32(grad_output, "grad_output");
    CHECK_EQ(weight_dims.size(), 4);

    const Conv2dDims d = ResolveConv2dDims(input->Dims(), weight_dims, stride, padding);
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), 4);
    CHECK_EQ(grad_output_dims[0], d.N);
    CHECK_EQ(grad_output_dims[1], d.C_out);
    CHECK_EQ(grad_output_dims[2], d.H_out);
    CHECK_EQ(grad_output_dims[3], d.W_out);

    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    float *grad_weight_ptr = static_cast<float *>(grad_weight->DataPtr());
    std::fill_n(grad_weight_ptr, grad_weight->NumElements(), 0.0f);

    const int64_t K = d.C_in * d.kH * d.kW;
    const int64_t P = d.H_out * d.W_out;
    RowMajorMatrixMap grad_weight_mat(grad_weight_ptr, d.C_out, K);
    RowMajorMatrix col(K, P);
    for (int64_t n = 0; n < d.N; ++n) {
        Im2Col(static_cast<const float *>(input->DataPtr()) + n * d.C_in * d.H * d.W, d.C_in, d.H, d.W, d.kH, d.kW,
               stride, padding, d.H_out, d.W_out, col.data());
        const ConstRowMajorMatrixMap grad_out_mat(static_cast<const float *>(grad_output->DataPtr()) + n * d.C_out * P,
                                                  d.C_out, P);
        grad_weight_mat.noalias() += grad_out_mat * col.transpose();
    }
    return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output) {
    /*
    grad_bias = sum over batch and spatial positions of grad_output
    grad_output: (N, C_out, H_out, W_out), grad_bias: (C_out)
    */
    CheckFloat32(grad_output, "grad_output");
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), 4);
    const int64_t N = grad_output_dims[0];
    const int64_t C_out = grad_output_dims[1];
    const int64_t P = grad_output_dims[2] * grad_output_dims[3];

    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{C_out}, DataType::kFLOAT32);
    float *grad_bias_ptr = static_cast<float *>(grad_bias->DataPtr());
    std::fill_n(grad_bias_ptr, C_out, 0.0f);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> grad_bias_vec(grad_bias_ptr, C_out);

    for (int64_t n = 0; n < N; ++n) {
        const ConstRowMajorMatrixMap grad_out_mat(static_cast<const float *>(grad_output->DataPtr()) + n * C_out * P,
                                                  C_out, P);
        grad_bias_vec += grad_out_mat.rowwise().sum();
    }
    return grad_bias;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CONV2D_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CONV2D_KERNEL(Conv2dForward)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardInput)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardWeight)
REGISTER_CPU_CONV2D_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CPU_CONV2D_KERNEL

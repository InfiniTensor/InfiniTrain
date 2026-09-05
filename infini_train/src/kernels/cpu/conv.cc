#include <cstdint>
#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// im2col for stride=1 / padding=0 cross-correlation: extracts every (C, kH, kW) sliding window of a
// (C, H, W) image into a (out_height*out_width, C*kH*kW) row-major matrix. The k-th column of a row
// corresponds to the (c, u, v) tap ordered like the row-major flattening of an (O, C, kH, kW)
// weight, so a conv step is a plain GEMM against the weight viewed as (O, C*kH*kW).
void Im2col(const float *image, int64_t channels, int64_t height, int64_t width, int64_t kernel_h, int64_t kernel_w,
            int64_t out_height, int64_t out_width, float *col) {
    const int64_t flat_kernel = channels * kernel_h * kernel_w;
    for (int64_t i = 0; i < out_height; ++i) {
        for (int64_t j = 0; j < out_width; ++j) {
            float *row = col + (i * out_width + j) * flat_kernel;
            for (int64_t c = 0; c < channels; ++c) {
                const float *window = image + (c * height + i) * width + j;
                for (int64_t u = 0; u < kernel_h; ++u) {
                    for (int64_t v = 0; v < kernel_w; ++v) {
                        row[(c * kernel_h + u) * kernel_w + v] = window[u * width + v];
                    }
                }
            }
        }
    }
}

// col2im: inverse of Im2col for gradients. An input element is covered by multiple output windows,
// so contributions from all patches touching it are accumulated.
void Col2im(const float *col, int64_t channels, int64_t height, int64_t width, int64_t kernel_h, int64_t kernel_w,
            int64_t out_height, int64_t out_width, float *grad_image) {
    const int64_t flat_kernel = channels * kernel_h * kernel_w;
    for (int64_t i = 0; i < out_height; ++i) {
        for (int64_t j = 0; j < out_width; ++j) {
            const float *row = col + (i * out_width + j) * flat_kernel;
            for (int64_t c = 0; c < channels; ++c) {
                float *window = grad_image + (c * height + i) * width + j;
                for (int64_t u = 0; u < kernel_h; ++u) {
                    for (int64_t v = 0; v < kernel_w; ++v) {
                        window[u * width + v] += row[(c * kernel_h + u) * kernel_w + v];
                    }
                }
            }
        }
    }
}

} // namespace

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias) {
    /*
    Cross-correlation (PyTorch conv2d semantics, kernel not flipped), stride=1, padding=0:
    output(n, o, i, j) = bias(o) + sum_{c,u,v} input(n, c, i+u, j+v) * weight(o, c, u, v)
    Computed per image as output(n) = weight(o, C*kH*kW) * im2col(input(n))^T + bias
    */

    const auto &input_dims = input->Dims();
    CHECK_EQ(input_dims.size(), 4);
    const int64_t batch = input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t height = input_dims[2];
    const int64_t width = input_dims[3];

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 4);
    const int64_t out_channels = weight_dims[0];
    const int64_t kernel_h = weight_dims[2];
    const int64_t kernel_w = weight_dims[3];
    CHECK_EQ(weight_dims[1], channels);
    CHECK_GE(height, kernel_h);
    CHECK_GE(width, kernel_w);
    CHECK(input->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK(weight->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_channels);
        CHECK(bias->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    }

    const int64_t out_height = height - kernel_h + 1;
    const int64_t out_width = width - kernel_w + 1;
    const int64_t patches = out_height * out_width;
    const int64_t flat_kernel = channels * kernel_h * kernel_w;

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch, out_channels, out_height, out_width},
                                           DataType::kFLOAT32, input->GetDevice());

    const float *input_data = static_cast<const float *>(input->DataPtr());
    const float *weight_data = static_cast<const float *>(weight->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());

    // (batch, patches, C*kH*kW) scratch: the im2col expansion of every image in the batch.
    const int64_t image_size = channels * height * width;
    std::vector<float> col_buffer(batch * patches * flat_kernel);
    for (int64_t n = 0; n < batch; ++n) {
        Im2col(input_data + n * image_size, channels, height, width, kernel_h, kernel_w, out_height, out_width,
               col_buffer.data() + n * patches * flat_kernel);
    }

    Eigen::Map<const RowMatrix> weight_mat(weight_data, out_channels, flat_kernel);
    for (int64_t n = 0; n < batch; ++n) {
        Eigen::Map<const RowMatrix> col_n(col_buffer.data() + n * patches * flat_kernel, patches, flat_kernel);
        Eigen::Map<RowMatrix> out_n(output_data + n * out_channels * patches, out_channels, patches);
        out_n.noalias() = weight_mat * col_n.transpose();
        if (bias) {
            out_n.colwise()
                += Eigen::Map<const Eigen::VectorXf>(static_cast<const float *>(bias->DataPtr()), out_channels);
        }
    }

    return output;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims) {
    /*
    grad_input(n, c, x, y)
        = sum_{o} sum_{0<=x-u<kH, 0<=y-v<kW} grad_output(n, o, x-u, y-v) * weight(o, c, u, v)
    Computed per image as col2im(grad_output(n)^T * weight), i.e. the transpose of the forward GEMM.
    */

    CHECK_EQ(input_dims.size(), 4);
    const int64_t batch = input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t height = input_dims[2];
    const int64_t width = input_dims[3];

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 4);
    const int64_t out_channels = weight_dims[0];
    const int64_t kernel_h = weight_dims[2];
    const int64_t kernel_w = weight_dims[3];
    CHECK_EQ(weight_dims[1], channels);
    CHECK_GE(height, kernel_h);
    CHECK_GE(width, kernel_w);
    CHECK(weight->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";

    const int64_t out_height = height - kernel_h + 1;
    const int64_t out_width = width - kernel_w + 1;
    const int64_t patches = out_height * out_width;
    const int64_t flat_kernel = channels * kernel_h * kernel_w;

    const auto &grad_dims = grad_output->Dims();
    CHECK_EQ(grad_dims.size(), 4);
    CHECK_EQ(grad_dims[0], batch);
    CHECK_EQ(grad_dims[1], out_channels);
    CHECK_EQ(grad_dims[2], out_height);
    CHECK_EQ(grad_dims[3], out_width);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill(0.0f);

    const float *weight_data = static_cast<const float *>(weight->DataPtr());
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());

    Eigen::Map<const RowMatrix> weight_mat(weight_data, out_channels, flat_kernel);
    const int64_t image_size = channels * height * width;
    std::vector<float> col_buffer(patches * flat_kernel);
    for (int64_t n = 0; n < batch; ++n) {
        Eigen::Map<const RowMatrix> grad_output_n(grad_output_data + n * out_channels * patches, out_channels, patches);
        Eigen::Map<RowMatrix> col_n(col_buffer.data(), patches, flat_kernel);
        col_n.noalias() = grad_output_n.transpose() * weight_mat;
        Col2im(col_buffer.data(), channels, height, width, kernel_h, kernel_w, out_height, out_width,
               grad_input_data + n * image_size);
    }

    return grad_input;
}

std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output) {
    /*
    grad_weight(o, c, u, v) = sum_{n, i, j} grad_output(n, o, i, j) * input(n, c, i+u, j+v)
    Computed per image as grad_output(n) * im2col(input(n)) accumulated over the batch, the
    transpose of the forward GEMM.
    */

    const auto &input_dims = input->Dims();
    CHECK_EQ(input_dims.size(), 4);
    const int64_t batch = input_dims[0];
    const int64_t channels = input_dims[1];
    const int64_t height = input_dims[2];
    const int64_t width = input_dims[3];

    const auto &grad_dims = grad_output->Dims();
    CHECK_EQ(grad_dims.size(), 4);
    CHECK_EQ(grad_dims[0], batch);
    CHECK(input->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";

    // The kernel extent is the difference between the input and output spatial extents.
    const int64_t kernel_h = height - grad_dims[2] + 1;
    const int64_t kernel_w = width - grad_dims[3] + 1;
    CHECK_GT(kernel_h, 0);
    CHECK_GT(kernel_w, 0);

    const int64_t out_channels = grad_dims[1];
    const int64_t out_height = grad_dims[2];
    const int64_t out_width = grad_dims[3];
    const int64_t patches = out_height * out_width;
    const int64_t flat_kernel = channels * kernel_h * kernel_w;

    auto grad_weight = std::make_shared<Tensor>(std::vector<int64_t>{out_channels, channels, kernel_h, kernel_w},
                                                DataType::kFLOAT32, input->GetDevice());
    grad_weight->Fill(0.0f);

    const float *input_data = static_cast<const float *>(input->DataPtr());
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    float *grad_weight_data = static_cast<float *>(grad_weight->DataPtr());

    const int64_t image_size = channels * height * width;
    std::vector<float> col_buffer(batch * patches * flat_kernel);
    for (int64_t n = 0; n < batch; ++n) {
        Im2col(input_data + n * image_size, channels, height, width, kernel_h, kernel_w, out_height, out_width,
               col_buffer.data() + n * patches * flat_kernel);
    }

    Eigen::Map<RowMatrix> grad_weight_mat(grad_weight_data, out_channels, flat_kernel);
    for (int64_t n = 0; n < batch; ++n) {
        Eigen::Map<const RowMatrix> grad_output_n(grad_output_data + n * out_channels * patches, out_channels, patches);
        Eigen::Map<const RowMatrix> col_n(col_buffer.data() + n * patches * flat_kernel, patches, flat_kernel);
        grad_weight_mat.noalias() += grad_output_n * col_n;
    }

    return grad_weight;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_channels) {
    /*
    grad_bias(o) = sum_{n, i, j} grad_output(n, o, i, j)
    */
    const auto &grad_dims = grad_output->Dims();
    CHECK_EQ(grad_dims.size(), 4);
    CHECK_EQ(grad_dims[1], out_channels);
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "Conv2d requires FP32 tensors";

    auto grad_bias
        = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32, grad_output->GetDevice());
    grad_bias->Fill(0.0f);
    Eigen::Map<Eigen::VectorXf> grad_bias_vec(static_cast<float *>(grad_bias->DataPtr()), out_channels);

    const int64_t batch = grad_dims[0];
    const int64_t patches = grad_dims[2] * grad_dims[3];
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    for (int64_t n = 0; n < batch; ++n) {
        Eigen::Map<const RowMatrix> grad_output_n(grad_output_data + n * out_channels * patches, out_channels, patches);
        grad_bias_vec += grad_output_n.rowwise().sum();
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

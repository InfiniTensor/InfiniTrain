// cpu版本用for循环
// cuda版本用线程束

// Conv2dForward: 前向
// Conv2dBackwardInput: 反向
// Conv2dBackwardWeight: 反向
// Conv2dBackwardBias: 反向

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {
    // OH = (H + 2*padding - kH) / stride + 1
    inline int64_t ConvOutputSize(int64_t size, int64_t kernel, int64_t stride, int padding) {
        return (size + 2 * padding - kernel) / stride + 1;
    }

    // grid->block->thread, 让block固定为256个线程
    inline dim3 CalcGrid(int64_t numel) {
        const int64_t block = 256;
        return dim3(static_cast<unsigned int>((numel + block - 1) / block));
    }

}

// 前向
// __global__: 在GPU执行，在CPU启动; 每个线程算出张量的一个元素 out[ni, oi, oy, ox]
__global__ void Conv2dForwardKernel(const float *__restrict__ input,
                                    const float *__restrict__ weight,
                                    const float *__restrict__ bias,
                                    float *__restrict__ output,
                                    int64_t n, int64_t c, int64_t h, int64_t w,
                                    int64_t o, int64_t kh, int64_t kw,
                                    int64_t oh, int64_t ow,
                                    int64_t stride, int64_t padding) {
    // blockIdx.x当前线程所在的block的编号
    // blockDim.x 每个block的线程数（256）
    // threadIdx.x 当前线程在block内的编号
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 越界判断
    if (idx >= n * o * oh * ow) {
        return;
    }

    // 反算坐标
    // idx = ((ni*o + oi) * oh + oy) * ow + ox
    const int64_t ox = idx % ow;
    const int64_t oy = (idx / ow) % oh;
    const int64_t oi = (idx / (ow * oh)) % o;
    const int64_t ni = idx / (ow * oh * o);

    float acc = bias ? bias[oi] : 0.0f;

    for (int64_t ci = 0; ci < c; ++ci) {
        for (int64_t ky = 0; ky < kh; ++ky) {
            const int64_t iy = oy * stride + ky - padding;
            if (iy < 0 || iy >= h) {
                continue;
            }

            for (int64_t kx = 0; kx < kw; ++kx) {
                const int64_t ix = ox * stride + kx - padding;
                if (ix < 0 || ix >= w) {
                    continue;
                }

                acc += input[((ni*c + ci) * h + iy) * w + ix] * weight[((oi*c + ci) * kh + ky) * kw + kx];
            }
        }
    }

    output[idx] = acc;
}

// CPU端
std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &weight,
                                    const std::shared_ptr<Tensor> &bias,
                                    int64_t stride, int64_t padding) {
    const auto &input_dims = input->Dims();
    const auto &weight_dims = weight->Dims();
    CHECK_EQ(input_dims.size(), 4);
    CHECK_EQ(weight_dims.size(), 4);
    CHECK_EQ(input_dims[1], weight_dims[1]);

    const int64_t n = input_dims[0], c = input_dims[1], h = input_dims[2], w = input_dims[3];
    const int64_t o = weight_dims[0], kh = weight_dims[2], kw = weight_dims[3];
    const int64_t oh = ConvOutputSize(h, kh, stride, padding);
    const int64_t ow = ConvOutputSize(w, kw, stride, padding);
    CHECK_GT(oh, 0);
    CHECK_GT(ow, 0);

    // CUDA的tensor构造需要指定device (I/O要在同一机器上)
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{n, o, oh, ow}, DataType::kFLOAT32, input->GetDevice());
    output->Fill(0.0f);

    const int64_t numel = n*o*oh*ow;
    // 并行计算启动
    Conv2dForwardKernel<<<CalcGrid(numel), 256>>>(
        static_cast<const float *>(input->DataPtr()),// DataPtr()返回GPU显存指针
        static_cast<const float *>(weight->DataPtr()),
        bias ? static_cast<const float *>(bias->DataPtr()) : nullptr,
        static_cast<float *>(output->DataPtr()), n, c, h, w, o, kh, kw, oh, ow, stride, padding);
    
    // GPU kernel异步启动，所以同步要等待算完
    cudaDeviceSynchronize();
    return output;
}


// 反向1：输入的梯度
__global__ void Conv2dBackwardInputKernel(const float *__restrict__ weight,
                                        const float *__restrict__ grad_output,
                                        float *__restrict__ grad_input,
                                        int64_t n, int64_t c, int64_t h, int64_t w,
                                        int64_t o, int64_t kh, int64_t kw, int64_t oh, int64_t ow,
                                        int64_t stride, int64_t padding) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n*c*h*w) {
        return;
    }

    const int64_t ix = idx % w;
    const int64_t iy = (idx / w) % h; 
    const int64_t ci = (idx / (w * h)) % c;
    const int64_t ni = idx / (w * h * c);

    float acc = 0.0f; 
    for (int64_t oi = 0; oi < o; ++oi) {
        for (int64_t ky = 0; ky < kh; ++ky) {
            const int64_t rem_y = iy + padding - ky;

            if (rem_y < 0 || rem_y % stride != 0) {
                continue;
            }

            const int64_t oy = rem_y / stride;
            if (oy >= oh) {
                continue;
            }

            for (int64_t kx = 0; kx < kw; ++kx) {
                const int64_t rem_x = ix + padding - kx;
                if (rem_x < 0 || rem_x % stride != 0) {
                    continue;
                }

                const int64_t ox = rem_x / stride;
                if (ox >= ow) {
                    continue;
                }

                acc += grad_output[((ni*o + oi) * oh + oy) * ow + ox] * weight[((oi * c + ci) * kh + ky) * kw + kx];
            }
        }
    }
    grad_input[idx] = acc;
}

std::shared_ptr<Tensor> Conv2dBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims,
                                            int64_t stride, int64_t padding) {
    const auto &grad_dims = grad_output->Dims();
    const int64_t n = input_dims[0], c = input_dims[1], h = input_dims[2], w = input_dims[3];
    const int64_t o = weight->Dims()[0], kh = weight->Dims()[2], kw = weight->Dims()[3];
    const int64_t oh = grad_dims[2], ow = grad_dims[3];


    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill(0.0f);

    const int64_t numel = n*c*h*w;
    Conv2dBackwardInputKernel<<<CalcGrid(numel), 256>>>(
        static_cast<const float *>(weight->DataPtr()), 
        static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()), n, c, h, w, o, kh, kw, oh, ow, stride, padding
    );

    cudaDeviceSynchronize();
    return grad_input;
}

// 反向2： 权重的梯度
__global__ void Conv2dBackwardWeightKernel(const float *__restrict__ input,
                                        const float *__restrict__ grad_output,
                                        float *__restrict__ grad_weight,
                                        int64_t n, int64_t c, int64_t h, int64_t w,
                                        int64_t o, int64_t kh, int64_t kw,
                                        int64_t oh, int64_t ow,
                                        int64_t stride, int64_t padding){
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= o*c*kh*kw) {
        return;
    }

    const int64_t kx = idx % kw; 
    const int64_t ky = (idx / kw) % kh;
    const int64_t ci = (idx / (kw * kh)) % c;
    const int64_t oi = idx / (kw * kh * c);

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

                acc += input[((ni*c + ci) * h + iy) * w + ix] * grad_output[((ni * o + oi) * oh + oy) * ow + ox];
            }
        }
    }
    grad_weight[idx] = acc;
}


std::shared_ptr<Tensor> Conv2dBackwardWeight(const std::shared_ptr<Tensor> &input,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &weight_dims,
                                            int64_t stride, int64_t padding) {
    const auto &input_dims = input->Dims();
    const auto &grad_dims = grad_output->Dims();
    const int64_t n = input_dims[0], c = input_dims[1], h = input_dims[2], w = input_dims[3];
    const int64_t o = weight_dims[0], kh = weight_dims[2], kw = weight_dims[3];
    const int64_t oh = grad_dims[2], ow = grad_dims[3];

    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_weight->Fill(0.0f);

    const int64_t numel = o*c*kh*kw;
    Conv2dBackwardWeightKernel<<<CalcGrid(numel), 256>>>(
        static_cast<const float *>(input->DataPtr()),
        static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_weight->DataPtr()), n, c, h, w, o, kh, kw, oh, ow, stride, padding
    );

    cudaDeviceSynchronize();
    return grad_weight;
}


// 反向3：偏置的梯度
__global__ void Conv2dBackwardBiasKernel(const float *__restrict__ grad_output,
                                        float *__restrict__ grad_bias,
                                        int64_t n, int64_t o, int64_t oh, int64_t ow) {
    const int64_t oi = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (oi >= o) {
        return;
    }

    float acc = 0.0f;
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t oy = 0; oy < oh; ++oy) {
            for (int64_t ox = 0; ox < ow; ++ox) {
                acc += grad_output[((ni*o + oi) * oh + oy) * ow + ox];
            }
        }
    }
    grad_bias[oi] = acc;
}

std::shared_ptr<Tensor> Conv2dBackwardBias(const std::shared_ptr<Tensor> &grad_output,
                                        int64_t out_channels) {
    const auto &grad_dims = grad_output->Dims();
    const int64_t n = grad_dims[0], oh = grad_dims[2], ow = grad_dims[3];

    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_channels}, DataType::kFLOAT32, grad_output->GetDevice());
    grad_bias->Fill(0.0f);

    Conv2dBackwardBiasKernel<<<CalcGrid(out_channels), 256>>>(
        static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_bias->DataPtr()), n, out_channels, oh, ow
    );
    cudaDeviceSynchronize();
    return grad_bias;

}


}

// 注册dispatcher
#define REGISTER_CUDA_CONV2D_KERNEL(kernel_name) \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CONV2D_KERNEL(Conv2dForward)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardInput)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardWeight)
REGISTER_CUDA_CONV2D_KERNEL(Conv2dBackwardBias)

#undef REGISTER_CUDA_CONV2D_KERNEL
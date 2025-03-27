#include "infini_train/include/ops.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "cublas_v2.h"
#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::ops {
namespace {
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
}

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

CUDALinear::CUDALinear(Tensor *weight, Tensor *bias) : Linear(weight, bias) {}

std::vector<std::shared_ptr<Tensor>> CUDALinear::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 1);

    auto &x = input_tensors_[0];
    CHECK_EQ(x->Dims().size(), 2);
    CHECK_EQ(x->Dims()[1], in_dim_);
    const int bs = x->Dims()[0];

    auto y = std::make_shared<Tensor>(std::vector<int64_t>{bs, out_dim_}, DataType::kFLOAT32,
                                      Device(DeviceType::kCUDA, 0));
    for (int idx = 0; idx < bs; ++idx) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<float *>(y->DataPtr()) + idx * out_dim_, b_->DataPtr(),
                              out_dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Y = X * W + B
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_dim_, bs, in_dim_, &alpha,
                             reinterpret_cast<const float *>(w_->DataPtr()), out_dim_,
                             reinterpret_cast<const float *>(x->DataPtr()), in_dim_, &beta,
                             reinterpret_cast<float *>(y->DataPtr()), out_dim_));

    CUBLAS_CHECK(cublasDestroy(handle));

    return {y};
}

__global__ void set_ones(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

void CUDALinear::BackwardImpl() {

    auto &y = output_tensors_[0];
    auto &x = input_tensors_[0];
    const int bs = x->Dims()[0];

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // dX = dY * W^T
    if (x->Gradient()) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim_, bs, out_dim_, &alpha,
                                 reinterpret_cast<const float *>(w_->DataPtr()), out_dim_,
                                 reinterpret_cast<const float *>(y->Gradient()->DataPtr()), out_dim_, &beta,
                                 reinterpret_cast<float *>(x->Gradient()->DataPtr()), in_dim_));
    }

    // dW = X^T * dY
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_dim_, in_dim_, bs, &alpha,
                             reinterpret_cast<const float *>(y->Gradient()->DataPtr()), out_dim_,
                             reinterpret_cast<const float *>(x->DataPtr()), in_dim_, &beta,
                             reinterpret_cast<float *>(w_->Gradient()->DataPtr()), out_dim_));
    // FIXME(dcj): remove this sync
    CUDA_CHECK(cudaDeviceSynchronize());

    // dB = \sum_i(i=0, bs-1) dY_i
    // TODO(dcj): use thrust::fill or reduce kernel do this
    auto ones = std::make_shared<Tensor>(std::vector<int64_t>{bs}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    float *d_ptr = reinterpret_cast<float *>(ones->DataPtr());

    // TODO(dcj): use const variable for threads_per_block and num_blocks
    int threads_per_block = 256;
    int num_blocks = (bs + threads_per_block - 1) / threads_per_block;
    set_ones<<<num_blocks, threads_per_block>>>(d_ptr, bs);

    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, out_dim_, bs, &alpha,
                             reinterpret_cast<const float *>(y->Gradient()->DataPtr()), out_dim_,
                             reinterpret_cast<const float *>(ones->DataPtr()), 1, &beta,
                             reinterpret_cast<float *>(b_->Gradient()->DataPtr()), 1));

    CUBLAS_CHECK(cublasDestroy(handle));
}

// Sigmoid CUDA Kernel
__global__ void SigmoidKernel(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

// Sigmoid backward CUDA Kernel
__global__ void SigmoidBackwardKernel(const float *output, const float *grad_output, float *grad_input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] = grad_output[i] * output[i] * (1 - output[i]);
    }
}

// Sigmoid forward
std::vector<std::shared_ptr<Tensor>> CUDASigmoid::ForwardImpl() {
    auto &input = input_tensors_[0];
    int n = input->NumElements();

    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    SigmoidKernel<<<num_blocks, threads_per_block>>>(reinterpret_cast<const float *>(input->DataPtr()),
                                                     reinterpret_cast<float *>(output->DataPtr()), n);

    return {output};
}

// Sigmoid backward
void CUDASigmoid::BackwardImpl() {
    auto &output = output_tensors_[0];
    auto &input = input_tensors_[0];
    int n = input->NumElements();

    if (input->Gradient()) {
        int threads_per_block = 256;
        int num_blocks = (n + threads_per_block - 1) / threads_per_block;

        SigmoidBackwardKernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const float *>(output->DataPtr()),
            reinterpret_cast<const float *>(output->Gradient()->DataPtr()),
            reinterpret_cast<float *>(input->Gradient()->DataPtr()), n);
    }
}

// CrossEntropy CUDA Kernel
__global__ void CrossEntropyKernel(const float *y_pred, const uint8_t *y_target, float *loss, int batch_size,
                                   int num_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float max_logit = kNegativeInfinity;
        for (int j = 0; j < num_classes; j++) { max_logit = max(max_logit, y_pred[i * num_classes + j]); }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) { sum_exp += expf(y_pred[i * num_classes + j] - max_logit); }
        loss[i] = -logf(expf(y_pred[i * num_classes + y_target[i]] - max_logit) / sum_exp);
    }
}

// CrossEntropy backward CUDA Kernel
__global__ void CrossEntropyBackwardKernel(float *input, float *input_grad, uint8_t *target, int bs, int num_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bs) {
        float max_logit = kNegativeInfinity;
        for (int j = 0; j < num_classes; j++) { max_logit = max(max_logit, input[i * num_classes + j]); }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) { sum_exp += expf(input[i * num_classes + j] - max_logit); }
        for (int j = 0; j < num_classes; j++) {
            int idx = i * num_classes + j;
            input_grad[idx] += (expf(input[idx] - max_logit) / sum_exp - (j == target[i] ? 1.0f : 0.0f)) / bs;
        }
    }
}

// CrossEntropy forward
std::vector<std::shared_ptr<Tensor>> CUDACrossEntropy::ForwardImpl() {
    auto &y_pred = input_tensors_[0];
    auto &y_target = input_tensors_[1];

    int batch_size = y_pred->Dims()[0];
    int num_classes = y_pred->Dims()[1];

    auto batched_loss
        = std::make_shared<Tensor>(std::vector<int64_t>{batch_size}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    CrossEntropyKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const float *>(y_pred->DataPtr()), reinterpret_cast<const uint8_t *>(y_target->DataPtr()),
        reinterpret_cast<float *>(batched_loss->DataPtr()), batch_size, num_classes);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto loss_cpu = batched_loss->To(Device());
    auto loss = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, Device());
    reinterpret_cast<float *>(loss->DataPtr())[0]
        = std::accumulate(reinterpret_cast<const float *>(loss_cpu.DataPtr()),
                          reinterpret_cast<const float *>(loss_cpu.DataPtr()) + batch_size, 0.0f)
        / batch_size;

    return {std::make_shared<Tensor>(loss->To(Device(DeviceType::kCUDA, 0)))};
}

// CrossEntropy backward
void CUDACrossEntropy::BackwardImpl() {
    auto &input = input_tensors_[0];
    auto &target = input_tensors_[1];

    int bs = input->Dims()[0];
    int num_classes = input->Dims()[1];

    if (input->Gradient()) {
        int threads_per_block = 256;
        int num_blocks = (bs + threads_per_block - 1) / threads_per_block;

        CrossEntropyBackwardKernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float *>(input->DataPtr()), reinterpret_cast<float *>(input->Gradient()->DataPtr()),
            reinterpret_cast<uint8_t *>(target->DataPtr()), bs, num_classes);
    }
}
} // namespace infini_train::ops

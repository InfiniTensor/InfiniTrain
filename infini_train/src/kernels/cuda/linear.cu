#include "infini_train/include/device.h"
#include "infini_train/include/kernels/cuda/linear.h"

#include <memory>
#include <tuple>

#include "cublas_v2.h"
#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

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

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias) {
    CHECK_EQ(input->Dims().size(), 2);
    const int bs = input->Dims()[0];
    const int in_feature = input->Dims()[1];
    CHECK_EQ(weight->Dims().size(), 2);
    CHECK_EQ(in_feature, weight->Dims()[0]);
    const int out_feature = weight->Dims()[1];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{bs, out_feature}, DataType::kFLOAT32,
                                           Device(DeviceType::kCUDA, 0));

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_feature);
        for (int i = 0; i < bs; ++i) {
            cudaMemcpy(static_cast<float *>(output->DataPtr()) + i * out_feature, bias->DataPtr(),
                       out_feature * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    } else {
        output->Fill<float>(0.0f);
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // output = alpha * (input * weight) + beta * output
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_feature, bs, in_feature, &alpha,
                static_cast<const float *>(weight->DataPtr()), out_feature,
                static_cast<const float *>(input->DataPtr()), in_feature, &beta,
                static_cast<float *>(output->DataPtr()), out_feature);

    cublasDestroy(handle);

    return {output};
}

__global__ void set_ones(float *data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = 1.0f;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
               const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 2);
    const int bs = input->Dims()[0];
    const int in_feature = input->Dims()[1];
    CHECK_EQ(weight->Dims().size(), 2);
    CHECK_EQ(in_feature, weight->Dims()[0]);
    const int out_feature = weight->Dims()[1];
    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_feature);
    }
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(bias->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
        grad_bias->Fill<float>(0.0f);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // d_input = d_output * weight^T --> d_input^T = weight * d_output^T
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_feature, bs, out_feature, &alpha,
                             static_cast<const float *>(weight->DataPtr()), out_feature,
                             static_cast<const float *>(grad_output->DataPtr()), out_feature, &beta,
                             static_cast<float *>(grad_input->DataPtr()), in_feature));

    // d_weight = input^T * d_output --> d_weight^T = d_output^T * input
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_feature, in_feature, bs, &alpha,
                             static_cast<const float *>(grad_output->DataPtr()), out_feature,
                             static_cast<const float *>(input->DataPtr()), in_feature, &beta,
                             static_cast<float *>(grad_weight->DataPtr()), out_feature));
    // FIXME(dcj): remove this sync
    CUDA_CHECK(cudaDeviceSynchronize());

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    // TODO(dcj): use thrust::fill or reduce kernel do this
    if (bias) {
        auto ones
            = std::make_shared<Tensor>(std::vector<int64_t>{bs}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
        float *ones_ptr = static_cast<float *>(ones->DataPtr());

        int threads_per_block = 256;
        int num_blocks = (bs + threads_per_block - 1) / threads_per_block;

        set_ones<<<num_blocks, threads_per_block>>>(ones_ptr, bs);

        CUBLAS_CHECK(cublasSgemv(
            handle, CUBLAS_OP_N, out_feature, bs, &alpha, static_cast<const float *>(grad_output->DataPtr()),
            out_feature, static_cast<float *>(ones_ptr), 1, &beta, static_cast<float *>(grad_bias->DataPtr()), 1));
    }

    cublasDestroy(handle);

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

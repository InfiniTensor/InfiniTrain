#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <numeric>

#include "infini_train/include/common/cuda/common_cuda.cuh"

namespace infini_train::kernels::cuda {
template <size_t BLOCK_SIZE, typename T>
__global__ void MSEForwardKernel(const T *__restrict__ input, const T *__restrict__ target, float *__restrict__ sum,
                                 size_t N) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;

    float thread_sum = 0.0f;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        float diff = infini_train::common::cuda::Cast<float>(input[idx])
                   - infini_train::common::cuda::Cast<float>(target[idx]);
        thread_sum += diff * diff;
    }

    float block_sum = BlockReduce(reduce_storage).Sum(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(sum, block_sum);
    }
}

template <typename T>
__global__ void MSEFinalizeKernel(T *__restrict__ loss_out, const float *__restrict__ sum, size_t N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mean = (*sum) / static_cast<float>(N);
        *loss_out = infini_train::common::cuda::Cast<T>(mean);
    }
}

std::shared_ptr<Tensor> MSEForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &target) {
    CHECK_EQ(input->NumElements(), target->NumElements());
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(target->Dtype()));

    const size_t n = input->NumElements();
    auto dtype = input->Dtype();

    auto loss = std::make_shared<Tensor>(std::vector<int64_t>{}, dtype, input->GetDevice());

    constexpr int BLOCK_SIZE = 256;
    int grid = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // FIXME(zbl): get maxGridX dynamically, or perform per-sample block reduce(grid = bs)
    grid = std::max(1, std::min(grid, 65535));

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    auto stream = cuda_device->Stream();

    float *d_sum = nullptr;
    cudaMallocAsync(&d_sum, sizeof(float), stream);
    cudaMemsetAsync(d_sum, 0, sizeof(float), stream);

    return DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            const T *in_ptr = static_cast<const T *>(input->DataPtr());
            const T *tgt_ptr = static_cast<const T *>(target->DataPtr());
            T *loss_ptr = static_cast<T *>(loss->DataPtr());

            MSEForwardKernel<BLOCK_SIZE, T><<<grid, BLOCK_SIZE, 0, stream>>>(in_ptr, tgt_ptr, d_sum, n);
            MSEFinalizeKernel<T><<<1, 1, 0, stream>>>(loss_ptr, d_sum, n);

            cudaFreeAsync(d_sum, stream);
            return loss;
        },
        "CUDA MSEForward");
}

template <size_t BLOCK_SIZE, typename T>
__global__ void MSEBackwardKernel(const T *__restrict__ input, const T *__restrict__ target, T *__restrict__ grad_input,
                                  const T *__restrict__ grad_output, size_t N) {
    const float go = infini_train::common::cuda::Cast<float>(grad_output[0]);
    const float scale = 2.0f / static_cast<float>(N);

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        float diff = infini_train::common::cuda::Cast<float>(input[idx])
                   - infini_train::common::cuda::Cast<float>(target[idx]);
        float g = scale * go * diff;
        grad_input[idx] = infini_train::common::cuda::Cast<T>(g);
    }
}

std::shared_ptr<Tensor> MSEBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &target,
                                    const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(grad_output->Dims().size(), 0);
    CHECK_EQ(input->NumElements(), target->NumElements());
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(target->Dtype()));
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(grad_output->Dtype()));

    const size_t n = input->NumElements();
    auto dtype = input->Dtype();

    auto grad_input = std::make_shared<Tensor>(input->Dims(), dtype, grad_output->GetDevice());

    constexpr int BLOCK_SIZE = 256;
    int grid = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // FIXME(zbl): get maxGridX dynamically, or perform per-sample block reduce(grid = bs)
    grid = std::max(1, std::min(grid, 65535));

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());

    DispatchFunc<INFINI_ALL_FLOATING_TYPES>(
        dtype,
        [=]<typename T>() {
            const T *in_ptr = static_cast<const T *>(input->DataPtr());
            const T *tgt_ptr = static_cast<const T *>(target->DataPtr());
            const T *go_ptr = static_cast<const T *>(grad_output->DataPtr());
            T *gin_ptr = static_cast<T *>(grad_input->DataPtr());

            MSEBackwardKernel<BLOCK_SIZE, T>
                <<<grid, BLOCK_SIZE, 0, cuda_device->Stream()>>>(in_ptr, tgt_ptr, gin_ptr, go_ptr, n);
        },
        "CUDA MSEBackward");

    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_MSE_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_MSE_KERNEL(MSEForward)
REGISTER_CUDA_MSE_KERNEL(MSEBackward)

#undef REGISTER_CUDA_MSE_KERNEL
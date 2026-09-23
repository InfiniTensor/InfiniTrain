#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {

namespace {

constexpr int kMaxScatterDims = 8;

struct ScatterMeta {
    int64_t rank;
    int64_t dim;
    int64_t input_strides[kMaxScatterDims];
    int64_t index_strides[kMaxScatterDims];
};

ScatterMeta MakeScatterMeta(const std::vector<int64_t> &input_dims, const std::vector<int64_t> &index_dims,
                            int64_t dim) {
    CHECK_EQ(input_dims.size(), index_dims.size());
    CHECK_LE(input_dims.size(), static_cast<size_t>(kMaxScatterDims));
    ScatterMeta meta{};
    meta.rank = static_cast<int64_t>(input_dims.size());
    meta.dim = dim;
    int64_t input_stride = 1;
    int64_t index_stride = 1;
    for (int64_t axis = meta.rank - 1; axis >= 0; --axis) {
        meta.input_strides[axis] = input_stride;
        meta.index_strides[axis] = index_stride;
        input_stride *= input_dims[axis];
        index_stride *= index_dims[axis];
    }
    return meta;
}

__device__ int64_t ScatterOutputOffset(int64_t linear_idx, int64_t scatter_index, ScatterMeta meta) {
    int64_t output_offset = 0;
    for (int64_t axis = 0; axis < meta.rank; ++axis) {
        const int64_t source_coordinate = linear_idx / meta.index_strides[axis];
        const int64_t coordinate = axis == meta.dim ? scatter_index : source_coordinate;
        output_offset += coordinate * meta.input_strides[axis];
        linear_idx %= meta.index_strides[axis];
    }
    return output_offset;
}

} // namespace

template <typename T>
__global__ void ScatterSrcForwardKernel(const T *__restrict__ src, const int64_t *__restrict__ indices,
                                        T *__restrict__ output, ScatterMeta meta, int64_t total) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    output[ScatterOutputOffset(idx, indices[idx], meta)] = src[idx];
}

std::shared_ptr<Tensor> ScatterSrcForward(const std::shared_ptr<Tensor> &input, int64_t dim,
                                         const std::shared_ptr<Tensor> &indices,
                                         const std::shared_ptr<Tensor> &src) {
    const int64_t rank = static_cast<int64_t>(input->Dims().size());
    CHECK_EQ(src->Dims().size(), input->Dims().size());
    CHECK(indices->Dims() == src->Dims());
    CHECK_GE(dim, 0);
    CHECK_LT(dim, rank);
    CHECK(indices->Dtype() == DataType::kINT64);
    CHECK(input->Dtype() == src->Dtype());
    for (int64_t axis = 0; axis < rank; ++axis) {
        if (axis != dim) { CHECK_LE(indices->Dims()[axis], input->Dims()[axis]); }
    }

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->hip_stream();
    HIP_CHECK(hipMemcpyAsync(output->DataPtr(), input->DataPtr(), input->SizeInBytes(), hipMemcpyDeviceToDevice, stream));
    const auto meta = MakeScatterMeta(input->Dims(), indices->Dims(), dim);
    const int64_t total = src->NumElements();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            ScatterSrcForwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(src->DataPtr()), static_cast<const int64_t *>(indices->DataPtr()),
                static_cast<T *>(output->DataPtr()), meta, total);
            HIP_CHECK(hipGetLastError());
        },
        "HIP ScatterSrcForward");
    return output;
}

template <typename T>
__global__ void ScatterSrcBackwardKernel(const T *__restrict__ grad_output,
                                         const int64_t *__restrict__ indices, T *__restrict__ grad_input,
                                         T *__restrict__ grad_src, ScatterMeta meta, int64_t total) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    const int64_t output_idx = ScatterOutputOffset(idx, indices[idx], meta);
    grad_src[idx] = grad_output[output_idx];
    grad_input[output_idx] = common::dcu::Cast<T>(0.0f);
}

std::vector<std::shared_ptr<Tensor>> ScatterSrcBackward(const std::shared_ptr<Tensor> &grad_output, int64_t dim,
                                                        const std::shared_ptr<Tensor> &indices) {
    const int64_t rank = static_cast<int64_t>(grad_output->Dims().size());
    CHECK_GE(dim, 0);
    CHECK_LT(dim, rank);
    const auto meta = MakeScatterMeta(grad_output->Dims(), indices->Dims(), dim);
    auto grad_input
        = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    auto grad_src = std::make_shared<Tensor>(indices->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    const auto device = grad_output->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->hip_stream();
    HIP_CHECK(hipMemcpyAsync(grad_input->DataPtr(), grad_output->DataPtr(), grad_output->SizeInBytes(),
                             hipMemcpyDeviceToDevice, stream));
    const int64_t total = grad_src->NumElements();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            ScatterSrcBackwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<const int64_t *>(indices->DataPtr()),
                static_cast<T *>(grad_input->DataPtr()), static_cast<T *>(grad_src->DataPtr()), meta, total);
            HIP_CHECK(hipGetLastError());
        },
        "HIP ScatterSrcBackward");
    return {grad_input, grad_src};
}

} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_SCATTER_SRC_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, kernel_name, infini_train::kernels::dcu::kernel_name)

REGISTER_HIP_SCATTER_SRC_KERNEL(ScatterSrcForward)
REGISTER_HIP_SCATTER_SRC_KERNEL(ScatterSrcBackward)

#undef REGISTER_HIP_SCATTER_SRC_KERNEL

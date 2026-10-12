#include <cstdint>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {
struct SliceMeta {
    int64_t new_dims[8];
    int64_t starts[8];
    int64_t steps[8];
    int64_t in_strides[8];
    int64_t out_strides[8];
    int32_t num_dims;
};


template <typename T>
__global__ void SliceForwardKernel(const T *input, T *output, SliceMeta meta, int64_t total_elements) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) {
        return;
    }

    int64_t in_index = 0;
    for (int i = 0; i < meta.num_dims; ++i) {
        int64_t idx = (out_idx / meta.out_strides[i]) % meta.new_dims[i];
        in_index += (meta.starts[i] + idx * meta.steps[i]) * meta.in_strides[i];
    }

    output[out_idx] = input[in_index];
}

std::shared_ptr<Tensor> SliceForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                                     const std::vector<int64_t> &ends, const std::vector<int64_t> &steps) {
    CHECK_EQ(starts.size(), ends.size());
    CHECK_EQ(starts.size(), steps.size());
    auto &dims = input->Dims();
    CHECK_EQ(starts.size(), dims.size());
    const int64_t num_dims = dims.size();
    CHECK_LE(num_dims, 8);

    std::vector<int64_t> new_dims;
    for (int i = 0; i < starts.size(); ++i) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto dtype = input->Dtype();
    auto new_tensor = std::make_shared<Tensor>(new_dims, dtype, input->GetDevice());
    // SliceForwardKernel writes every output element exactly once.

    std::vector<int64_t> src_strides(dims.size(), 0), dst_strides(new_dims.size(), 0);
    int64_t stride = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
        src_strides[i] = stride;
        stride *= dims[i];
    }

    stride = 1;
    for (int i = new_dims.size() - 1; i >= 0; --i) {
        dst_strides[i] = stride;
        stride *= new_dims[i];
    }

    int64_t total_elements = stride;

    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->hip_stream();

    SliceMeta meta{};
    meta.num_dims = static_cast<int32_t>(num_dims);
    for (int i = 0; i < num_dims; ++i) {
        meta.new_dims[i] = new_dims[i];
        meta.starts[i] = starts[i];
        meta.steps[i] = steps[i];
        meta.in_strides[i] = src_strides[i];
        meta.out_strides[i] = dst_strides[i];
    }

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    core::dcu::DispatchDcuFunc<INFINI_ALL_TYPES>(
        dtype,
        [=]<typename T>() {
            SliceForwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const T *>(input->DataPtr()), static_cast<T *>(new_tensor->DataPtr()), meta,
                total_elements);
        },
        "HIP SliceForward");

    return new_tensor;
}

template <typename T>
__global__ void SliceBackwardKernel(const T *grad_output, T *grad_input, SliceMeta meta,
                                    int64_t total_elements) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) {
        return;
    }

    int64_t in_index = 0;
    for (int i = 0; i < meta.num_dims; ++i) {
        int64_t idx = (out_idx / meta.out_strides[i]) % meta.new_dims[i];
        in_index += (meta.starts[i] + idx * meta.steps[i]) * meta.in_strides[i];
    }
    grad_input[in_index] = grad_output[out_idx];
}

std::shared_ptr<Tensor> SliceBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                      const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                      const std::vector<int64_t> &steps) {
    CHECK_EQ(starts.size(), ends.size());
    CHECK_EQ(starts.size(), steps.size());
    auto &dims = input->Dims();
    CHECK_EQ(starts.size(), dims.size());
    const int64_t num_dims = dims.size();
    CHECK_LE(num_dims, 8);

    std::vector<int64_t> new_dims;
    for (int i = 0; i < starts.size(); ++i) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto grad_output_dtype = grad_output->Dtype();
    auto grad_input = std::make_shared<Tensor>(input->Dims(), grad_output_dtype, grad_output->GetDevice());
    grad_input->Fill(0.0);

    std::vector<int64_t> src_strides(dims.size());
    int64_t stride = 1;
    for (int i = src_strides.size() - 1; i >= 0; --i) {
        src_strides[i] = stride;
        stride *= dims[i];
    }

    std::vector<int64_t> dst_strides(new_dims.size());
    stride = 1;
    for (int i = dst_strides.size() - 1; i >= 0; --i) {
        dst_strides[i] = stride;
        stride *= new_dims[i];
    }

    int64_t total_elements = stride;

    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->hip_stream();
    SliceMeta meta{};
    meta.num_dims = static_cast<int32_t>(num_dims);
    for (int i = 0; i < num_dims; ++i) {
        meta.new_dims[i] = new_dims[i];
        meta.starts[i] = starts[i];
        meta.steps[i] = steps[i];
        meta.in_strides[i] = src_strides[i];
        meta.out_strides[i] = dst_strides[i];
    }

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    core::dcu::DispatchDcuFunc<INFINI_ALL_TYPES>(
        grad_output_dtype,
        [=]<typename T>() {
            SliceBackwardKernel<<<num_blocks, threads_per_block, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_input->DataPtr()), meta,
                total_elements);
        },
        "HIP SliceBackward");

    return grad_input;
}
} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_SLICE_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, kernel_name, infini_train::kernels::dcu::kernel_name)

REGISTER_HIP_SLICE_KERNEL(SliceForward)
REGISTER_HIP_SLICE_KERNEL(SliceBackward)

#undef REGISTER_HIP_SLICE_KERNEL

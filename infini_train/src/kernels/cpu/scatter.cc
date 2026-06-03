#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> ScatterForward(const std::shared_ptr<Tensor> &values, const std::shared_ptr<Tensor> &indices,
                                       const std::vector<int64_t> &output_dims) {
    CHECK(indices->Dtype() == DataType::kINT64) << "CPU ScatterForward expects int64 indices";
    CHECK(values->Dims() == indices->Dims());
    CHECK(!output_dims.empty());
    CHECK_EQ(values->Dims().size(), output_dims.size());
    CHECK_GT(values->Dims().back(), 0);
    CHECK_GT(output_dims.back(), 0);

    const int64_t topk = values->Dims().back();
    const int64_t num_experts = output_dims.back();
    const int64_t rows = values->NumElements() / topk;
    size_t output_numel = 1;
    for (const auto dim : output_dims) { output_numel *= static_cast<size_t>(dim); }
    CHECK_EQ(output_numel, static_cast<size_t>(rows * num_experts));

    auto output = std::make_shared<Tensor>(output_dims, values->Dtype(), values->GetDevice());
    std::memset(output->DataPtr(), 0, output->SizeInBytes());

    const size_t elem_size = kDataTypeToSize.at(values->Dtype());
    const auto *src = static_cast<const std::byte *>(values->DataPtr());
    auto *dst = static_cast<std::byte *>(output->DataPtr());
    const auto *idx = static_cast<const int64_t *>(indices->DataPtr());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t selected = 0; selected < topk; ++selected) {
            const int64_t expert_idx = idx[row * topk + selected];
            CHECK_GE(expert_idx, 0);
            CHECK_LT(expert_idx, num_experts);
            std::memcpy(dst + (row * num_experts + expert_idx) * elem_size, src + (row * topk + selected) * elem_size,
                        elem_size);
        }
    }

    return output;
}

std::shared_ptr<Tensor> ScatterBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &indices) {
    CHECK(indices->Dtype() == DataType::kINT64) << "CPU ScatterBackward expects int64 indices";
    CHECK_GE(grad_output->Dims().size(), 1);
    CHECK_GE(indices->Dims().size(), 1);

    const int64_t num_experts = grad_output->Dims().back();
    const int64_t topk = indices->Dims().back();
    const int64_t rows = indices->NumElements() / topk;
    CHECK_EQ(grad_output->NumElements(), static_cast<size_t>(rows * num_experts));

    auto grad_values = std::make_shared<Tensor>(indices->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    const size_t elem_size = kDataTypeToSize.at(grad_output->Dtype());
    const auto *src = static_cast<const std::byte *>(grad_output->DataPtr());
    auto *dst = static_cast<std::byte *>(grad_values->DataPtr());
    const auto *idx = static_cast<const int64_t *>(indices->DataPtr());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t selected = 0; selected < topk; ++selected) {
            const int64_t expert_idx = idx[row * topk + selected];
            CHECK_GE(expert_idx, 0);
            CHECK_LT(expert_idx, num_experts);
            std::memcpy(dst + (row * topk + selected) * elem_size, src + (row * num_experts + expert_idx) * elem_size,
                        elem_size);
        }
    }

    return grad_values;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_SCATTER_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_SCATTER_KERNEL(ScatterForward)
REGISTER_CPU_SCATTER_KERNEL(ScatterBackward)

#undef REGISTER_CPU_SCATTER_KERNEL

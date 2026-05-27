#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::vector<std::shared_ptr<Tensor>> TopKForward(const std::shared_ptr<Tensor> &input, int64_t topk, int64_t dim,
                                                 bool largest, bool sorted) {
    CHECK(input->Dtype() == DataType::kFLOAT32) << "CPU TopKForward currently supports float32 only";
    CHECK_GE(input->Dims().size(), 1);
    (void)sorted;

    const auto &dims = input->Dims();
    if (dim < 0) {
        dim += static_cast<int64_t>(dims.size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(dims.size()));

    const int64_t dim_size = dims[dim];
    CHECK_GT(dim_size, 0);
    CHECK_GT(topk, 0);
    CHECK_LE(topk, dim_size);

    int64_t outer_size = 1;
    for (int64_t idx = 0; idx < dim; ++idx) { outer_size *= dims[idx]; }
    int64_t inner_size = 1;
    for (size_t idx = static_cast<size_t>(dim) + 1; idx < dims.size(); ++idx) { inner_size *= dims[idx]; }

    auto topk_dims = dims;
    topk_dims[dim] = topk;
    auto top_values = std::make_shared<Tensor>(topk_dims, input->Dtype(), input->GetDevice());
    auto top_indices = std::make_shared<Tensor>(topk_dims, DataType::kINT64, input->GetDevice());

    const float *in = static_cast<const float *>(input->DataPtr());
    float *values = static_cast<float *>(top_values->DataPtr());
    int64_t *indices = static_cast<int64_t *>(top_indices->DataPtr());
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            std::vector<bool> selected_indices(dim_size, false);
            for (int64_t selected = 0; selected < topk; ++selected) {
                int64_t best_idx = -1;
                float best_value
                    = largest ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
                for (int64_t idx = 0; idx < dim_size; ++idx) {
                    if (selected_indices[idx]) {
                        continue;
                    }
                    const float value = in[outer * dim_size * inner_size + idx * inner_size + inner];
                    const bool better = largest ? value > best_value : value < best_value;
                    if (better) {
                        best_value = value;
                        best_idx = idx;
                    }
                }
                CHECK_GE(best_idx, 0);
                selected_indices[best_idx] = true;
                const int64_t out_offset = outer * topk * inner_size + selected * inner_size + inner;
                values[out_offset] = best_value;
                indices[out_offset] = best_idx;
            }
        }
    }

    return {top_values, top_indices};
}

std::shared_ptr<Tensor> TopKBackward(const std::shared_ptr<Tensor> &grad_values, const std::shared_ptr<Tensor> &indices,
                                     const std::vector<int64_t> &input_dims, int64_t dim) {
    CHECK(indices->Dtype() == DataType::kINT64) << "CPU TopKBackward expects int64 indices";
    CHECK(grad_values->Dims() == indices->Dims());
    CHECK(!input_dims.empty());
    if (dim < 0) {
        dim += static_cast<int64_t>(input_dims.size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(input_dims.size()));

    const int64_t dim_size = input_dims[dim];
    const int64_t topk = indices->Dims()[dim];
    int64_t outer_size = 1;
    for (int64_t idx = 0; idx < dim; ++idx) { outer_size *= input_dims[idx]; }
    int64_t inner_size = 1;
    for (size_t idx = static_cast<size_t>(dim) + 1; idx < input_dims.size(); ++idx) { inner_size *= input_dims[idx]; }

    auto grad_input = std::make_shared<Tensor>(input_dims, grad_values->Dtype(), grad_values->GetDevice());
    std::memset(grad_input->DataPtr(), 0, grad_input->SizeInBytes());

    const size_t elem_size = kDataTypeToSize.at(grad_values->Dtype());
    const auto *src = static_cast<const std::byte *>(grad_values->DataPtr());
    auto *dst = static_cast<std::byte *>(grad_input->DataPtr());
    const auto *idx_ptr = static_cast<const int64_t *>(indices->DataPtr());
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            for (int64_t selected = 0; selected < topk; ++selected) {
                const int64_t out_offset = outer * topk * inner_size + selected * inner_size + inner;
                const int64_t selected_idx = idx_ptr[out_offset];
                CHECK_GE(selected_idx, 0);
                CHECK_LT(selected_idx, dim_size);
                std::memcpy(dst + (outer * dim_size * inner_size + selected_idx * inner_size + inner) * elem_size,
                            src + out_offset * elem_size, elem_size);
            }
        }
    }

    return grad_input;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_TOPK_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_TOPK_KERNEL(TopKForward)
REGISTER_CPU_TOPK_KERNEL(TopKBackward)

#undef REGISTER_CPU_TOPK_KERNEL

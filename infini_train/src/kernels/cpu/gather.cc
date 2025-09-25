#include <cmath>
#include <functional>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
// FIXME(zbl): This kernel aligns with torch.gather
//             Currently named IndexGather to avoid conflict with communication operators
//             Should be renamed to Gather later for interface consistency
std::shared_ptr<Tensor> IndexGatherForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &index,
                                           int64_t dim) {
    const auto &in_dims = input->Dims();
    const auto &idx_dims = index->Dims();
    CHECK_EQ(in_dims.size(), idx_dims.size());
    int64_t num_dims = in_dims.size();
    if (dim < 0) {
        dim += num_dims;
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, num_dims);

    // NOTE(zbl): Assume index to be int64 Tensors
    CHECK(index->Dtype() == DataType::kINT64);

    for (int d = 0; d < num_dims; ++d) {
        if (d == dim) {
            continue;
        }
        // Align with PyTorch semantics: index.size(d) <= input.size(d) for d != dim
        CHECK_LE(idx_dims[d], in_dims[d])
            << "index.size(" << d << ") must be <= input.size(" << d << ") on non-gather dims";
    }

    auto out = std::make_shared<Tensor>(idx_dims, input->Dtype(), input->GetDevice());

    std::vector<int64_t> in_strides(in_dims.size());
    int64_t s = 1;
    for (int i = (int)in_dims.size() - 1; i >= 0; --i) {
        in_strides[i] = s;
        s *= in_dims[i];
    }
    std::vector<int64_t> out_strides(idx_dims.size());
    s = 1;
    for (int i = (int)idx_dims.size() - 1; i >= 0; --i) {
        out_strides[i] = s;
        s *= idx_dims[i];
    }

    const int64_t gather_dim_size = in_dims[dim];
    int64_t total = 1;
    for (auto v : idx_dims) { total *= v; }

    std::vector<int64_t> norm_index(total);
    {
        const int64_t *idx_ptr = static_cast<const int64_t *>(index->DataPtr());
        for (int64_t i = 0; i < total; ++i) {
            int64_t v = idx_ptr[i];
            // Normalize like PyTorch: allow negative, clamp to [0, dim_size_gather-1]
            v = (v < 0) ? (v + gather_dim_size) : v;
            if (v < 0) {
                v = 0;
            }
            if (v >= gather_dim_size) {
                v = gather_dim_size - 1;
            }
            norm_index[i] = v;
        }
    }

    std::vector<int64_t> dst_index(idx_dims.size(), 0);
    std::vector<int64_t> src_index(in_dims.size(), 0);

    const float *in_ptr = static_cast<const float *>(input->DataPtr());
    float *out_ptr = static_cast<float *>(out->DataPtr());

    std::function<void(int)> recurse = [&](int d) {
        if (d == (int)in_dims.size()) {
            int64_t dst_offset = 0;
            for (int i = (int)idx_dims.size() - 1; i >= 0; --i) { dst_offset += dst_index[i] * out_strides[i]; }

            int64_t gather_j = norm_index[dst_offset];
            for (int i = 0; i < (int)in_dims.size(); ++i) { src_index[i] = (i == dim) ? gather_j : dst_index[i]; }

            int64_t src_offset = 0;
            for (int i = (int)in_dims.size() - 1; i >= 0; --i) { src_offset += src_index[i] * in_strides[i]; }

            out_ptr[dst_offset] = in_ptr[src_offset];
            return;
        }

        int64_t limit = idx_dims[d];
        for (int64_t i = 0; i < limit; ++i) {
            dst_index[d] = i;
            recurse(d + 1);
        }
    };

    recurse(0);
    return out;
}

std::shared_ptr<Tensor> IndexGatherBackward(const std::shared_ptr<Tensor> &grad_output,
                                            const std::shared_ptr<Tensor> &index, int64_t dim,
                                            const std::vector<int64_t> &input_dims) {
    const auto &in_dims = input_dims;
    const auto &idx_dims = index->Dims();
    CHECK_EQ(in_dims.size(), idx_dims.size());
    int64_t num_dims = in_dims.size();
    if (dim < 0) {
        dim += num_dims;
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, num_dims);

    // NOTE(zbl): Assume index to be int64 Tensors
    CHECK(index->Dtype() == DataType::kINT64);

    for (int d = 0; d < num_dims; ++d) {
        if (d == dim) {
            continue;
        }
        // Align with PyTorch semantics: index.size(d) <= input.size(d) for d != dim
        CHECK_LE(idx_dims[d], in_dims[d])
            << "index.size(" << d << ") must be <= input.size(" << d << ") on non-gather dims";
    }

    auto grad_input = std::make_shared<Tensor>(in_dims, grad_output->Dtype(), grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);

    std::vector<int64_t> in_strides(in_dims.size());
    int64_t s = 1;
    for (int i = (int)in_dims.size() - 1; i >= 0; --i) {
        in_strides[i] = s;
        s *= in_dims[i];
    }
    std::vector<int64_t> out_strides(idx_dims.size());
    s = 1;
    for (int i = (int)idx_dims.size() - 1; i >= 0; --i) {
        out_strides[i] = s;
        s *= idx_dims[i];
    }

    const int64_t gather_dim_size = in_dims[dim];
    int64_t total = 1;
    for (auto v : idx_dims) { total *= v; }

    std::vector<int64_t> norm_index(total);
    {
        const int64_t *idx_ptr = static_cast<const int64_t *>(index->DataPtr());
        for (int64_t i = 0; i < total; ++i) {
            int64_t v = idx_ptr[i];
            // Normalize like PyTorch: allow negative, clamp to [0, dim_size_gather-1]
            v = (v < 0) ? (v + gather_dim_size) : v;
            if (v < 0) {
                v = 0;
            }
            if (v >= gather_dim_size) {
                v = gather_dim_size - 1;
            }
            norm_index[i] = v;
        }
    }

    std::vector<int64_t> dst_index(idx_dims.size(), 0);
    std::vector<int64_t> src_index(in_dims.size(), 0);

    const float *go_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *gi_ptr = static_cast<float *>(grad_input->DataPtr());

    std::function<void(int)> recurse = [&](int d) {
        if (d == (int)in_dims.size()) {
            int64_t dst_offset = 0;
            for (int i = (int)idx_dims.size() - 1; i >= 0; --i) { dst_offset += dst_index[i] * out_strides[i]; }
            int64_t gather_j = norm_index[dst_offset];
            for (int i = 0; i < (int)in_dims.size(); ++i) { src_index[i] = (i == dim) ? gather_j : dst_index[i]; }
            int64_t src_offset = 0;
            for (int i = (int)in_dims.size() - 1; i >= 0; --i) { src_offset += src_index[i] * in_strides[i]; }
            gi_ptr[src_offset] += go_ptr[dst_offset];
            return;
        }

        int64_t limit = idx_dims[d];
        for (int64_t i = 0; i < limit; ++i) {
            dst_index[d] = i;
            recurse(d + 1);
        }
    };

    recurse(0);
    return grad_input;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_GATHER_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_GATHER_KERNEL(IndexGatherForward)
REGISTER_CPU_GATHER_KERNEL(IndexGatherBackward)

#undef REGISTER_CPU_GATHER_KERNEL

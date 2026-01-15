#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> ConcatForward(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    CHECK(!inputs.empty());

    const auto &base_dims = inputs[0]->Dims();
    if (dim < 0) {
        dim += base_dims.size();
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(base_dims.size()));

    std::vector<int64_t> Ks;
    Ks.reserve(inputs.size());
    auto dtype = inputs[0]->Dtype();
    auto device = inputs[0]->GetDevice();

    for (const auto &t : inputs) {
        CHECK(t->Dtype() == dtype);
        CHECK_EQ(t->Dims().size(), base_dims.size());
        for (size_t ax = 0; ax < base_dims.size(); ++ax) {
            if (static_cast<int64_t>(ax) == dim) {
                continue;
            }
            CHECK_EQ(t->Dims()[ax], base_dims[ax]) << "All non-concat dims must match";
        }
        Ks.push_back(t->Dims()[dim]);
    }

    std::vector<int64_t> output_dims = base_dims;
    const int64_t K_total = std::accumulate(Ks.begin(), Ks.end(), int64_t{0});
    output_dims[dim] = K_total;

    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    const int64_t outer_size
        = std::accumulate(output_dims.begin(), output_dims.begin() + dim, 1LL, std::multiplies<int64_t>());
    const int64_t inner_size
        = std::accumulate(output_dims.begin() + dim + 1, output_dims.end(), 1LL, std::multiplies<int64_t>());
    const size_t elem_size = sizeof(float);

    float *dst_ptr_base = static_cast<float *>(output->DataPtr());
    for (int64_t n = 0; n < outer_size; ++n) {
        int64_t offset_k = 0;
        float *dst_block = dst_ptr_base + n * K_total * inner_size;

        for (size_t i = 0; i < inputs.size(); ++i) {
            const int64_t Ki = Ks[i];
            const float *src_ptr = static_cast<const float *>(inputs[i]->DataPtr()) + n * Ki * inner_size;
            float *dst_ptr = dst_block + offset_k * inner_size;
            std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(Ki) * inner_size * elem_size);
            offset_k += Ki;
        }
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> ConcatBackward(const std::shared_ptr<Tensor> &grad_output,
                                                    const std::vector<std::vector<int64_t>> &input_dims_list,
                                                    int64_t dim) {
    CHECK(!input_dims_list.empty());

    const auto &go_dims = grad_output->Dims();
    if (dim < 0) {
        dim += go_dims.size();
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(go_dims.size()));

    const size_t rank = input_dims_list[0].size();
    for (const auto &d : input_dims_list) { CHECK_EQ(d.size(), rank); }
    for (size_t ax = 0; ax < rank; ++ax) {
        if (static_cast<int64_t>(ax) == dim) {
            continue;
        }
        for (size_t i = 1; i < input_dims_list.size(); ++i) {
            CHECK_EQ(input_dims_list[i][ax], input_dims_list[0][ax]);
        }
    }

    std::vector<int64_t> Ks;
    Ks.reserve(input_dims_list.size());
    for (const auto &d : input_dims_list) { Ks.push_back(d[dim]); }

    const int64_t outer_size = std::accumulate(input_dims_list[0].begin(), input_dims_list[0].begin() + dim, 1LL,
                                               std::multiplies<int64_t>());
    const int64_t inner_size = std::accumulate(input_dims_list[0].begin() + dim + 1, input_dims_list[0].end(), 1LL,
                                               std::multiplies<int64_t>());
    const int64_t K_total = std::accumulate(Ks.begin(), Ks.end(), int64_t{0});
    const size_t elem_size = sizeof(float);

    auto dtype = grad_output->Dtype();
    CHECK(dtype == DataType::kFLOAT32) << "CPU ConcatBackward assumes float32";
    std::vector<std::shared_ptr<Tensor>> grads;
    grads.reserve(input_dims_list.size());
    for (const auto &d : input_dims_list) { grads.emplace_back(std::make_shared<Tensor>(d, DataType::kFLOAT32)); }

    const float *src_base = static_cast<const float *>(grad_output->DataPtr());

    for (int64_t n = 0; n < outer_size; ++n) {
        int64_t offset_k = 0;
        const float *src_block = src_base + n * K_total * inner_size;

        for (size_t i = 0; i < grads.size(); ++i) {
            const int64_t Ki = Ks[i];
            const float *src_ptr = src_block + offset_k * inner_size;
            float *dst_ptr = static_cast<float *>(grads[i]->DataPtr()) + n * Ki * inner_size;
            std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(Ki) * inner_size * elem_size);
            offset_k += Ki;
        }
    }

    return grads;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CONCAT_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CONCAT_KERNEL(ConcatForward)
REGISTER_CPU_CONCAT_KERNEL(ConcatBackward)

#undef REGISTER_CPU_CONCAT_KERNEL

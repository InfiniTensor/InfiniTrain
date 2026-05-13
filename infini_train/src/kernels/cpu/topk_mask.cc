#include <limits>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> TopKMaskForward(const std::shared_ptr<Tensor> &input, int64_t topk) {
    CHECK(input->Dtype() == DataType::kFLOAT32) << "CPU TopKMaskForward currently supports float32 only";
    CHECK_GE(input->Dims().size(), 1);

    const auto &dims = input->Dims();
    const int64_t num_experts = dims.back();
    CHECK_GT(num_experts, 0);
    CHECK_GT(topk, 0);
    CHECK_LE(topk, num_experts);
    const int64_t rows = input->NumElements() / num_experts;

    auto output = std::make_shared<Tensor>(dims, input->Dtype(), input->GetDevice());
    output->Fill(0.0f);

    const float *in = static_cast<const float *>(input->DataPtr());
    float *out = static_cast<float *>(output->DataPtr());
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t row_offset = row * num_experts;
        std::vector<bool> selected_experts(num_experts, false);
        float selected_sum = 0.0f;
        for (int64_t selected = 0; selected < topk; ++selected) {
            int64_t best_idx = -1;
            float best_value = -std::numeric_limits<float>::infinity();
            for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                if (selected_experts[expert_idx]) {
                    continue;
                }
                const float value = in[row_offset + expert_idx];
                if (value > best_value) {
                    best_value = value;
                    best_idx = expert_idx;
                }
            }
            CHECK_GE(best_idx, 0);
            selected_experts[best_idx] = true;
            out[row_offset + best_idx] = best_value;
            selected_sum += best_value;
        }
        if (topk > 1 && selected_sum != 0.0f) {
            for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                out[row_offset + expert_idx]
                    = out[row_offset + expert_idx] == 0.0f ? 0.0f : out[row_offset + expert_idx] / selected_sum;
            }
        }
    }

    return output;
}

std::shared_ptr<Tensor> TopKMaskBackward(const std::shared_ptr<Tensor> &grad_output,
                                         const std::shared_ptr<Tensor> &mask_values) {
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "CPU TopKMaskBackward currently supports float32 only";
    CHECK(mask_values->Dtype() == DataType::kFLOAT32);
    CHECK(grad_output->Dims() == mask_values->Dims());

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    grad_input->Fill(0.0f);

    const float *grad = static_cast<const float *>(grad_output->DataPtr());
    const float *mask = static_cast<const float *>(mask_values->DataPtr());
    float *out = static_cast<float *>(grad_input->DataPtr());
    for (int64_t i = 0; i < static_cast<int64_t>(grad_output->NumElements()); ++i) {
        out[i] = mask[i] != 0.0f ? grad[i] : 0.0f;
    }

    return grad_input;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_TOPK_MASK_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_TOPK_MASK_KERNEL(TopKMaskForward)
REGISTER_CPU_TOPK_MASK_KERNEL(TopKMaskBackward)

#undef REGISTER_CPU_TOPK_MASK_KERNEL

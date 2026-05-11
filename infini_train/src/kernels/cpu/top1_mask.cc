#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> Top1MaskForward(const std::shared_ptr<Tensor> &input) {
    CHECK(input->Dtype() == DataType::kFLOAT32) << "CPU Top1MaskForward currently supports float32 only";
    CHECK_GE(input->Dims().size(), 1);

    const auto &dims = input->Dims();
    const int64_t num_experts = dims.back();
    CHECK_GT(num_experts, 0);
    const int64_t rows = input->NumElements() / num_experts;

    auto output = std::make_shared<Tensor>(dims, input->Dtype(), input->GetDevice());
    output->Fill(0.0f);

    const float *in = static_cast<const float *>(input->DataPtr());
    float *out = static_cast<float *>(output->DataPtr());
    for (int64_t row = 0; row < rows; ++row) {
        int64_t best_idx = 0;
        float best_value = in[row * num_experts];
        for (int64_t expert_idx = 1; expert_idx < num_experts; ++expert_idx) {
            const float value = in[row * num_experts + expert_idx];
            if (value > best_value) {
                best_value = value;
                best_idx = expert_idx;
            }
        }
        out[row * num_experts + best_idx] = best_value;
    }

    return output;
}

std::shared_ptr<Tensor> Top1MaskBackward(const std::shared_ptr<Tensor> &grad_output,
                                         const std::shared_ptr<Tensor> &mask_values) {
    CHECK(grad_output->Dtype() == DataType::kFLOAT32) << "CPU Top1MaskBackward currently supports float32 only";
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

#define REGISTER_CPU_TOP1_MASK_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_TOP1_MASK_KERNEL(Top1MaskForward)
REGISTER_CPU_TOP1_MASK_KERNEL(Top1MaskBackward)

#undef REGISTER_CPU_TOP1_MASK_KERNEL

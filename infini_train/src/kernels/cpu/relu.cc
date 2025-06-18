#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> ReluForward(const std::shared_ptr<Tensor> &input) {
    // Save input dimensions
    const auto &input_dims = input->Dims();
    const size_t num_elements = input->NumElements();

    // Create output tensor and mask tensor (same dimensions as input)
    auto output = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto mask = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);

    // Get data pointers (assumed to be contiguous float arrays)
    const float *input_data = static_cast<const float *>(input->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());
    float *mask_data = static_cast<float *>(mask->DataPtr());

    // Process each element
    for (size_t i = 0; i < num_elements; ++i) {
        const float val = input_data[i];
        if (val > 0.0f) {
            output_data[i] = val; // Keep positive values
            mask_data[i] = 1.0f;  // Mark as active position
        } else {
            output_data[i] = 0.0f; // Set non-positive values to zero
            mask_data[i] = 0.0f;   // Mark as inactive position
        }
    }

    return {output, mask};
}

std::shared_ptr<Tensor> ReluBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask) {
    // Validate input dimensions
    const auto &go_dims = grad_output->Dims();
    const auto &mask_dims = mask->Dims();
    CHECK_EQ(go_dims.size(), mask_dims.size());

    const size_t num_elements = grad_output->NumElements();

    // Create gradient tensor for input (same dimensions as input)
    auto grad_input = std::make_shared<Tensor>(go_dims, DataType::kFLOAT32);
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());
    const float *go_data = static_cast<const float *>(grad_output->DataPtr());
    const float *mask_data = static_cast<const float *>(mask->DataPtr());

    // Process each element, propagate gradient based on mask
    for (size_t i = 0; i < num_elements; ++i) {
        // Gradient is only propagated where the input was positive during forward pass (mask=1)
        grad_input_data[i] = go_data[i] * mask_data[i];
    }

    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_RELU_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_RELU_KERNEL(ReluForward)
REGISTER_CPU_RELU_KERNEL(ReluBackward)

#undef REGISTER_CPU_RELU_KERNEL
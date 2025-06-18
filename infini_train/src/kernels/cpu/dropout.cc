#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
DropoutForward(const std::shared_ptr<Tensor> input, float p, bool training, bool inplace = false) {
    CHECK(p >= 0.0f && p < 1.0f) << "Dropout probability must be in [0, 1)";
    CHECK(input != nullptr) << "Input tensor cannot be null";

    const auto &input_dims = input->Dims();
    const int64_t numel = input->NumElements();

    if (!training || p == 0.0f) {
        auto mask = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
        mask->Fill(1.0f);

        if (inplace) {
            return {input, mask};
        } else {
            auto output = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
            std::memcpy(output->DataPtr(), input->DataPtr(), numel * sizeof(float));
            return {output, mask};
        }
    }

    const float scale = 1.0f / (1.0f - p);

    std::shared_ptr<Tensor> output;
    auto mask = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);

    if (inplace) {
        output = input;
    } else {
        output = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    }

    const float *input_data = static_cast<const float *>(input->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());
    float *mask_data = static_cast<float *>(mask->DataPtr());

    static thread_local std::mt19937 generator(std::random_device{}());
    std::bernoulli_distribution distribution(1.0f - p);

    for (int64_t i = 0; i < numel; ++i) {
        bool keep = distribution(generator);
        mask_data[i] = keep ? scale : 0.0f;
        output_data[i] = input_data[i] * mask_data[i];
    }

    return {output, mask};
}

std::shared_ptr<Tensor> DropoutBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask,
                                        float p, bool training, bool inplace) {
    CHECK(grad_output != nullptr) << "Gradient output tensor cannot be null";
    CHECK(mask != nullptr) << "Mask tensor cannot be null";
    CHECK(p >= 0.0f && p < 1.0f) << "Dropout probability must be in [0, 1)";

    const auto &grad_dims = grad_output->Dims();
    const int64_t numel = grad_output->NumElements();
    const auto &mask_dims = mask->Dims();
    CHECK(grad_dims == mask_dims) << "Gradient and mask dimension mismatch";

    if (!training || p == 0.0f) {
        if (inplace) {
            return grad_output;
        } else {
            auto grad_input = std::make_shared<Tensor>(grad_dims, DataType::kFLOAT32);
            std::memcpy(grad_input->DataPtr(), grad_output->DataPtr(), numel * sizeof(float));
            return grad_input;
        }
    }

    const float scale = 1.0f / (1.0f - p);

    std::shared_ptr<Tensor> grad_input;
    if (inplace) {
        grad_input = grad_output;
    } else {
        grad_input = std::make_shared<Tensor>(grad_dims, DataType::kFLOAT32);
    }

    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    const float *mask_data = static_cast<const float *>(mask->DataPtr());
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());

    for (int64_t i = 0; i < numel; ++i) { grad_input_data[i] = grad_output_data[i] * mask_data[i]; }

    return grad_input;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_DROPOUT_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_DROPOUT_KERNEL(DropoutForward)
REGISTER_CPU_DROPOUT_KERNEL(DropoutBackward)

#undef REGISTER_CPU_DROPOUT_KERNEL

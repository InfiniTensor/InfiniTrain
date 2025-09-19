#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> MSEForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &target) {
    CHECK_EQ(input->NumElements(), target->NumElements());
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(target->Dtype())) << "Input/target dtype must match";
    // TODO(zbl): support multi datatypes later
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "CPU MSE currently supports float32 only";

    const int64_t n = input->NumElements();

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *target_ptr = static_cast<const float *>(target->DataPtr());

    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float d = input_ptr[i] - target_ptr[i];
        sum_sq += d * d;
    }

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    static_cast<float *>(output->DataPtr())[0] = sum_sq / static_cast<float>(n);
    return {output};
}

std::shared_ptr<Tensor> MSEBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &target,
                                    const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->NumElements(), target->NumElements());
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(target->Dtype())) << "Input/target dtype must match";
    // TODO(zbl): support multi datatypes later
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "CPU MSE currently supports float32 only";

    CHECK_EQ(grad_output->Dims().size(), 0) << "grad_output must be a scalar";
    // TODO(zbl): support grad dtype conversion later
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(DataType::kFLOAT32))
        << "grad_output must be float32 for now";

    const int64_t n = input->NumElements();

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *target_ptr = static_cast<const float *>(target->DataPtr());
    const float grad_output_ptr = static_cast<const float *>(grad_output->DataPtr())[0];
    const float scale = (2.0f / static_cast<float>(n)) * grad_output_ptr;

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());

    for (int64_t i = 0; i < n; ++i) { grad_input_ptr[i] = scale * (input_ptr[i] - target_ptr[i]); }

    return {grad_input};
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_MSE_KERNEL(kernel_name)                                                                           \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_MSE_KERNEL(MSEForward)
REGISTER_CPU_MSE_KERNEL(MSEBackward)

#undef REGISTER_CPU_MSE_KERNEL

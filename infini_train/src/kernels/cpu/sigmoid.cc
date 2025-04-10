#include "infini_train/include/kernels/cpu/sigmoid.h"

#include <cmath>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> SigmoidForward(const std::shared_ptr<Tensor> &input) {
    CHECK_EQ(input->Dims().size(), 2);
    const int bs = input->Dims()[0];
    const int out_dim = input->Dims()[1];
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < out_dim; ++j) {
            const auto idx = i * out_dim + j;
            static_cast<float *>(output->DataPtr())[idx]
                = 1.0f / (1.0f + exp(-static_cast<const float *>(input->DataPtr())[idx]));
        }
    }
    return output;
}

std::shared_ptr<Tensor> SigmoidBackward(const std::shared_ptr<Tensor> &output,
                                        const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(output->Dims().size(), 2);
    const int bs = output->Dims()[0];
    const int out_dim = output->Dims()[1];
    CHECK_EQ(grad_output->Dims().size(), 2);
    CHECK_EQ(bs, grad_output->Dims()[0]);
    CHECK_EQ(out_dim, grad_output->Dims()[1]);
    auto grad_input = std::make_shared<Tensor>(output->Dims(), DataType::kFLOAT32);
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < out_dim; ++j) {
            const auto idx = i * out_dim + j;
            const float x = static_cast<const float *>(output->DataPtr())[idx];
            static_cast<float *>(grad_input->DataPtr())[idx]
                = static_cast<const float *>(grad_output->DataPtr())[idx] * x * (1.0f - x);
        }
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu

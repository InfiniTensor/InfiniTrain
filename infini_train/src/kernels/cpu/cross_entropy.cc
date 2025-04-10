#include "infini_train/include/kernels/cpu/cross_entropy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
}

// TODO(dcj): support target which is not index value
std::shared_ptr<Tensor> CrossEntropyForward(const std::shared_ptr<Tensor> &input,
                                            const std::shared_ptr<Tensor> &target) {
    CHECK_EQ(input->Dims().size(), 2);
    const int bs = input->Dims()[0];
    const int num_classes = input->Dims()[1];
    CHECK_EQ(target->Dims().size(), 2);
    CHECK_EQ(bs, target->Dims()[0]);
    CHECK_EQ(1, target->Dims()[1]);
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);
    static_cast<float *>(output->DataPtr())[0] = 0.0f;
    for (int64_t i = 0; i < bs; ++i) {
        float max_logit = kNegativeInfinity;
        for (int64_t j = 0; j < num_classes; ++j) {
            max_logit = std::max(max_logit, reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j]);
        }
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < num_classes; ++j) {
            sum_exp += exp(reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j] - max_logit);
        }
        static_cast<float *>(output->DataPtr())[0]
            -= log(exp(reinterpret_cast<const float *>(
                           input->DataPtr())[i * num_classes + reinterpret_cast<const uint8_t *>(target->DataPtr())[i]]
                       - max_logit)
                   / sum_exp);
    }
    static_cast<float *>(output->DataPtr())[0] /= bs;
    return {output};
}

std::shared_ptr<Tensor> CrossEntropyBackward(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &target,
                                             const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 2);
    const int bs = input->Dims()[0];
    const int num_classes = input->Dims()[1];
    CHECK_EQ(target->Dims().size(), 2);
    CHECK_EQ(bs, target->Dims()[0]);
    CHECK_EQ(1, target->Dims()[1]);
    CHECK_EQ(grad_output->Dims().size(), 0);
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    std::vector<float> softmax(bs * num_classes, 0.0f);
    for (int64_t i = 0; i < bs; ++i) {
        float max_logit = kNegativeInfinity;
        for (int64_t j = 0; j < num_classes; ++j) {
            max_logit = std::max(max_logit, reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j]);
        }
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < num_classes; ++j) {
            sum_exp += exp(reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j] - max_logit);
        }
        for (int64_t j = 0; j < num_classes; ++j) {
            const auto idx = i * num_classes + j;
            softmax[idx] = exp(reinterpret_cast<const float *>(input->DataPtr())[idx] - max_logit) / sum_exp;
        }
    }
    for (int64_t i = 0; i < bs; ++i) {
        const auto target_idx = reinterpret_cast<const uint8_t *>(target->DataPtr())[i];
        for (int64_t j = 0; j < num_classes; ++j) {
            const auto idx = i * num_classes + j;
            static_cast<float *>(grad_input->DataPtr())[idx]
                = reinterpret_cast<const float *>(grad_output->DataPtr())[0]
                * (softmax[idx] - (j == target_idx ? 1.0f : 0.0f)) / bs;
        }
    }
    return {grad_input};
}
} // namespace infini_train::kernels::cpu

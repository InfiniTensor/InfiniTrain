#include "infini_train/include/kernels/cpu/linear.h"

#include <memory>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      const std::shared_ptr<Tensor> &bias) {
    CHECK_EQ(input->Dims().size(), 2);
    const int bs = input->Dims()[0];
    const int in_feature = input->Dims()[1];
    CHECK_EQ(weight->Dims().size(), 2);
    CHECK_EQ(in_feature, weight->Dims()[0]);
    const int out_feature = weight->Dims()[1];
    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_feature);
    }
    auto output = std::make_shared<Tensor>(std::vector<int64_t>{bs, out_feature}, DataType::kFLOAT32);
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < out_feature; ++j) {
            auto *data_ptr = static_cast<float *>(output->DataPtr()) + i * out_feature + j;
            *data_ptr = 0.0f;
            for (int64_t k = 0; k < in_feature; ++k) {
                *data_ptr += reinterpret_cast<const float *>(input->DataPtr())[i * in_feature + k]
                           * reinterpret_cast<const float *>(weight->DataPtr())[k * out_feature + j];
            }
            if (bias) {
                *data_ptr += reinterpret_cast<const float *>(bias->DataPtr())[j];
            }
        }
    }
    return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
               const std::shared_ptr<Tensor> &bias, const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 2);
    const int bs = input->Dims()[0];
    const int in_feature = input->Dims()[1];
    CHECK_EQ(weight->Dims().size(), 2);
    CHECK_EQ(in_feature, weight->Dims()[0]);
    const int out_feature = weight->Dims()[1];
    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_feature);
    }
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32);
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(bias->Dims(), DataType::kFLOAT32);
        grad_bias->Fill<float>(0.0f);
    }
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < in_feature; ++j) {
            const auto input_idx = i * in_feature + j;
            auto *data_ptr = static_cast<float *>(grad_input->DataPtr()) + input_idx;
            *data_ptr = 0.0f;
            for (int64_t k = 0; k < out_feature; ++k) {
                const auto weight_idx = j * out_feature + k;
                const auto grad = reinterpret_cast<const float *>(grad_output->DataPtr())[i * out_feature + k];
                *data_ptr += grad * reinterpret_cast<const float *>(weight->DataPtr())[weight_idx];
                static_cast<float *>(grad_weight->DataPtr())[weight_idx]
                    += grad * reinterpret_cast<const float *>(input->DataPtr())[input_idx];
            }
        }
        if (bias) {
            for (int64_t k = 0; k < out_feature; ++k) {
                static_cast<float *>(grad_bias->DataPtr())[k]
                    += reinterpret_cast<const float *>(grad_output->DataPtr())[i * out_feature + k];
            }
        }
    }
    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

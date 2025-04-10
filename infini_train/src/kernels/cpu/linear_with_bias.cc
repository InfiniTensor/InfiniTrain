#include "infini_train/include/kernels/cpu/linear_with_bias.h"

#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> LinearWithBiasForward(const std::shared_ptr<Tensor> &input,
                                              const std::shared_ptr<Tensor> &weight,
                                              const std::shared_ptr<Tensor> &bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin(), input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[0]);
    const int out_features = weight_dims[1];

    const auto &bias_dims = bias->Dims();
    CHECK_EQ(bias_dims.size(), 1);
    CHECK_EQ(bias_dims[0], out_features);

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < out_features; ++j) {
            auto *data_ptr = static_cast<float *>(output->DataPtr()) + i * out_features + j;
            *data_ptr = 0.0f;
            for (int64_t k = 0; k < in_features; ++k) {
                *data_ptr += reinterpret_cast<const float *>(input->DataPtr())[i * in_features + k]
                           * reinterpret_cast<const float *>(weight->DataPtr())[k * out_features + j];
            }
            *data_ptr += reinterpret_cast<const float *>(bias->DataPtr())[j];
        }
    }
    return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearWithBiasBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                       int64_t out_features, const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin(), input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[0]);
    CHECK_EQ(out_features, weight_dims[1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    grad_weight->Fill<float>(0.0f);
    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    grad_bias->Fill<float>(0.0f);

    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < in_features; ++j) {
            const auto input_idx = i * in_features + j;
            auto *data_ptr = static_cast<float *>(grad_input->DataPtr()) + input_idx;
            *data_ptr = 0.0f;
            for (int64_t k = 0; k < out_features; ++k) {
                const auto weight_idx = j * out_features + k;
                const auto grad = reinterpret_cast<const float *>(grad_output->DataPtr())[i * out_features + k];
                *data_ptr += grad * reinterpret_cast<const float *>(weight->DataPtr())[weight_idx];
                static_cast<float *>(grad_weight->DataPtr())[weight_idx]
                    += grad * reinterpret_cast<const float *>(input->DataPtr())[input_idx];
            }
        }
        for (int64_t k = 0; k < out_features; ++k) {
            static_cast<float *>(grad_bias->DataPtr())[k]
                += reinterpret_cast<const float *>(grad_output->DataPtr())[i * out_features + k];
        }
    }
    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

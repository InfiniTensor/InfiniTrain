#include "infini_train/include/kernels/cpu/linear.h"

#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> LinearWithoutBiasForward(const std::shared_ptr<Tensor> &input1,
                                                 const std::shared_ptr<Tensor> &input2) {
    const auto &input1_dims = input1->Dims();
    CHECK_GE(input1_dims.size(), 2);
    const int64_t bs = std::accumulate(input1_dims.rbegin(), input1_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input1_dims.rbegin();

    const auto &input2_dims = input2->Dims();
    CHECK_EQ(input2_dims.size(), 2);
    CHECK_EQ(in_features, input2_dims[0]);
    const int out_features = input2_dims[1];

    auto output_dims = input1_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < out_features; ++j) {
            auto *data_ptr = static_cast<float *>(output->DataPtr()) + i * out_features + j;
            *data_ptr = 0.0f;
            for (int64_t k = 0; k < in_features; ++k) {
                *data_ptr += reinterpret_cast<const float *>(input1->DataPtr())[i * in_features + k]
                           * reinterpret_cast<const float *>(input2->DataPtr())[k * out_features + j];
            }
        }
    }
    return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearWithoutBiasBackward(const std::shared_ptr<Tensor> &input1, const std::shared_ptr<Tensor> &input2,
                          int64_t out_features, const std::shared_ptr<Tensor> &grad_output) {
    const auto &input1_dims = input1->Dims();
    CHECK_GE(input1_dims.size(), 2);
    const int64_t bs = std::accumulate(input1_dims.rbegin(), input1_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input1_dims.rbegin();

    const auto &input2_dims = input2->Dims();
    CHECK_EQ(input2_dims.size(), 2);
    CHECK_EQ(in_features, input2_dims[0]);
    CHECK_EQ(out_features, input2_dims[1]);

    auto grad_input1 = std::make_shared<Tensor>(input1_dims, DataType::kFLOAT32);
    auto grad_input2 = std::make_shared<Tensor>(input2_dims, DataType::kFLOAT32);
    grad_input2->Fill<float>(0.0f);

    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < in_features; ++j) {
            const auto input1_idx = i * in_features + j;
            auto *data_ptr = static_cast<float *>(grad_input1->DataPtr()) + input1_idx;
            *data_ptr = 0.0f;
            for (int64_t k = 0; k < out_features; ++k) {
                const auto input2_idx = j * out_features + k;
                const auto grad = reinterpret_cast<const float *>(grad_output->DataPtr())[i * out_features + k];
                *data_ptr += grad * reinterpret_cast<const float *>(input2->DataPtr())[input2_idx];
                static_cast<float *>(grad_input2->DataPtr())[input2_idx]
                    += grad * reinterpret_cast<const float *>(input1->DataPtr())[input1_idx];
            }
        }
    }
    return {grad_input1, grad_input2};
}
} // namespace infini_train::kernels::cpu

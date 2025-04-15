#include "infini_train/include/kernels/cpu/linear.h"

#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    /*
    output[*, m, n] = input[*, m, k] * other[*, k, n]
    */
    // TODO(dcj): support broadcast later
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);
    const int64_t n = other_dims[other_dims.size() - 1];

    const int64_t bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});
    for (int64_t i = 0; i < input_dims.size() - 2; ++i) {
        CHECK_EQ(input_dims[i], other_dims[i]) << "Batch dims must match";
    }

    std::vector<int64_t> output_dims = input_dims;
    output_dims[output_dims.size() - 1] = n;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    for (int64_t b = 0; b < bs; ++b) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    acc += reinterpret_cast<const float *>(input->DataPtr())[b * m * k + i * k + p]
                         * reinterpret_cast<const float *>(other->DataPtr())[b * k * n + p * n + j];
                }
                static_cast<float *>(output->DataPtr())[b * m * n + i * n + j] = acc;
            }
        }
    }
    return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    /*
    grad_input[*, m, k] = grad_output[*, m, n] * other[*, k, n]^T
    grad_other[*, k, n] = input[*, m, k]^T * grad_output[*, m, n]
    */
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_output_dims = grad_output->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());
    CHECK_EQ(input_dims.size(), grad_output_dims.size());

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);
    const int64_t n = other_dims[other_dims.size() - 1];
    CHECK_EQ(m, grad_output_dims[grad_output_dims.size() - 2]);
    CHECK_EQ(n, grad_output_dims[grad_output_dims.size() - 1]);

    const int64_t bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});
    for (int64_t i = 0; i < input_dims.size() - 2; ++i) {
        CHECK_EQ(input_dims[i], other_dims[i]) << "Batch dims must match";
        CHECK_EQ(input_dims[i], grad_output_dims[i]) << "Batch dims must match";
    }

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    for (int64_t b = 0; b < bs; ++b) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                const float grad = static_cast<float *>(grad_output->DataPtr())[b * m * n + i * n + j];
                for (int64_t p = 0; p < k; ++p) {
                    const auto input_idx = b * m * k + i * k + p;
                    const auto other_idx = b * k * n + p * n + j;
                    static_cast<float *>(grad_input->DataPtr())[input_idx]
                        += grad * reinterpret_cast<const float *>(other->DataPtr())[other_idx];
                    static_cast<float *>(grad_other->DataPtr())[other_idx]
                        += grad * reinterpret_cast<const float *>(input->DataPtr())[input_idx];
                }
            }
        }
    }
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

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
                           * reinterpret_cast<const float *>(
                                 weight->DataPtr())[transpose ? j * in_features + k : k * out_features + j];
            }
            *data_ptr += reinterpret_cast<const float *>(bias->DataPtr())[j];
        }
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

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
                const auto weight_idx = transpose ? k * in_features + j : j * out_features + k;
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

#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::shared_ptr<Tensor> LinearBackwardInput(const std::shared_ptr<Tensor> &weight,
                                            const std::shared_ptr<Tensor> &grad_output, bool transpose,
                                            int64_t in_features, int64_t out_features,
                                            const std::vector<int64_t> &input_dims) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    */
    CHECK_GE(input_dims.size(), 2);
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
    }
    return grad_input;
}

std::shared_ptr<Tensor> LinearBackwardWeight(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output, bool transpose,
                                             int64_t in_features, int64_t out_features) {
    /*
    transpose:
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]

    !transpose:
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    */
    std::vector<int64_t> weight_dims
        = transpose ? std::vector<int64_t>{out_features, in_features} : std::vector<int64_t>{in_features, out_features};
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    if (transpose) {
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    return grad_weight;
}

std::shared_ptr<Tensor> LinearBackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_features) {
    /*
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */
    auto grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    return grad_bias;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackwardInput)
REGISTER_CPU_LINEAR_KERNEL(LinearBackwardWeight)
REGISTER_CPU_LINEAR_KERNEL(LinearBackwardBias)

#undef REGISTER_CPU_LINEAR_KERNEL

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
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
    for (int64_t i = 0; i < static_cast<int64_t>(input_dims.size()) - 2; ++i) {
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
                    acc += static_cast<const float *>(input->DataPtr())[b * m * k + i * k + p]
                         * static_cast<const float *>(other->DataPtr())[b * k * n + p * n + j];
                }
                static_cast<float *>(output->DataPtr())[b * m * n + i * n + j] = acc;
            }
        }
    }
    return {output};
}

std::shared_ptr<Tensor> MatmulBackwardInput(const std::shared_ptr<Tensor> &other,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims) {
    /*
    grad_input[*, m, k] = grad_output[*, m, n] * other[*, k, n]^T
    */
    const auto &other_dims = other->Dims();
    const auto &grad_output_dims = grad_output->Dims();

    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(other_dims.size(), grad_output_dims.size());

    const int64_t m = grad_output_dims[grad_output_dims.size() - 2];
    const int64_t k = other_dims[other_dims.size() - 2];
    const int64_t n = grad_output_dims[grad_output_dims.size() - 1];

    const int64_t bs
        = std::accumulate(grad_output_dims.rbegin() + 2, grad_output_dims.rend(), 1, std::multiplies<int64_t>{});
    for (int64_t i = 0; i < static_cast<int64_t>(grad_output_dims.size()) - 2; ++i) {
        CHECK_EQ(grad_output_dims[i], other_dims[i]) << "Batch dims must match";
    }

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    grad_input->Fill<float>(0.0f);

    for (int64_t b = 0; b < bs; ++b) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                const float grad = static_cast<float *>(grad_output->DataPtr())[b * m * n + i * n + j];
                for (int64_t p = 0; p < k; ++p) {
                    const auto other_idx = b * k * n + p * n + j;
                    static_cast<float *>(grad_input->DataPtr())[b * m * k + i * k + p]
                        += grad * static_cast<const float *>(other->DataPtr())[other_idx];
                }
            }
        }
    }
    return grad_input;
}

std::shared_ptr<Tensor> MatmulBackwardOther(const std::shared_ptr<Tensor> &input1,
                                            const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &other_dims) {
    /*
    grad_other[*, k, n] = input[*, m, k]^T * grad_output[*, m, n]
    */
    const auto &input_dims = input1->Dims();
    const auto &grad_output_dims = grad_output->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), grad_output_dims.size());

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims[input_dims.size() - 1];
    const int64_t n = grad_output_dims[grad_output_dims.size() - 1];
    CHECK_EQ(m, grad_output_dims[grad_output_dims.size() - 2]);
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);

    const int64_t bs = std::accumulate(input_dims.rbegin() + 2, input_dims.rend(), 1, std::multiplies<int64_t>{});
    for (int64_t i = 0; i < static_cast<int64_t>(input_dims.size()) - 2; ++i) {
        CHECK_EQ(input_dims[i], grad_output_dims[i]) << "Batch dims must match";
        CHECK_EQ(input_dims[i], other_dims[i]) << "Batch dims must match";
    }

    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);
    grad_other->Fill<float>(0.0f);

    for (int64_t b = 0; b < bs; ++b) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                const float grad = static_cast<float *>(grad_output->DataPtr())[b * m * n + i * n + j];
                for (int64_t p = 0; p < k; ++p) {
                    const auto input_idx = b * m * k + i * k + p;
                    static_cast<float *>(grad_other->DataPtr())[b * k * n + p * n + j]
                        += grad * static_cast<const float *>(input1->DataPtr())[input_idx];
                }
            }
        }
    }
    return grad_other;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_MATMUL_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_MATMUL_KERNEL(MatmulForward)
REGISTER_CPU_MATMUL_KERNEL(MatmulBackwardInput)
REGISTER_CPU_MATMUL_KERNEL(MatmulBackwardOther)

#undef REGISTER_CPU_MATMUL_KERNEL

#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> OuterForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    /*
    output[i, j] = input[i] * other[j]
    output shape: [input.size(0), other.size(0)]
    */

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    // TODO(zbl): support batched outer?
    CHECK_EQ(input_dims.size(), 1);
    CHECK_EQ(other_dims.size(), 1);

    std::vector<int64_t> out_shape = {input_dims[0], other_dims[0]};
    auto output = std::make_shared<Tensor>(out_shape, DataType::kFLOAT32);

    // [M, N] = [M, 1] * [1, N]
    output->EigenMatrix() = input->EigenVector().transpose() * other->EigenVector();

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> OuterBackward(const std::shared_ptr<Tensor> &input,
                                                                           const std::shared_ptr<Tensor> &other,
                                                                           const std::shared_ptr<Tensor> &grad_output) {
    /*
    grad_input[i] = sum_j(grad_output[i, j] * other[j])
    grad_other[j] = sum_i(grad_output[i, j] * input[i])
    */

    const int64_t m = input->Dims()[0];
    const int64_t n = other->Dims()[0];
    // TODO(zbl): support batched outer?
    CHECK_EQ(grad_output->Dims().size(), 2);
    CHECK_EQ(grad_output->Dims()[0], m);
    CHECK_EQ(grad_output->Dims()[1], n);

    auto grad_input = std::make_shared<Tensor>(std::vector<int64_t>{m}, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(std::vector<int64_t>{n}, DataType::kFLOAT32);
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    grad_input->EigenVector() = grad_output->EigenMatrix() * other->EigenVector().transpose();
    grad_other->EigenVector() = grad_output->EigenMatrix().transpose() * input->EigenVector().transpose();

    return {grad_input, grad_other};
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_OUTER_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_OUTER_KERNEL(OuterForward)
REGISTER_CPU_OUTER_KERNEL(OuterBackward)

#undef REGISTER_CPU_OUTER_KERNEL

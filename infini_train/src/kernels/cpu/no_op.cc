#include "infini_train/include/kernels/cpu/no_op.h"

#include <memory>
#include <numeric>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> NoOpForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &dims) {
    const int64_t num_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    CHECK_EQ(input->NumElements(), num_elements);

    auto output = std::make_shared<Tensor>(*input, 0, dims);
    return output;
}

std::shared_ptr<Tensor> NoOpBackward(const std::vector<int64_t> &dims, const std::shared_ptr<Tensor> &grad_output) {
    auto num_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    CHECK_EQ(num_elements, grad_output->NumElements());

    auto grad_input = std::make_shared<Tensor>(*grad_output, 0, dims);
    return grad_input;
}
} // namespace infini_train::kernels::cpu

#include "infini_train/include/nn/modules/flatten.h"

#include "infini_train/include/tensor.h"

namespace infini_train::nn {
std::vector<std::shared_ptr<Tensor>> Flatten::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK(inputs[0]);
    const int64_t rank = inputs[0]->Dims().size();
    const int64_t start = start_dim_ < 0 ? start_dim_ + rank : start_dim_;
    const int64_t end = end_dim_ < 0 ? end_dim_ + rank : end_dim_;
    CHECK_GE(start, 0);
    CHECK_GE(end, start);
    CHECK_LT(end, rank);
    return {inputs[0]->Flatten(start, end)};
}
} // namespace infini_train::nn

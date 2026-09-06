#include "infini_train/include/nn/modules/flatten.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::nn {
std::vector<std::shared_ptr<Tensor>> Flatten::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    return {input_tensors[0]->Flatten(start_dim_, end_dim_)};
}
} // namespace infini_train::nn

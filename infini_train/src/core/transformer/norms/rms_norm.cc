#include "infini_train/include/core/transformer/norms/rms_norm.h"

#include <cmath>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_builders.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
RMSNorm::RMSNorm(int64_t dim, float eps, infini_train::Device device) : CloneableModule(kType), eps_(eps) {
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{dim}, DataType::kFLOAT32, device)->RequiresGrad();
    nn::init::Ones(parameters_[kParamWeightName]);
}

std::vector<std::shared_ptr<Tensor>> RMSNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // broadcasted Mul([4, 64, 2048] * [4, 64, 1])
    auto norm = x[0] * nn::function::Rsqrt(nn::function::Mean(nn::function::Pow(x[0], 2), -1, true) + eps_);
    return {norm * parameters_[kParamWeightName]};
}
} // namespace infini_train::nn
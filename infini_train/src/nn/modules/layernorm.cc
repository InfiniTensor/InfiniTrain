#include "infini_train/include/nn/modules/layernorm.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/layernorm.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
constexpr char kParamWeightName[] = "weight";
constexpr char kParamBiasName[] = "bias";
} // namespace

LayerNorm::LayerNorm(int64_t embed_dim, float eps, Device device) : eps_(eps) {
    device_ = device;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{embed_dim}, DataType::kFLOAT32, device)->RequiresGrad();
    parameters_[kParamBiasName]
        = std::make_shared<Tensor>(std::vector<int64_t>{embed_dim}, DataType::kFLOAT32, device)->RequiresGrad();

    auto *w = parameters_[kParamWeightName].get();
    w->Fill<float>(0.5);
    auto *b = parameters_[kParamBiasName].get();
    b->Fill<float>(0.5);
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::LayerNorm>(eps_)->Apply(
        {input_tensors[0], parameters_[kParamWeightName], parameters_[kParamBiasName]});
}
} // namespace infini_train::nn

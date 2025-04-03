#include "infini_train/include/nn/modules/linear.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
constexpr char kParamWeightName[] = "weight";
constexpr char kParamBiasName[] = "bias";
} // namespace

Linear::Linear(int64_t in_dim, int64_t out_dim, Device device) {
    device_ = device;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{in_dim, out_dim}, DataType::kFLOAT32, device)->RequiresGrad();
    parameters_[kParamBiasName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_dim}, DataType::kFLOAT32, device)->RequiresGrad();

    auto *w = parameters_[kParamWeightName].get();
    w->Fill<float>(0.5);
    auto *b = parameters_[kParamBiasName].get();
    b->Fill<float>(0.5);
}

std::vector<std::shared_ptr<Tensor>> Linear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::Linear>()->Apply(
        {input_tensors[0], parameters_[kParamWeightName], parameters_[kParamBiasName]});
}
} // namespace infini_train::nn

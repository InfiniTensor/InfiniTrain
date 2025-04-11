#include "infini_train/include/nn/modules/embedding.h"

#include <memory>
#include <random>
#include <vector>

#include "infini_train/include/autograd/embedding.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
constexpr char kParamWeightName[] = "weight";
} // namespace

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, Device device) {
    device_ = device;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{num_embeddings, embedding_dim}, DataType::kFLOAT32, device)
              ->RequiresGrad();

    auto *weight = parameters_[kParamWeightName].get();
    weight->Fill<float>(0.5);
}

std::vector<std::shared_ptr<Tensor>> Embedding::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::Embedding>()->Apply({input_tensors[0], parameters_[kParamWeightName]});
}
} // namespace infini_train::nn

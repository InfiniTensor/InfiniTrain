#include "infini_train/include/nn/modules/sparse.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/sparse.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

Embedding::Embedding(int num_embeddings, int embedding_dim, Device device) : Module(kType) {
    device_ = device;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{num_embeddings, embedding_dim}, DataType::kFLOAT32, device)
              ->RequiresGrad();
    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> Embedding::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::Embedding>()->Apply({input_tensors[0], parameters_[kParamWeightName]});
}

void Embedding::ResetParameters() { init::Normal(parameters_[kParamWeightName]); }
} // namespace infini_train::nn

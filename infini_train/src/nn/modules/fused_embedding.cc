#include "infini_train/include/nn/modules/fused_embedding.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/fused_embedding.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
constexpr char kParamTokenEmbeddingName[] = "tok_emb";
constexpr char kParamPosEmbeddingName[] = "pos_emb";
} // namespace

FusedEmbedding::FusedEmbedding(int64_t vocab_size, int64_t max_position, int64_t embed_dim, Device device) {
    device_ = device;

    parameters_[kParamTokenEmbeddingName]
        = std::make_shared<Tensor>(std::vector<int64_t>{vocab_size, embed_dim}, DataType::kFLOAT32, device)
              ->RequiresGrad();
    parameters_[kParamPosEmbeddingName]
        = std::make_shared<Tensor>(std::vector<int64_t>{max_position, embed_dim}, DataType::kFLOAT32, device)
              ->RequiresGrad();

    auto *tok_emb = parameters_[kParamTokenEmbeddingName].get();
    tok_emb->Fill<float>(0.5);
    auto *pos_emb = parameters_[kParamPosEmbeddingName].get();
    pos_emb->Fill<float>(0.5);
}

std::vector<std::shared_ptr<Tensor>> FusedEmbedding::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::FusedEmbedding>()->Apply(
        {input_tensors[0], parameters_[kParamTokenEmbeddingName], parameters_[kParamPosEmbeddingName]});
}
} // namespace infini_train::nn

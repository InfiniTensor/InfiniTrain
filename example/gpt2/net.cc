#include "example/gpt2/net.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/embedding.h"
#include "infini_train/include/nn/modules/fused_embedding.h"
#include "infini_train/include/nn/modules/layernorm.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

namespace {
constexpr int kVocabSize = 50257;
constexpr int kSequenceLength = 64;
constexpr int kEmbeddingDim = 768;
}; // namespace

GPT2::GPT2() {
    modules_["embedding"] = std::make_unique<nn::FusedEmbedding>(kVocabSize, kSequenceLength, kEmbeddingDim);
    modules_["layernorm"] = std::make_unique<nn::LayerNorm>(kEmbeddingDim);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto x1 = modules_["embedding"]->Forward(x);
    auto x2 = modules_["layernorm"]->Forward(x1);
    return x2;
}

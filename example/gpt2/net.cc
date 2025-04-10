#include "example/gpt2/net.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/embedding.h"
#include "infini_train/include/nn/modules/layernorm.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

GPT2::GPT2() {
    modules_["embedding"] = std::make_unique<nn::Embedding>(50257, 64, 768);
    modules_["layernorm"] = std::make_unique<nn::LayerNorm>(768);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto x1 = modules_["embedding"]->Forward(x);
    auto x2 = modules_["layernorm"]->Forward(x1);
    return x2;
}

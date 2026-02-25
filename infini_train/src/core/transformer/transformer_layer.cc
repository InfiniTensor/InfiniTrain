#include "infini_train/include/core/transformer/transformer_layer.h"

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

TransformerLayer::TransformerLayer(std::shared_ptr<Module> attn, std::shared_ptr<Module> mlp,
                                   std::shared_ptr<Module> norm1, std::shared_ptr<Module> norm2)
    : attention_(attn), mlp_(mlp), norm1_(norm1), norm2_(norm2) {}

} // namespace infini_train::nn
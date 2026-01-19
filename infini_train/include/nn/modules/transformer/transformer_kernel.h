#pragma once
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/config.h"
#include <memory>

namespace infini_train::nn {

class TransformerKernel {
public:
    virtual ~TransformerKernel() = default;

    // === Embedding / Position ===
    virtual bool UseAbsolutePositionEmbedding() const = 0;

    // === Block ===
    virtual std::shared_ptr<Module> MakeBlock(const TransformerConfig &config) = 0;

    // === Norm ===
    virtual std::shared_ptr<Module> MakeFinalNorm(const TransformerConfig &config) = 0;
};

} // namespace infini_train::nn

#pragma once

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

class TransformerLayer : public Module {
public:
    TransformerLayer(std::shared_ptr<Module> attn, std::shared_ptr<Module> mlp, std::shared_ptr<Module> norm1,
                     std::shared_ptr<Module> norm2);

private:
    std::shared_ptr<Module> attention_;
    std::shared_ptr<Module> mlp_;
    std::shared_ptr<Module> norm1_;
    std::shared_ptr<Module> norm2_;
};
} // namespace infini_train::nn
#pragma once

#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

class TransformerBlock : public Module {
public:
    TransformerBlock(std::vector<std::shared_ptr<Module>> layers);

private:
    std::vector<std::shared_ptr<Module>> layers_;
};
} // namespace infini_train::nn
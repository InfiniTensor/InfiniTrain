#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

class TensorParallelMLP : public infini_train::nn::Module {
public:
    TensorParallelMLP();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

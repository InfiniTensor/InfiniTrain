#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

class MNIST : public infini_train::nn::Module {
public:
    MNIST();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

// Small CNN for MNIST: Conv2d(1,16,3)->ReLU->Conv2d(16,32,3)->ReLU->Flatten(1)->Linear(18432,10).
// Input arrives as [B, 1, 28, 28] (NCHW) from the dataloader pipeline; no reshape hacks needed.
class MnistCnn : public infini_train::nn::Module {
public:
    MnistCnn();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

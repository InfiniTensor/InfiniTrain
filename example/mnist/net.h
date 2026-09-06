#pragma once

#include <cstdlib>
#include <memory>
#include <string>
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

// CNN classifier for MNIST:
// Conv2d(1, 16, 3) -> ReLU -> Conv2d(16, 32, 3) -> ReLU -> Flatten -> Linear(32*24*24, 10)
class MnistCNN : public infini_train::nn::Module {
public:
    static constexpr int kImageSize = 28;
    static constexpr int kNumClasses = 10;

    MnistCNN();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

std::shared_ptr<infini_train::nn::Module> CreateMNISTNetwork(const std::string &model);

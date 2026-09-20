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


// CNN手写数字识别网络
class MNISTCNN : public infini_train::nn::Module {
public:
    MNISTCNN();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

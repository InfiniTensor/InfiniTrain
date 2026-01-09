#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

MNIST::MNIST() {
    std::vector<std::shared_ptr<nn::Module>> layers;
    layers.push_back(std::make_shared<nn::Linear>(784, 30));
    layers.push_back(std::make_shared<nn::Sigmoid>());
    modules_["sequential"] = std::make_shared<nn::Sequential>(std::move(layers));
    modules_["linear2"] = std::make_shared<nn::Linear>(30, 10);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MNIST::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto x1 = (*modules_["sequential"])(x);
    auto x2 = (*modules_["linear2"])(x1);
    return x2;
}

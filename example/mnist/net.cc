#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

MNIST::MNIST() {
    std::vector<std::unique_ptr<nn::Module>> layers;
    layers.push_back(std::make_unique<nn::Linear>(784, 30));
    layers.push_back(std::make_unique<nn::Sigmoid>());
    AddNamedLayer("sequential", std::make_unique<nn::Sequential>(std::move(layers)));
    AddNamedLayer("linear2", std::make_unique<nn::Linear>(30, 10));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MNIST::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto x1 = GetLayer("sequential")->Forward(x);
    auto x2 = GetLayer("linear2")->Forward(x1);
    return x2;
}

#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/convolution.h"
#include "infini_train/include/nn/modules/dropout.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/maxpool.h"
#include "infini_train/include/nn/modules/module.h"

#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

MNIST::MNIST() {
    std::vector<std::unique_ptr<nn::Module>> layers1;
    layers1.push_back(std::make_unique<nn::Conv2D>(1, 10, 3));
    layers1.push_back(std::make_unique<nn::Relu>());
    layers1.push_back(std::make_unique<nn::MaxPool2D>(2, 2));
    layers1.push_back(std::make_unique<nn::Conv2D>(10, 20, 4));
    layers1.push_back(std::make_unique<nn::Relu>());
    layers1.push_back(std::make_unique<nn::MaxPool2D>(2, 2));
    modules_["features"] = std::make_unique<nn::Sequential>(std::move(layers1));

    std::vector<std::unique_ptr<nn::Module>> layers2;
    layers2.push_back(std::make_unique<nn::Dropout>());
    layers2.push_back(std::make_unique<nn::Linear>(500, 50));
    layers2.push_back(std::make_unique<nn::Relu>());
    layers2.push_back(std::make_unique<nn::Dropout>());
    layers2.push_back(std::make_unique<nn::Linear>(50, 10));
    modules_["classifier"] = std::make_unique<nn::Sequential>(std::move(layers2));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MNIST::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto x1 = modules_["features"]->Forward(x)[0];
    auto x2 = x1->Flatten(1);
    auto x3 = modules_["classifier"]->Forward({x2});
    auto x4 = nn::function::LogSoftmax(x3[0], 1);

    return {x4};
}

#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/conv2d.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

MNIST::MNIST() {
    std::vector<std::shared_ptr<nn::Module>> layers;
    layers.push_back(std::make_shared<nn::Conv2d>(1, 16, 3));
    layers.push_back(std::make_shared<nn::Sigmoid>());
    layers.push_back(std::make_shared<nn::Conv2d>(16, 32, 3));
    layers.push_back(std::make_shared<nn::Sigmoid>());
    modules_["sequential"] = std::make_shared<nn::Sequential>(std::move(layers));
    modules_["linear2"] = std::make_shared<nn::Linear>(32 * 24 * 24, 10);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MNIST::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto reshaped = x[0]->View({x[0]->Dims()[0], 1, 28, 28});
    std::vector<std::shared_ptr<infini_train::Tensor>> x_reshaped = {reshaped};
    auto x1 = (*modules_["sequential"])(x_reshaped);
    auto x2 = x1[0]->View({x1[0]->Dims()[0], 32 * 24 * 24})->Contiguous();
    auto x3 = (*modules_["linear2"])({x2});
    return x3;
}
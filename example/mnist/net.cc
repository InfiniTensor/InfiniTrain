#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/conv2d.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

MNIST::MNIST() {
    modules_["conv1"] = std::make_shared<nn::Conv2d>(1, 8, 3, 1, 1);
    modules_["relu1"] = std::make_shared<nn::ReLU>();
    modules_["conv2"] = std::make_shared<nn::Conv2d>(8, 16, 3, 2, 1);
    modules_["relu2"] = std::make_shared<nn::ReLU>();
    modules_["classifier"] = std::make_shared<nn::Linear>(16 * 14 * 14, 10);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MNIST::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto output = (*modules_["conv1"])(x);
    output = (*modules_["relu1"])(output);
    output = (*modules_["conv2"])(output);
    output = (*modules_["relu2"])(output);
    return (*modules_["classifier"])({output[0]->Flatten(1)});
}

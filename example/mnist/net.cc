#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/conv.h"
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
    // Batches arrive as [B, 1, 28, 28] (NCHW); flatten trailing dims so the MLP sees [B, 784] exactly as before.
    auto x0 = x[0]->Flatten(1);
    auto x1 = (*modules_["sequential"])({x0});
    auto x2 = (*modules_["linear2"])(x1);
    return x2;
}

MnistCnn::MnistCnn() {
    modules_["conv1"] = std::make_shared<nn::Conv2d>(1, 16, 3);
    modules_["relu1"] = std::make_shared<nn::ReLU>();
    modules_["conv2"] = std::make_shared<nn::Conv2d>(16, 32, 3);
    modules_["relu2"] = std::make_shared<nn::ReLU>();
    // Shape math (kernel 3, stride 1, padding 0): 28x28 -> 26x26 (16ch) -> 24x24 (32ch),
    // so the classifier sees 32 * 24 * 24 = 18432 features.
    modules_["fc"] = std::make_shared<nn::Linear>(18432, 10);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MnistCnn::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    // Input is [B, 1, 28, 28] (NCHW); conv stack keeps NCHW, output is [B, 32, 24, 24].
    auto h1 = (*modules_["conv1"])(x);
    auto h2 = (*modules_["relu1"])(h1);
    auto h3 = (*modules_["conv2"])(h2);
    auto h4 = (*modules_["relu2"])(h3);
    auto flat = h4[0]->Flatten(1);
    CHECK_EQ(flat->Dims()[1], 32 * 24 * 24) << "MnistCnn feature size mismatch: expected 18432";
    return (*modules_["fc"])({flat});
}

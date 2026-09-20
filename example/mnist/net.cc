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

#include "infini_train/include/nn/modules/conv2d.h"


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

// CNN手写数字识别网络
MNISTCNN::MNISTCNN() {
    modules_["conv1"] = std::make_shared<nn::Conv2d>(1, 16, 3); //in_channels=1, out_channels=16, kernel_size=3
    modules_["relu1"] = std::make_shared<nn::Relu>();
    modules_["conv2"] = std::make_shared<nn::Conv2d>(16, 32, 3); // in_channels=16, out_channels=32, kernel_size=3
    modules_["relu2"] = std::make_shared<nn::Relu>();

    // 变成10个class
    modules_["fc"]= std::make_shared<nn::Linear>(32 * 24 * 24, 10);
}


std::vector<std::shared_ptr<infini_train::Tensor>>
MNISTCNN::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    const auto &input = x[0];
    const int64_t batch_size = input->Dims()[0];

    // DataLoader: {bs, 28, 28}  , Conv2d要求：4-D NCHW -> View {bs, 1, 28, 28}
    // View通过Noop接入autograd，反向时梯度形状自动还原
    auto out = (*modules_["conv1"])({input->View({batch_size, 1, 28, 28})});
    out = (*modules_["relu1"])(out);
    out = (*modules_["conv2"])(out);
    out = (*modules_["relu2"])(out);

    // {bs, 32, 24, 24} ->拉平后接全连接层
    out = {out[0]->Flatten(1)};
    out = (*modules_["fc"])(out);
    return out;
}

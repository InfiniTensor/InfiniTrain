#include "example/mnist/net.h"

#include <memory>
#include <utility>
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
    // Two 3x3 valid convs shrink 28 -> 26 -> 24; the reference net has no pooling.
    layers.push_back(std::make_shared<nn::Conv2d>(1, 16, 3));
    layers.push_back(std::make_shared<nn::ReLU>());
    layers.push_back(std::make_shared<nn::Conv2d>(16, 32, 3));
    layers.push_back(std::make_shared<nn::ReLU>());
    modules_["sequential"] = std::make_shared<nn::Sequential>(std::move(layers));
    // 32 * 24 * 24 = 18432 flattened features into the 10-class head.
    modules_["linear"] = std::make_shared<nn::Linear>(32 * 24 * 24, 10);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MNIST::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    // The loader hands over each image flattened to (N, 784); restore the (N, 1, 28, 28)
    // spatial layout the conv stack expects. View preserves the element order.
    const auto batch = x[0]->Dims()[0];
    auto x_view = x[0]->View({batch, 1, 28, 28});
    auto x_feat = (*modules_["sequential"])({x_view})[0]->Flatten(1, -1);
    return (*modules_["linear"])({x_feat});
}

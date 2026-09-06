#pragma once

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

// Small CNN classifier for the MNIST demo. Structure matches the reference:
// Conv2d(1,16,3) -> ReLU -> Conv2d(16,32,3) -> ReLU -> Flatten -> Linear(18432,10).
// The DataLoader::Stack helper flattens each image into a [N, 784] matrix, so the
// Forward entry restores the (N, 1, 28, 28) spatial layout before the first conv.
class MnistCnn : public infini_train::nn::Module {
public:
    MnistCnn() {
        std::vector<std::shared_ptr<infini_train::nn::Module>> layers;
        // Two 3x3 valid convs shrink 28 -> 26 -> 24; no pooling in the reference net.
        layers.push_back(std::make_shared<infini_train::nn::Conv2d>(1, 16, 3));
        layers.push_back(std::make_shared<infini_train::nn::ReLU>());
        layers.push_back(std::make_shared<infini_train::nn::Conv2d>(16, 32, 3));
        layers.push_back(std::make_shared<infini_train::nn::ReLU>());
        modules_["sequential"] = std::make_shared<infini_train::nn::Sequential>(std::move(layers));
        // 32 * 24 * 24 = 18432 flattened features into the 10-class head.
        modules_["linear"] = std::make_shared<infini_train::nn::Linear>(32 * 24 * 24, 10);
    }

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override {
        CHECK_EQ(x.size(), 1);
        // Restore the batch dimension from the runtime shape, then reshape the flattened
        // [N, 784] input to (N, 1, 28, 28). View is element-order preserving.
        const auto batch = x[0]->Dims()[0];
        auto x_view = x[0]->View({batch, 1, 28, 28});
        auto x_feat = (*modules_["sequential"])({x_view})[0]->Flatten(1, -1);
        return (*modules_["linear"])({x_feat});
    }
};

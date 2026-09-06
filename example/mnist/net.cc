#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/conv.h"
#include "infini_train/include/nn/modules/flatten.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

namespace {
constexpr char kModelMLP[] = "mlp";
constexpr char kModelCNN[] = "cnn";
} // namespace

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

MnistCNN::MnistCNN() {
    std::vector<std::shared_ptr<nn::Module>> features;
    features.push_back(std::make_shared<nn::Conv2d>(1, 16, 3));
    features.push_back(std::make_shared<nn::Relu>());
    features.push_back(std::make_shared<nn::Conv2d>(16, 32, 3));
    features.push_back(std::make_shared<nn::Relu>());
    modules_["features"] = std::make_shared<nn::Sequential>(std::move(features));
    modules_["flatten"] = std::make_shared<nn::Flatten>();
    modules_["classifier"] = std::make_shared<nn::Linear>(32 * 24 * 24, kNumClasses);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MnistCNN::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    // The MNIST DataLoader stacks flattened images as (N, 784); restore the (N, 1, H, W) layout for conv.
    const auto &input = x[0];
    CHECK_EQ(input->Dims().size(), 2);
    auto image = input->View({input->Dims()[0], 1, kImageSize, kImageSize});

    auto features = (*modules_["features"])({image});
    auto flattened = (*modules_["flatten"])(features);
    return (*modules_["classifier"])(flattened);
}

std::shared_ptr<infini_train::nn::Module> CreateMNISTNetwork(const std::string &model) {
    if (model == kModelMLP) {
        return std::make_shared<MNIST>();
    }
    if (model == kModelCNN) {
        return std::make_shared<MnistCNN>();
    }
    LOG(FATAL) << "Unknown model type: " << model << " (expected '" << kModelMLP << "' or '" << kModelCNN << "')";
    return nullptr;
}

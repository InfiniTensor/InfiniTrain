#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/convolution.h"
#include "infini_train/include/nn/modules/flatten.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

MNIST::MNIST() : CloneableModule(kType) {
    modules_["conv1"] = std::make_shared<nn::Conv2d>(1, 16, 3);
    modules_["relu1"] = std::make_shared<nn::ReLU>();
    modules_["conv2"] = std::make_shared<nn::Conv2d>(16, 32, 3);
    modules_["relu2"] = std::make_shared<nn::ReLU>();
    modules_["flatten"] = std::make_shared<nn::Flatten>();
    modules_["fc"] = std::make_shared<nn::Linear>(kFlattenFeatures, kNumClasses);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MNIST::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    CHECK(x[0]);
    CHECK(x[0]->Dtype() == infini_train::DataType::kFLOAT32) << "MNIST CNN expects FP32 images";
    const auto &dims = x[0]->Dims();
    CHECK(dims.size() == 2 || dims.size() == 4) << "MNIST CNN expects [N,784] or [N,1,28,28]";
    CHECK_GT(dims[0], 0);
    auto image = x[0];
    if (dims.size() == 2) {
        CHECK_EQ(dims[1], kImageSize * kImageSize);
        image = image->View({dims[0], 1, kImageSize, kImageSize});
    } else {
        CHECK_EQ(dims[1], 1);
        CHECK_EQ(dims[2], kImageSize);
        CHECK_EQ(dims[3], kImageSize);
    }
    std::vector<std::shared_ptr<infini_train::Tensor>> outputs{image};
    for (const auto *name : {"conv1", "relu1", "conv2", "relu2", "flatten", "fc"}) {
        outputs = (*modules_.at(name))(outputs);
    }
    return outputs;
}

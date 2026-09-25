#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

// FP32 CNN for 28x28 grayscale images. Accepts [N,784] (DataLoader) or
// [N,1,28,28] and returns unnormalized [N,10] logits for CrossEntropyLoss.
class MNIST : public infini_train::nn::CloneableModule<MNIST> {
public:
    static constexpr char kType[] = "MNIST";
    static constexpr int64_t kImageSize = 28;
    static constexpr int64_t kNumClasses = 10;
    static constexpr int64_t kFlattenFeatures = 32 * 24 * 24;
    MNIST();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

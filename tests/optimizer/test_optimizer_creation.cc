#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerCreationTest : public infini_train::test::InfiniTrainTest {};

TEST_F(OptimizerCreationTest, SGDCreation) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    EXPECT_NE(optimizer, nullptr);
}

TEST_F(OptimizerCreationTest, AdamCreation) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::Adam>(params, 0.001);

    EXPECT_NE(optimizer, nullptr);
}

TEST_F(OptimizerCreationTest, SGDWithMomentum) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    EXPECT_NE(optimizer, nullptr);
}

TEST_F(OptimizerCreationTest, SGDMultiParams) {
    std::vector<std::shared_ptr<Tensor>> params;
    for (int i = 0; i < 3; ++i) {
        auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                              Device(Device::DeviceType::kCPU, 0));
        param->set_requires_grad(true);
        params.push_back(param);
    }

    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);
    EXPECT_NE(optimizer, nullptr);
}

TEST_F(OptimizerCreationTest, AdamMultiParams) {
    std::vector<std::shared_ptr<Tensor>> params;
    for (int i = 0; i < 3; ++i) {
        auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                              Device(Device::DeviceType::kCPU, 0));
        param->set_requires_grad(true);
        params.push_back(param);
    }

    auto optimizer = std::make_shared<optimizers::Adam>(params, 0.001);
    EXPECT_NE(optimizer, nullptr);
}

TEST_F(OptimizerCreationTest, SGDWithWeightDecay) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    EXPECT_NE(optimizer, nullptr);
}

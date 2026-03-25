#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerStepTest : public infini_train::test::InfiniTrainTest {};

TEST_F(OptimizerStepTest, SGDStep) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);
    auto* data = static_cast<float*>(param->DataPtr());
    for (int i = 0; i < 6; ++i) data[i] = 1.0f;

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    optimizer->ZeroGrad();
    optimizer->Step();
}

TEST_F(OptimizerStepTest, AdamStep) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);
    auto* data = static_cast<float*>(param->DataPtr());
    for (int i = 0; i < 6; ++i) data[i] = 1.0f;

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::Adam>(params, 0.001);

    optimizer->ZeroGrad();
    optimizer->Step();
}

TEST_F(OptimizerStepTest, ZeroGrad) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    optimizer->ZeroGrad();
}

TEST_F(OptimizerStepTest, ZeroGradWithNone) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    optimizer->ZeroGrad(false);
}

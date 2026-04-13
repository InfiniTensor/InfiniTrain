#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class OptimizerStepTest : public infini_train::test::InfiniTrainTestP {};

TEST_P(OptimizerStepTest, SGDStep) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    infini_train::test::FillConstantTensor(param, 1.0f);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
    optimizer->ZeroGrad();
    optimizer->Step();
}

TEST_P(OptimizerStepTest, AdamStep) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    infini_train::test::FillConstantTensor(param, 1.0f);
    auto optimizer = std::make_shared<optimizers::Adam>(std::vector<std::shared_ptr<Tensor>>{param}, 0.001);
    optimizer->ZeroGrad();
    optimizer->Step();
}

TEST_P(OptimizerStepTest, ZeroGrad) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
    optimizer->ZeroGrad();
}

TEST_P(OptimizerStepTest, ZeroGradWithNone) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
    optimizer->ZeroGrad(false);
}

TEST_P(OptimizerStepTest, SGDMultiParams) {
    std::vector<std::shared_ptr<Tensor>> params;
    for (int i = 0; i < 3; ++i) {
        auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
        param->set_requires_grad(true);
        params.push_back(param);
    }
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);
    EXPECT_NE(optimizer, nullptr);
    optimizer->ZeroGrad();
}

INFINI_TRAIN_REGISTER_TEST(OptimizerStepTest);

// ---------------------------------------------------------------------------
// Distributed
// ---------------------------------------------------------------------------

class OptimizerStepDistributedTest : public infini_train::test::DistributedInfiniTrainTestP {};

TEST_P(OptimizerStepDistributedTest, ZeroGrad) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
    optimizer->ZeroGrad();
}

TEST_P(OptimizerStepDistributedTest, StepMultiParams) {
    std::vector<std::shared_ptr<Tensor>> params;
    for (int i = 0; i < 2; ++i) {
        auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
        param->set_requires_grad(true);
        params.push_back(param);
    }
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);
    EXPECT_NE(optimizer, nullptr);
    optimizer->ZeroGrad();
    optimizer->Step();
}

INFINI_TRAIN_REGISTER_TEST_DISTRIBUTED(OptimizerStepDistributedTest);

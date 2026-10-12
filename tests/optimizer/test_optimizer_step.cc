#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerStepTest : public infini_train::test::InfiniTrainTest {};

TEST_P(OptimizerStepTest, SGDStep) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    param->Fill(1.0f);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
    optimizer->ZeroGrad();
    optimizer->Step();
}

TEST_P(OptimizerStepTest, AdamStep) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    param->Fill(1.0f);
    auto optimizer = std::make_shared<optimizers::Adam>(std::vector<std::shared_ptr<Tensor>>{param}, 0.001);
    optimizer->ZeroGrad();
    optimizer->Step();
}

TEST_P(OptimizerStepTest, AdamUpdatesBF16ParameterWithFP32State) {
    if (GetDevice().type() != Device::DeviceType::kCUDA) {
        GTEST_SKIP() << "BF16 Adam update is CUDA-only";
    }
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kBFLOAT16, GetDevice());
    param->set_requires_grad(true);
    param->Fill(1.0f);
    auto grad = std::make_shared<Tensor>(param->Dims(), DataType::kBFLOAT16, GetDevice());
    grad->Fill(0.5f);
    param->set_grad(grad);

    auto optimizer = std::make_shared<optimizers::Adam>(std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
    optimizer->Step();

    const auto updated = param->To(DataType::kFLOAT32).To(Device());
    const auto *data = static_cast<const float *>(updated.DataPtr());
    EXPECT_LT(data[0], 1.0f);
    EXPECT_EQ(optimizer->StateDict().at("adam.m.0")->Dtype(), DataType::kFLOAT32);
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

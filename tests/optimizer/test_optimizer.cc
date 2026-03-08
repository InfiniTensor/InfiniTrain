#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "test_utils.h"

using namespace infini_train;

class OptimizerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);
    }
};

TEST_F(OptimizerTest, SGDCreation) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    EXPECT_NE(optimizer, nullptr);
}

TEST_F(OptimizerTest, AdamCreation) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::Adam>(params, 0.001);

    EXPECT_NE(optimizer, nullptr);
}

TEST_F(OptimizerTest, ZeroGrad) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    optimizer->ZeroGrad();
}

TEST_F(OptimizerTest, SGDMultiParams) {
    std::vector<std::shared_ptr<Tensor>> params;
    for (int i = 0; i < 3; ++i) {
        auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                              Device(Device::DeviceType::kCPU, 0));
        param->set_requires_grad(true);
        params.push_back(param);
    }

    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);
    EXPECT_NE(optimizer, nullptr);

    optimizer->ZeroGrad();
}

TEST_F(OptimizerTest, SGDCreationCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    EXPECT_NE(optimizer, nullptr);
    EXPECT_TRUE(param->IsCUDA());
#endif
}

TEST_F(OptimizerTest, AdamCreationCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::Adam>(params, 0.001);

    EXPECT_NE(optimizer, nullptr);
    EXPECT_TRUE(param->IsCUDA());
#endif
}

TEST_F(OptimizerTest, ZeroGradCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    optimizer->ZeroGrad();
    EXPECT_TRUE(param->IsCUDA());
#endif
}

TEST_F(OptimizerTest, SGDMultiParamsCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    std::vector<std::shared_ptr<Tensor>> params;
    for (int i = 0; i < 3; ++i) {
        auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                              Device(Device::DeviceType::kCUDA, 0));
        param->set_requires_grad(true);
        params.push_back(param);
    }

    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);
    EXPECT_NE(optimizer, nullptr);

    optimizer->ZeroGrad();
#endif
}

TEST_F(OptimizerTest, DistributedSGD) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    EXPECT_NE(optimizer, nullptr);
    EXPECT_TRUE(param->IsCUDA());
#endif
}

TEST_F(OptimizerTest, DistributedAdam) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{4, 4}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::Adam>(params, 0.001);

    EXPECT_NE(optimizer, nullptr);
    EXPECT_TRUE(param->IsCUDA());
#endif
}

TEST_F(OptimizerTest, DistributedZeroGrad) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    optimizer->ZeroGrad();
#endif
}

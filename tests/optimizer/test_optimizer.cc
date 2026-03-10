#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "test_utils.h"

using namespace infini_train;

class OptimizerTestBase : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);
    }
};

class OptimizerCreationTest : public OptimizerTestBase {};
class OptimizerGradTest : public OptimizerTestBase {};
class OptimizerCudaTest : public OptimizerTestBase {};
class OptimizerDistributedTest : public OptimizerTestBase {};

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

TEST_F(OptimizerGradTest, ZeroGrad) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCPU, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    optimizer->ZeroGrad();
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

    optimizer->ZeroGrad();
}

TEST_F(OptimizerCudaTest, SGDCreationCUDA) {
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

TEST_F(OptimizerCudaTest, AdamCreationCUDA) {
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

TEST_F(OptimizerCudaTest, ZeroGradCUDA) {
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

TEST_F(OptimizerCudaTest, SGDMultiParamsCUDA) {
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

TEST_F(OptimizerDistributedTest, DistributedSGD) {
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

TEST_F(OptimizerDistributedTest, DistributedAdam) {
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

TEST_F(OptimizerDistributedTest, DistributedZeroGrad) {
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

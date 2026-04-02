#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerDistributedTest : public infini_train::test::InfiniTrainTest {};

TEST_F(OptimizerDistributedTest, DistributedSGD) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    param->set_requires_grad(true);

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);

    EXPECT_NE(optimizer, nullptr);
    EXPECT_TRUE(param->GetDevice().IsCUDA());
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
    EXPECT_TRUE(param->GetDevice().IsCUDA());
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

TEST_F(OptimizerDistributedTest, DistributedMultiParams) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    std::vector<std::shared_ptr<Tensor>> params;
    for (int i = 0; i < 2; ++i) {
        auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                              Device(Device::DeviceType::kCUDA, 0));
        param->set_requires_grad(true);
        params.push_back(param);
    }

    auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);
    EXPECT_NE(optimizer, nullptr);

    optimizer->ZeroGrad();
    optimizer->Step();
#endif
}

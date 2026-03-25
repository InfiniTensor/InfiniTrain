#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerCUDATest : public infini_train::test::InfiniTrainTest {};

TEST_F(OptimizerCUDATest, SGDCreationCUDA) {
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

TEST_F(OptimizerCUDATest, AdamCreationCUDA) {
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

TEST_F(OptimizerCUDATest, ZeroGradCUDA) {
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

TEST_F(OptimizerCUDATest, SGDMultiParamsCUDA) {
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

TEST_F(OptimizerCUDATest, AdamStepCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    param->set_requires_grad(true);
    auto* data = static_cast<float*>(param->DataPtr());
    for (int i = 0; i < 6; ++i) data[i] = 1.0f;

    std::vector<std::shared_ptr<Tensor>> params = {param};
    auto optimizer = std::make_shared<optimizers::Adam>(params, 0.001);

    optimizer->ZeroGrad();
    optimizer->Step();
    EXPECT_TRUE(param->IsCUDA());
#endif
}

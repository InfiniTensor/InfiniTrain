#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "test_utils.h"

using namespace infini_train;

class TensorTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);
    }
};

TEST_F(TensorTest, CreateAndDestroy) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensor->Dtype(), DataType::kFLOAT32);
}

TEST_F(TensorTest, RequiresGrad) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    EXPECT_FALSE(tensor->requires_grad());
    tensor->set_requires_grad(true);
    EXPECT_TRUE(tensor->requires_grad());
}

TEST_F(TensorTest, DataPointer) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    EXPECT_NE(tensor->DataPtr(), nullptr);
}

TEST_F(TensorTest, DifferentShapes) {
    std::vector<std::vector<int64_t>> shapes = {
        {2, 3},
        {4, 5, 6},
        {10},
        {1, 1, 1, 1}
    };

    for (const auto& shape : shapes) {
        auto tensor = std::make_shared<Tensor>(shape, DataType::kFLOAT32,
                                               Device(Device::DeviceType::kCPU, 0));
        EXPECT_EQ(tensor->Dims(), shape);
    }
}

TEST_F(TensorTest, DifferentDataTypes) {
    std::vector<DataType> dtypes = {
        DataType::kFLOAT32,
        DataType::kBFLOAT16,
    };

    for (const auto& dtype : dtypes) {
        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, dtype,
                                               Device(Device::DeviceType::kCPU, 0));
        EXPECT_EQ(tensor->Dtype(), dtype);
    }
}

TEST_F(TensorTest, CreateCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensor->Dtype(), DataType::kFLOAT32);
    EXPECT_TRUE(tensor->IsCUDA());
#endif
}

TEST_F(TensorTest, RequiresGradCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    EXPECT_FALSE(tensor->requires_grad());
    tensor->set_requires_grad(true);
    EXPECT_TRUE(tensor->requires_grad());
#endif
}

TEST_F(TensorTest, DataPointerCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    EXPECT_NE(tensor->DataPtr(), nullptr);
#endif
}

TEST_F(TensorTest, TensorCopyCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto cpu_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                               Device(Device::DeviceType::kCPU, 0));
    auto cuda_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                                 Device(Device::DeviceType::kCUDA, 0));
    
    auto* cpu_data = static_cast<float*>(cpu_tensor->DataPtr());
    for (int i = 0; i < 6; ++i) cpu_data[i] = static_cast<float>(i);
    
    cuda_tensor->CopyDataFrom(cpu_tensor.get());
    
    EXPECT_TRUE(cuda_tensor->IsCUDA());
#endif
}

TEST_F(TensorTest, MatmulCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                      Device(Device::DeviceType::kCUDA, 0));
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32,
                                      Device(Device::DeviceType::kCUDA, 0));
    auto c = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kFLOAT32,
                                      Device(Device::DeviceType::kCUDA, 0));
    EXPECT_NE(a->DataPtr(), nullptr);
    EXPECT_NE(b->DataPtr(), nullptr);
    EXPECT_NE(c->DataPtr(), nullptr);
    EXPECT_TRUE(a->IsCUDA());
    EXPECT_TRUE(b->IsCUDA());
    EXPECT_TRUE(c->IsCUDA());
#endif
}

TEST_F(TensorTest, DistributedAllReduce) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    tensor->set_requires_grad(true);
    
    auto* data = static_cast<float*>(tensor->DataPtr());
    for (int i = 0; i < 6; ++i) data[i] = 1.0f;
    
    EXPECT_TRUE(tensor->IsCUDA());
    EXPECT_TRUE(tensor->requires_grad());
#endif
}

TEST_F(TensorTest, DistributedAllGather) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{4, 4}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    tensor->set_requires_grad(true);
    
    EXPECT_TRUE(tensor->IsCUDA());
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{4, 4}));
#endif
}

TEST_F(TensorTest, DistributedReduceScatter) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 8}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    tensor->set_requires_grad(true);
    
    EXPECT_TRUE(tensor->IsCUDA());
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 8}));
#endif
}

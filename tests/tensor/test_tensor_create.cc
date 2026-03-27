#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "test_utils.h"

using namespace infini_train;

class TensorCreateTest : public infini_train::test::TensorTestBase {};

TEST_F(TensorCreateTest, CreatesCpuTensorWithShapeAndType) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensor->Dtype(), DataType::kFLOAT32);
}

TEST_F(TensorCreateTest, TracksRequiresGrad) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    EXPECT_FALSE(tensor->requires_grad());
    tensor->set_requires_grad(true);
    EXPECT_TRUE(tensor->requires_grad());
}

TEST_F(TensorCreateTest, ProvidesDataPointer) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    EXPECT_NE(tensor->DataPtr(), nullptr);
}

TEST_F(TensorCreateTest, SupportsMultipleShapes) {
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

TEST_F(TensorCreateTest, SupportsMultipleDtypes) {
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

TEST_F(TensorCreateTest, CreatesTensorOnCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    EXPECT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->GetDevice().IsCUDA());
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensor->Dtype(), DataType::kFLOAT32);
#endif
}

TEST_F(TensorCreateTest, TracksRequiresGradOnCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    EXPECT_FALSE(tensor->requires_grad());
    tensor->set_requires_grad(true);
    EXPECT_TRUE(tensor->requires_grad());
#endif
}

TEST_F(TensorCreateTest, ProvidesDataPointerOnCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    EXPECT_NE(tensor->DataPtr(), nullptr);
#endif
}

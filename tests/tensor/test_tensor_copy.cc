#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "test_utils.h"

using namespace infini_train;

class TensorCopyTest : public infini_train::test::TensorTestBase {};

static void FillSequential(const std::shared_ptr<Tensor>& tensor, float start = 0.0f) {
    auto* data = static_cast<float*>(tensor->DataPtr());
    size_t n = 1;
    for (auto dim : tensor->Dims()) {
        n *= static_cast<size_t>(dim);
    }
    for (size_t i = 0; i < n; ++i) {
        data[i] = start + static_cast<float>(i);
    }
}

TEST_F(TensorCopyTest, CopiesCPUToCPU) {
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    FillSequential(source, 1.0f);

    target->CopyFrom(source);

    auto* target_data = static_cast<float*>(target->DataPtr());
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(target_data[i], 1.0f + static_cast<float>(i));
    }
}

TEST_F(TensorCopyTest, CopiesCPUToCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto cpu_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                               Device(Device::DeviceType::kCPU, 0));
    auto cuda_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                                 Device(Device::DeviceType::kCUDA, 0));

    FillSequential(cpu_tensor, 0.0f);
    cuda_tensor->CopyFrom(cpu_tensor);

    EXPECT_TRUE(cuda_tensor->IsCUDA());
#endif
}

TEST_F(TensorCopyTest, CopiesCUDAtoCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    FillSequential(source, 2.0f);

    target->CopyFrom(source);

    EXPECT_TRUE(target->IsCUDA());
#endif
}

TEST_F(TensorCopyTest, CopiesCUDAtoCPU) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto cuda_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                                 Device(Device::DeviceType::kCUDA, 0));
    auto cpu_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                                Device(Device::DeviceType::kCPU, 0));

    FillSequential(cuda_tensor, 1.0f);
    cpu_tensor->CopyFrom(cuda_tensor);

    EXPECT_FALSE(cpu_tensor->IsCUDA());
    EXPECT_TRUE(cpu_tensor->IsCPU());
#endif
}

TEST_F(TensorCopyTest, CopiesBetweenSameShape) {
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{4, 5, 6}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{4, 5, 6}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    FillSequential(source, 0.0f);

    target->CopyFrom(source);

    EXPECT_EQ(source->Dims(), target->Dims());
}

TEST_F(TensorCopyTest, CopiesPreservesDataType) {
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));

    EXPECT_EQ(source->Dtype(), target->Dtype());
    target->CopyFrom(source);
    EXPECT_EQ(target->Dtype(), DataType::kFLOAT32);
}

TEST_F(TensorCopyTest, CopiesWithDifferentDeviceId) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 1));
    FillSequential(source, 5.0f);

    target->CopyFrom(source);

    EXPECT_EQ(source->GetDevice().index(), 0);
    EXPECT_EQ(target->GetDevice().index(), 1);
#endif
}

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class TensorCopyCudaTest : public ::testing::Test {};

TEST_F(TensorCopyCudaTest, CopiesCPUToCUDA) {
    auto cpu_tensor
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    auto cuda_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                                Device(Device::DeviceType::kCUDA, 0));
    cpu_tensor->Fill(0.0f);
    cuda_tensor->CopyFrom(cpu_tensor);
    EXPECT_TRUE(cuda_tensor->GetDevice().IsCUDA());
}

TEST_F(TensorCopyCudaTest, CopiesCUDAtoCPU) {
    auto cuda_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                                Device(Device::DeviceType::kCUDA, 0));
    auto cpu_tensor
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    cuda_tensor->Fill(1.0f);
    cpu_tensor->CopyFrom(cuda_tensor);
    EXPECT_TRUE(cpu_tensor->GetDevice().IsCPU());
}

TEST_F(TensorCopyCudaTest, CopiesCUDAtoCUDA) {
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    source->Fill(2.0f);
    target->CopyFrom(source);
    EXPECT_TRUE(target->GetDevice().IsCUDA());
}

TEST_F(TensorCopyCudaTest, CopiesWithDifferentDeviceId) {
    REQUIRE_MIN_DEVICES(2);
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 1));
    source->Fill(5.0f);
    target->CopyFrom(source);
    EXPECT_EQ(source->GetDevice().index(), 0);
    EXPECT_EQ(target->GetDevice().index(), 1);
}

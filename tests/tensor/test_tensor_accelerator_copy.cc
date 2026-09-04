#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class TensorAcceleratorCopyTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TensorAcceleratorCopyTest, CopiesCPUToAccelerator) {
    SKIP_CPU();
    const Device accelerator = GetDevice();
    auto cpu_tensor
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    auto accelerator_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, accelerator);
    cpu_tensor->Fill(0.0f);
    accelerator_tensor->CopyFrom(cpu_tensor);
    EXPECT_EQ(accelerator_tensor->GetDevice(), accelerator);
}

TEST_P(TensorAcceleratorCopyTest, CopiesAcceleratorToCPU) {
    SKIP_CPU();
    const Device accelerator = GetDevice();
    auto accelerator_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, accelerator);
    auto cpu_tensor
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    accelerator_tensor->Fill(1.0f);
    cpu_tensor->CopyFrom(accelerator_tensor);
    EXPECT_TRUE(cpu_tensor->GetDevice().IsCPU());
}

TEST_P(TensorAcceleratorCopyTest, CopiesWithinAccelerator) {
    SKIP_CPU();
    const Device accelerator = GetDevice();
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, accelerator);
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, accelerator);
    source->Fill(2.0f);
    target->CopyFrom(source);
    EXPECT_EQ(target->GetDevice(), accelerator);
}

TEST_P(TensorAcceleratorCopyTest, CopiesAcrossAcceleratorDevices) {
    SKIP_CPU();
    REQUIRE_MIN_DEVICES(2);
    const Device source_device = GetDevice();
    const Device target_device(source_device.type(), 1);
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, source_device);
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, target_device);
    source->Fill(5.0f);
    target->CopyFrom(source);
    EXPECT_EQ(source->GetDevice(), source_device);
    EXPECT_EQ(target->GetDevice(), target_device);
}

INFINI_TRAIN_REGISTER_TEST(TensorAcceleratorCopyTest);

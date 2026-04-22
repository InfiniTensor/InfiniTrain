#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class TensorCopyTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TensorCopyTest, CopiesCPUToCPU) {
    ONLY_CPU();
    auto source
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    auto target
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    infini_train::test::FillSequentialTensor(source, 1.0f);
    target->CopyFrom(source);
    auto *target_data = static_cast<float *>(target->DataPtr());
    for (int i = 0; i < 6; ++i) { EXPECT_FLOAT_EQ(target_data[i], 1.0f + static_cast<float>(i)); }
}

TEST_P(TensorCopyTest, CopiesBetweenSameShape) {
    auto source = createTensor({4, 5, 6});
    auto target = createTensor({4, 5, 6});
    infini_train::test::FillSequentialTensor(source, 0.0f);
    target->CopyFrom(source);
    EXPECT_EQ(source->Dims(), target->Dims());
}

TEST_P(TensorCopyTest, CopiesPreservesDataType) {
    auto source = createTensor({2, 3});
    auto target = createTensor({2, 3});
    EXPECT_EQ(source->Dtype(), target->Dtype());
    target->CopyFrom(source);
    EXPECT_EQ(target->Dtype(), DataType::kFLOAT32);
}

TEST_P(TensorCopyTest, CopiesCPUToCUDA) {
    ONLY_CUDA();
    auto cpu_tensor
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    auto cuda_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                                Device(Device::DeviceType::kCUDA, 0));
    infini_train::test::FillSequentialTensor(cpu_tensor, 0.0f);
    cuda_tensor->CopyFrom(cpu_tensor);
    EXPECT_TRUE(cuda_tensor->GetDevice().IsCUDA());
}

TEST_P(TensorCopyTest, CopiesCUDAtoCPU) {
    ONLY_CUDA();
    auto cuda_tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                                Device(Device::DeviceType::kCUDA, 0));
    auto cpu_tensor
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    infini_train::test::FillSequentialTensor(cuda_tensor, 1.0f);
    cpu_tensor->CopyFrom(cuda_tensor);
    EXPECT_TRUE(cpu_tensor->GetDevice().IsCPU());
}

TEST_P(TensorCopyTest, CopiesCUDAtoCUDA) {
    ONLY_CUDA();
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    infini_train::test::FillSequentialTensor(source, 2.0f);
    target->CopyFrom(source);
    EXPECT_TRUE(target->GetDevice().IsCUDA());
}

TEST_P(TensorCopyTest, CopiesWithDifferentDeviceId) {
    ONLY_CUDA();
    REQUIRE_MIN_DEVICES(2);
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 1));
    infini_train::test::FillSequentialTensor(source, 5.0f);
    target->CopyFrom(source);
    EXPECT_EQ(source->GetDevice().index(), 0);
    EXPECT_EQ(target->GetDevice().index(), 1);
}

INFINI_TRAIN_REGISTER_TEST(TensorCopyTest);

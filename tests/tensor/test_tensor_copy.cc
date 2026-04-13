#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class TensorCopyTest : public infini_train::test::TensorTestBaseP {};

// CPU-to-CPU copy is special: source is always CPU regardless of param device.
TEST_P(TensorCopyTest, CopiesCPUToCPU) {
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
    if (GetParam() != Device::DeviceType::kCUDA) {
        GTEST_SKIP() << "CPU-to-CUDA copy only runs in CUDA instantiation";
    }
    auto cpu_tensor
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    auto cuda_tensor = createTensor({2, 3});
    infini_train::test::FillSequentialTensor(cpu_tensor, 0.0f);
    cuda_tensor->CopyFrom(cpu_tensor);
    EXPECT_TRUE(cuda_tensor->GetDevice().IsCUDA());
}

TEST_P(TensorCopyTest, CopiesCUDAtoCUDA) {
    if (GetParam() != Device::DeviceType::kCUDA) {
        GTEST_SKIP() << "CUDA-to-CUDA copy only runs in CUDA instantiation";
    }
    auto source = createTensor({2, 3});
    auto target = createTensor({2, 3});
    infini_train::test::FillSequentialTensor(source, 2.0f);
    target->CopyFrom(source);
    EXPECT_TRUE(target->GetDevice().IsCUDA());
}

TEST_P(TensorCopyTest, CopiesCUDAtoCPU) {
    if (GetParam() != Device::DeviceType::kCUDA) {
        GTEST_SKIP() << "CUDA-to-CPU copy only runs in CUDA instantiation";
    }
    auto cuda_tensor = createTensor({2, 3});
    auto cpu_tensor
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    infini_train::test::FillSequentialTensor(cuda_tensor, 1.0f);
    cpu_tensor->CopyFrom(cuda_tensor);
    EXPECT_FALSE(cpu_tensor->GetDevice().IsCUDA());
    EXPECT_TRUE(cpu_tensor->GetDevice().IsCPU());
}

TEST_P(TensorCopyTest, CopiesWithDifferentDeviceId) {
    if (GetParam() != Device::DeviceType::kCUDA) {
        GTEST_SKIP() << "multi-GPU copy only runs in CUDA instantiation";
    }
    REQUIRE_MIN_GPUS(2);
#if defined(USE_CUDA)
    auto source = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    auto target = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 1));
    infini_train::test::FillSequentialTensor(source, 5.0f);
    target->CopyFrom(source);
    EXPECT_EQ(source->GetDevice().index(), 0);
    EXPECT_EQ(target->GetDevice().index(), 1);
#endif
}

INFINI_TRAIN_REGISTER_TEST(TensorCopyTest);

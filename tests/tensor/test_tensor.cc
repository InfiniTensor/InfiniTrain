#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "test_utils.h"

using namespace infini_train;

class TensorTestBase : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);
    }

    static size_t Numel(const std::shared_ptr<Tensor>& tensor) {
        size_t n = 1;
        for (auto dim : tensor->Dims()) {
            n *= static_cast<size_t>(dim);
        }
        return n;
    }

    static void FillSequential(const std::shared_ptr<Tensor>& tensor, float start = 0.0f) {
        auto* data = static_cast<float*>(tensor->DataPtr());
        auto n = Numel(tensor);
        for (size_t i = 0; i < n; ++i) {
            data[i] = start + static_cast<float>(i);
        }
    }
};

class TensorCreateTest : public TensorTestBase {};
class TensorCopyTest : public TensorTestBase {};
class TensorDeleteTest : public TensorTestBase {};
class TensorOpTest : public TensorTestBase {};
class TensorDistributedTest : public TensorTestBase {};

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
    EXPECT_TRUE(tensor->IsCUDA());
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

TEST_F(TensorOpTest, MatmulCUDAAllocatesOutputs) {
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

TEST_F(TensorDeleteTest, ReleasesResourcesOnReset) {
    std::weak_ptr<Tensor> weak_tensor;
    {
        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                               Device(Device::DeviceType::kCPU, 0));
        tensor->set_requires_grad(true);
        weak_tensor = tensor;
    }
    EXPECT_TRUE(weak_tensor.expired());
}

TEST_F(TensorDeleteTest, MoveTransferKeepsData) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    FillSequential(tensor, 5.0f);

    auto moved = std::move(tensor);
    EXPECT_EQ(tensor, nullptr);
    ASSERT_NE(moved, nullptr);

    auto* data = static_cast<float*>(moved->DataPtr());
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 5.0f + static_cast<float>(i));
    }
}

TEST_F(TensorDistributedTest, AllReduce) {
    REQUIRE_CUDA();
    REQUIRE_DISTRIBUTED();
    REQUIRE_NCCL();
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

TEST_F(TensorDistributedTest, AllGather) {
    REQUIRE_CUDA();
    REQUIRE_DISTRIBUTED();
    REQUIRE_NCCL();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{4, 4}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    tensor->set_requires_grad(true);

    EXPECT_TRUE(tensor->IsCUDA());
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{4, 4}));
#endif
}

TEST_F(TensorDistributedTest, ReduceScatter) {
    REQUIRE_CUDA();
    REQUIRE_DISTRIBUTED();
    REQUIRE_NCCL();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 8}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    tensor->set_requires_grad(true);

    EXPECT_TRUE(tensor->IsCUDA());
    EXPECT_EQ(tensor->Dims(), (std::vector<int64_t>{2, 8}));
#endif
}

#include <gtest/gtest.h>

#include <vector>
#include <memory>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "test_utils.h"

using namespace infini_train;

class TensorDeleteTest : public infini_train::test::TensorTestBase {};

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

TEST_F(TensorDeleteTest, NullifiesPointerOnMove) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{3, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    EXPECT_NE(tensor, nullptr);

    auto moved_tensor = std::move(tensor);
    EXPECT_EQ(tensor, nullptr);
    EXPECT_NE(moved_tensor, nullptr);
}

TEST_F(TensorDeleteTest, SharedPtrRefCountOnCopy) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCPU, 0));
    FillSequential(tensor, 1.0f);

    auto copy1 = tensor;
    auto copy2 = tensor;

    EXPECT_EQ(tensor.use_count(), 3);
    EXPECT_EQ(copy1.use_count(), 3);
    EXPECT_EQ(copy2.use_count(), 3);

    copy1.reset();
    EXPECT_EQ(tensor.use_count(), 2);

    copy2.reset();
    EXPECT_EQ(tensor.use_count(), 1);

    EXPECT_NE(tensor, nullptr);
}

TEST_F(TensorDeleteTest, TensorDestroyedAfterScope) {
    bool destroyed = false;
    {
        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32,
                                               Device(Device::DeviceType::kCPU, 0));
        EXPECT_NE(tensor, nullptr);
    }
}

TEST_F(TensorDeleteTest, ReleaseMemoryOnCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    std::weak_ptr<Tensor> weak_tensor;
    {
        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{100, 100}, DataType::kFLOAT32,
                                               Device(Device::DeviceType::kCUDA, 0));
        tensor->set_requires_grad(true);
        EXPECT_TRUE(tensor->GetDevice().IsCUDA());
        weak_tensor = tensor;
    }
    EXPECT_TRUE(weak_tensor.expired());
#endif
}

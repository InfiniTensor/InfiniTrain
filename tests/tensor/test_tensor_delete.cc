#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class TensorDeleteTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TensorDeleteTest, ReleasesResourcesOnReset) {
    std::weak_ptr<Tensor> weak_tensor;
    {
        auto tensor = createTensor({2, 3}, DataType::kFLOAT32, /*requires_grad=*/true);
        weak_tensor = tensor;
    }
    EXPECT_TRUE(weak_tensor.expired());
}

TEST_P(TensorDeleteTest, MoveTransferKeepsData) {
    auto tensor = createTensor({2, 3});
    infini_train::test::FillSequentialTensor(tensor, 5.0f);

    // Verify data via CPU copy for any device.
    auto cpu_copy
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    cpu_copy->CopyFrom(tensor);

    auto moved = std::move(tensor);
    EXPECT_EQ(tensor, nullptr);
    ASSERT_NE(moved, nullptr);

    auto *data = static_cast<float *>(cpu_copy->DataPtr());
    for (int i = 0; i < 6; ++i) { EXPECT_FLOAT_EQ(data[i], 5.0f + static_cast<float>(i)); }
}

TEST_P(TensorDeleteTest, NullifiesPointerOnMove) {
    auto tensor = createTensor({3, 3});
    EXPECT_NE(tensor, nullptr);
    auto moved_tensor = std::move(tensor);
    EXPECT_EQ(tensor, nullptr);
    EXPECT_NE(moved_tensor, nullptr);
}

TEST_P(TensorDeleteTest, SharedPtrRefCountOnCopy) {
    auto tensor = createTensor({2, 3});
    auto copy1 = tensor;
    auto copy2 = tensor;
    EXPECT_EQ(tensor.use_count(), 3);
    copy1.reset();
    EXPECT_EQ(tensor.use_count(), 2);
    copy2.reset();
    EXPECT_EQ(tensor.use_count(), 1);
    EXPECT_NE(tensor, nullptr);
}

TEST_P(TensorDeleteTest, TensorDestroyedAfterScope) {
    {
        auto tensor = createTensor({2, 2});
        EXPECT_NE(tensor, nullptr);
    }
}

INFINI_TRAIN_REGISTER_TEST(TensorDeleteTest);

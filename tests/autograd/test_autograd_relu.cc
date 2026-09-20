// ReLU autograd节点的单元测试 （CPU/CUDA 双设备）
// Forward Backward各一个用例, 均为手算精确值

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
// CPU上写入, 搬运到设备上， 打开requires_grad
std::shared_ptr<Tensor> MakeTensor(const std::vector<int64_t> &dims, const std::vector<float> &values, Device device) {
    auto cpu_tensor = std::make_shared<Tensor>(dims, DataType::kFLOAT32);
    std::copy(values.begin(), values.end(), static_cast<float *>(cpu_tensor->DataPtr()));

    return std::make_shared<Tensor>(cpu_tensor->To(device))->RequiresGrad();
}
}

class AutogradReluTest : public infini_train::test::InfiniTrainTest {};

// 前向y = max(x, 0)
// 输入： {-1.5, 0.0, 0.5, 2.0}->{0.0, 0.0, 0.5, 2.0}


TEST_P(AutogradReluTest, Forward) {
    auto input = MakeTensor({4}, {-1.5f, 0.0f, 0.5f, 2.0f}, GetDevice());

    auto relu_fn = std::make_shared<autograd::Relu>();
    auto result = relu_fn->Apply({input});

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{4}));
    test::ExpectTensorNear(result[0], std::vector<float>{0.0f, 0.0f, 0.5f, 2.0f}, 1e-6f);

}


TEST_P(AutogradReluTest, Backward) {
    auto input = MakeTensor({4}, {-1.5f, 0.0f, 0.5f, 2.0f}, GetDevice());

    auto relu_fn = std::make_shared<autograd::Relu>();
    auto result = relu_fn->Apply({input});
    EXPECT_EQ(result.size(), 1);

    auto grad_output = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    grad_output->Fill(1.0f);
    auto grads = relu_fn->Backward({grad_output});

    ASSERT_EQ(grads.size(), 1);
    test::ExpectTensorNear(grads[0], std::vector<float>{0.0f, 0.0f, 1.0f, 1.0f}, 1e-6f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradReluTest);
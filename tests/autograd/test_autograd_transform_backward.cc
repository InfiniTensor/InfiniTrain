#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradTransformBackwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradTransformBackwardTest, TransposeBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto transpose_fn = std::make_shared<autograd::Transpose>(0, 1);
    auto result = transpose_fn->Apply({a});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{3, 2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = transpose_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

INFINI_TRAIN_REGISTER_TEST(AutogradTransformBackwardTest);

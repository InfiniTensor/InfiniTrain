#include <gtest/gtest.h>

#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/transform.h"
#include "test_utils.h"

using namespace infini_train;

class AutogradTransformBackwardTest : public infini_train::test::AutogradTestBase {};

TEST_F(AutogradTransformBackwardTest, TransposeBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto transpose_fn = std::make_shared<autograd::Transpose>(0, 1);
    auto result = transpose_fn->Apply({a});
    auto grad = createTensor({3, 2}, 1.0f);
    auto grad_inputs = transpose_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 1);
}

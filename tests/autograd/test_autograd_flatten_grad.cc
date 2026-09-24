#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

// Gradient contract of Tensor::Flatten: it is a pure reshape (via the no-op view),
// so d(loss)/d(x) must be the reshape-back of the downstream gradient.
// The composition x -> flatten -> x*x -> sum makes the expected grad = 2*x.
class AutogradFlattenGradTest : public infini_train::test::InfiniTrainTest {};

namespace {
void CheckFlattenGrad(const Device &device, const std::vector<int64_t> &input_dims, int64_t start, int64_t end,
                      const std::vector<int64_t> &expected_flat_dims) {
    const Device host = Device();

    std::vector<float> values;
    for (size_t idx = 0; idx < static_cast<size_t>(
                             std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<int64_t>()));
         ++idx) {
        values.push_back(((idx % 17) - 8) / 4.0f);
    }
    auto x = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, device, true);
    auto storage = std::make_shared<Tensor>(values.data(), input_dims, DataType::kFLOAT32, device);
    x->CopyFrom(*storage);

    auto flat = x->Flatten(start, end);
    ASSERT_EQ(flat->Dims(), expected_flat_dims);

    auto sq = flat->Mul(flat);
    auto reduced = sq;
    while (reduced->Dims().size() > 0) { reduced = reduced->Sum(0); }
    reduced->Backward();

    const auto &grad = x->grad();
    ASSERT_NE(grad, nullptr);
    ASSERT_EQ(grad->Dims(), input_dims);
    const auto grad_cpu = grad->To(host);
    const float *grad_data = static_cast<const float *>(grad_cpu.DataPtr());
    for (size_t idx = 0; idx < values.size(); ++idx) {
        EXPECT_FLOAT_EQ(grad_data[idx], 2.0f * values[idx]) << "gradient mismatch at flat index " << idx;
    }
}
} // namespace

// 4-D conv-output layout: (N, C, H, W) flattening the channel dim with start=1,
// the same call pattern as the MNIST demo.
TEST_P(AutogradFlattenGradTest, FlattenGradFourDConvLayout) {
    CheckFlattenGrad(GetDevice(), {2, 1, 4, 4}, 1, -1, {2, 16});
}

TEST_P(AutogradFlattenGradTest, FlattenGradThreeDStartEnd) { CheckFlattenGrad(GetDevice(), {2, 3, 4}, 0, 1, {6, 4}); }

TEST_P(AutogradFlattenGradTest, FlattenGradTwoDStartEnd) { CheckFlattenGrad(GetDevice(), {3, 4}, 0, 1, {12}); }

TEST_P(AutogradFlattenGradTest, FlattenGradOneD) { CheckFlattenGrad(GetDevice(), {5}, 0, 0, {5}); }

INFINI_TRAIN_REGISTER_TEST(AutogradFlattenGradTest);

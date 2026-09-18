#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/kernels/common/conv.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradConvIm2colTest : public infini_train::test::InfiniTrainTest {};

namespace {

std::shared_ptr<Tensor> CallIm2col(const std::shared_ptr<Tensor> &input, const kernels::Conv2dMeta &meta,
                                   Device::DeviceType device_type) {
    return Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device_type, "Im2colForward"}, input, meta);
}

} // namespace

TEST_P(AutogradConvIm2colTest, Im2colBasic) {
    ONLY_CUDA();
    // input 3x3 = 1..9, kernel 2x2, stride 1, padding 0 -> [N=1, K=4, P=4],
    // row k = (kh * Kw + kw), column p = (oh * Wout + ow).
    const float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto input
        = std::make_shared<Tensor>(input_values, std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32, GetDevice());
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice());
    weight->Fill(0.0f);
    const auto meta = kernels::MakeConv2dMeta(input, weight, /*stride=*/1, /*padding=*/0);
    auto columns = CallIm2col(input, meta, GetDevice().type());
    ASSERT_NE(columns, nullptr);
    EXPECT_EQ(columns->Dims(), (std::vector<int64_t>{1, 4, 4}));
    test::ExpectTensorFloatEqual(
        columns, {1.0f, 2.0f, 4.0f, 5.0f, 2.0f, 3.0f, 5.0f, 6.0f, 4.0f, 5.0f, 7.0f, 8.0f, 5.0f, 6.0f, 8.0f, 9.0f});
}

TEST_P(AutogradConvIm2colTest, Im2colStridePadding) {
    ONLY_CUDA();
    // input 4x4 = 1..16, kernel 3x3, stride 2, padding 1 -> [N=1, K=9, P=4].
    std::vector<float> input_values(16);
    for (int i = 0; i < 16; ++i) { input_values[i] = static_cast<float>(i + 1); }
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 4, 4}, DataType::kFLOAT32,
                                          GetDevice());
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32, GetDevice());
    weight->Fill(0.0f);
    const auto meta = kernels::MakeConv2dMeta(input, weight, /*stride=*/2, /*padding=*/1);
    auto columns = CallIm2col(input, meta, GetDevice().type());
    ASSERT_NE(columns, nullptr);
    EXPECT_EQ(columns->Dims(), (std::vector<int64_t>{1, 9, 4}));
    test::ExpectTensorFloatEqual(columns,
                                 {0.0f, 0.0f, 0.0f, 6.0f,  0.0f, 0.0f, 5.0f,  7.0f,  0.0f, 0.0f, 6.0f,  8.0f,
                                  0.0f, 2.0f, 0.0f, 10.0f, 1.0f, 3.0f, 9.0f,  11.0f, 2.0f, 4.0f, 10.0f, 12.0f,
                                  0.0f, 6.0f, 0.0f, 14.0f, 5.0f, 7.0f, 13.0f, 15.0f, 6.0f, 8.0f, 14.0f, 16.0f});
}

INFINI_TRAIN_REGISTER_TEST(AutogradConvIm2colTest);

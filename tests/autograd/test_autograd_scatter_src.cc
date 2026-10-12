#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradScatterSrcTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradScatterSrcTest, ScatterForwardAndBackward) {
    if (GetParam() != Device::DeviceType::kDCU) { GTEST_SKIP() << "DCU-only multimodal scatter"; }
    std::vector<float> text_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto text = std::make_shared<Tensor>(text_values.data(), std::vector<int64_t>{1, 4, 2}, DataType::kFLOAT32,
                                         GetDevice())
                    ->RequiresGrad();
    std::vector<float> vision_values{10.0f, 20.0f, 30.0f, 40.0f};
    auto vision = std::make_shared<Tensor>(vision_values.data(), std::vector<int64_t>{1, 2, 2}, DataType::kFLOAT32,
                                           GetDevice())
                      ->RequiresGrad();

    auto indices_cpu = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2}, DataType::kINT64);
    std::vector<int64_t> indices_values{1, 1, 3, 3};
    std::copy(indices_values.begin(), indices_values.end(), static_cast<int64_t *>(indices_cpu->DataPtr()));
    auto indices = std::make_shared<Tensor>(indices_cpu->To(GetDevice()));

    auto output = nn::function::Scatter(text, 1, indices, vision);
    test::ExpectTensorNear(output, std::vector<float>{1.0f, 2.0f, 10.0f, 20.0f, 5.0f, 6.0f, 30.0f, 40.0f},
                           1e-6f);

    output->Sum(2)->Sum(1)->Sum(0)->Backward();
    test::ExpectTensorNear(text->grad(), std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f},
                           1e-6f);
    test::ExpectTensorNear(vision->grad(), std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, 1e-6f);
}

TEST_P(AutogradScatterSrcTest, ScatterAllowsSmallerNonScatterDimensions) {
    if (GetParam() != Device::DeviceType::kDCU) { GTEST_SKIP() << "DCU-only multimodal scatter"; }
    std::vector<float> input_values(24);
    for (size_t i = 0; i < input_values.size(); ++i) { input_values[i] = static_cast<float>(i + 1); }
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32,
                                         GetDevice())
                     ->RequiresGrad();
    std::vector<float> src_values{101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f};
    auto src = std::make_shared<Tensor>(src_values.data(), std::vector<int64_t>{1, 2, 3}, DataType::kFLOAT32,
                                       GetDevice())
                   ->RequiresGrad();

    auto indices_cpu = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 3}, DataType::kINT64);
    std::vector<int64_t> indices_values{0, 0, 0, 1, 1, 1};
    std::copy(indices_values.begin(), indices_values.end(), static_cast<int64_t *>(indices_cpu->DataPtr()));
    auto indices = std::make_shared<Tensor>(indices_cpu->To(GetDevice()));

    auto output = nn::function::Scatter(input, 1, indices, src);
    test::ExpectTensorNear(output,
                           std::vector<float>{101.0f, 102.0f, 103.0f, 4.0f, 104.0f, 105.0f, 106.0f, 8.0f,
                                              9.0f,   10.0f,  11.0f,  12.0f, 13.0f,  14.0f,  15.0f,  16.0f,
                                              17.0f,  18.0f,  19.0f,  20.0f, 21.0f,  22.0f,  23.0f,  24.0f},
                           1e-6f);

    output->Sum(2)->Sum(1)->Sum(0)->Backward();
    test::ExpectTensorNear(input->grad(),
                           std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                              1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                              1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                           1e-6f);
    test::ExpectTensorNear(src->grad(), std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 1e-6f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradScatterSrcTest);

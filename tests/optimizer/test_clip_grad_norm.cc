#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class ClipGradNormTest : public infini_train::test::InfiniTrainTest {};

static std::shared_ptr<Tensor> MakeTensor(Device device, const std::vector<float> &values) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(values.size())},
                                           DataType::kFLOAT32, device);
    if (device.IsCPU()) {
        std::copy(values.begin(), values.end(), static_cast<float *>(tensor->DataPtr()));
    } else {
        auto cpu = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(values.size())},
                                            DataType::kFLOAT32, Device());
        std::copy(values.begin(), values.end(), static_cast<float *>(cpu->DataPtr()));
        tensor->CopyFrom(*cpu);
    }
    return tensor;
}

static std::shared_ptr<Tensor> MakeTensor(Device device, DataType dtype, const std::vector<float> &values) {
    auto fp32 = std::make_shared<Tensor>(
        values.data(), std::vector<int64_t>{static_cast<int64_t>(values.size())}, DataType::kFLOAT32, device);
    auto converted = fp32->To(dtype);
    auto result = std::make_shared<Tensor>(converted.Dims(), dtype, device);
    result->CopyFrom(converted);
    return result;
}

static float ScalarCPU(const std::shared_ptr<Tensor> &value) {
    auto cpu = value->To(Device());
    return *static_cast<const float *>(cpu.DataPtr());
}

TEST_P(ClipGradNormTest, L2ClipsInPlace) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    auto grad = MakeTensor(GetDevice(), {3.0f, 4.0f});
    param->set_grad(grad);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.1f);

    auto total_norm = optimizer->ClipGradNorm({param}, 2.0f, 2.0f);
    EXPECT_NEAR(ScalarCPU(total_norm), 5.0f, 1e-5f);
    auto clipped = grad->To(Device());
    const float *values = static_cast<const float *>(clipped.DataPtr());
    EXPECT_NEAR(values[0], 1.2f, 1e-5f);
    EXPECT_NEAR(values[1], 1.6f, 1e-5f);
}

TEST_P(ClipGradNormTest, SupportsL1AndInfinity) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    auto grad = MakeTensor(GetDevice(), {-2.0f, 1.0f, 3.0f});
    param->set_grad(grad);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.1f);

    auto l1 = optimizer->ClipGradNorm({param}, 3.0f, 1.0f);
    EXPECT_NEAR(ScalarCPU(l1), 6.0f, 1e-5f);
    auto after_l1 = grad->To(Device());
    const float *l1_values = static_cast<const float *>(after_l1.DataPtr());
    EXPECT_NEAR(l1_values[0], -1.0f, 1e-5f);
    EXPECT_NEAR(l1_values[1], 0.5f, 1e-5f);
    EXPECT_NEAR(l1_values[2], 1.5f, 1e-5f);

    auto inf_grad = MakeTensor(GetDevice(), {-2.0f, 1.0f, 3.0f});
    param->set_grad(inf_grad);
    auto inf = optimizer->ClipGradNorm({param}, 1.0f, std::numeric_limits<float>::infinity());
    EXPECT_NEAR(ScalarCPU(inf), 3.0f, 1e-5f);
    auto after_inf = inf_grad->To(Device());
    const float *inf_values = static_cast<const float *>(after_inf.DataPtr());
    EXPECT_NEAR(inf_values[0], -2.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(inf_values[2], 1.0f, 1e-5f);
}

TEST_P(ClipGradNormTest, IgnoresMissingAndDuplicateGradients) {
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    first->set_grad(MakeTensor(GetDevice(), {3.0f}));
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{first, second}, 0.1f);

    auto total_norm = optimizer->ClipGradNorm({first, first, second}, 1.0f, 2.0f);
    EXPECT_NEAR(ScalarCPU(total_norm), 3.0f, 1e-5f);
    auto clipped = first->grad()->To(Device());
    EXPECT_NEAR(*static_cast<const float *>(clipped.DataPtr()), 1.0f, 1e-5f);

    auto empty = optimizer->ClipGradNorm({second}, 1.0f, 2.0f);
    EXPECT_FLOAT_EQ(ScalarCPU(empty), 0.0f);
}

INFINI_TRAIN_REGISTER_TEST(ClipGradNormTest);

TEST_P(ClipGradNormTest, NonIntegerPAndZeroMaxNorm) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    auto grad = MakeTensor(GetDevice(), {2.0f, 2.0f});
    param->set_grad(grad);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.1f);

    const float expected = std::pow(2.0f * std::pow(2.0f, 3.5f), 1.0f / 3.5f);
    auto total_norm = optimizer->ClipGradNorm({param}, 0.0f, 3.5f);
    EXPECT_NEAR(ScalarCPU(total_norm), expected, 1e-5f);
    auto clipped = grad->To(Device());
    const float *values = static_cast<const float *>(clipped.DataPtr());
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 0.0f);
}

TEST_P(ClipGradNormTest, SupportsZeroNegativeAndNegativeInfinityNorms) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    auto grad = MakeTensor(GetDevice(), {-2.0f, 0.0f, 4.0f});
    param->set_grad(grad);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.1f);

    auto zero = optimizer->ClipGradNorm({param}, 1.0f, 0.0f);
    // PyTorch computes the 0-norm of each gradient tensor and then the
    // 0-norm of that list, so one non-empty gradient tensor contributes one.
    EXPECT_FLOAT_EQ(ScalarCPU(zero), 1.0f);

    auto neg_grad = MakeTensor(GetDevice(), {-2.0f, 0.0f, 4.0f});
    param->set_grad(neg_grad);
    auto negative = optimizer->ClipGradNorm({param}, 1.0f, -1.0f);
    EXPECT_TRUE(std::isfinite(ScalarCPU(negative)) || ScalarCPU(negative) == 0.0f);

    auto min_grad = MakeTensor(GetDevice(), {-2.0f, 0.5f, 4.0f});
    param->set_grad(min_grad);
    auto neg_inf = optimizer->ClipGradNorm({param}, 0.25f, -std::numeric_limits<float>::infinity());
    EXPECT_NEAR(ScalarCPU(neg_inf), 0.5f, 1e-5f);
}

TEST_P(ClipGradNormTest, ForeachTrueUsesMultiTensorSemantics) {
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    first->set_grad(MakeTensor(GetDevice(), {3.0f, 4.0f}));
    second->set_grad(MakeTensor(GetDevice(), {0.0f, 12.0f}));
    auto optimizer = std::make_shared<optimizers::SGD>(
        std::vector<std::shared_ptr<Tensor>>{first, second}, 0.1f);
    auto total = optimizer->ClipGradNorm({first, second}, 6.5f, 2.0f, false, true);
    EXPECT_NEAR(ScalarCPU(total), 13.0f, 1e-5f);
    auto clipped = second->grad()->To(Device());
    EXPECT_NEAR(static_cast<const float *>(clipped.DataPtr())[1], 6.0f, 1e-4f);
}

TEST_P(ClipGradNormTest, AccumulatesReducedPrecisionGradientsInFloat32) {
    auto fp16_param = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT16, GetDevice());
    auto bf16_param = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kBFLOAT16, GetDevice());
    fp16_param->set_grad(MakeTensor(GetDevice(), DataType::kFLOAT16, {3.0f, 4.0f}));
    bf16_param->set_grad(MakeTensor(GetDevice(), DataType::kBFLOAT16, {0.0f, 12.0f}));
    auto optimizer = std::make_shared<optimizers::SGD>(
        std::vector<std::shared_ptr<Tensor>>{fp16_param, bf16_param}, 0.1f);

    auto total = optimizer->ClipGradNorm({fp16_param, bf16_param}, 6.5f, 2.0f, false, true);
    EXPECT_NEAR(ScalarCPU(total), 13.0f, 2e-2f);
    auto fp16_cpu = fp16_param->grad()->To(Device());
    auto bf16_cpu = bf16_param->grad()->To(Device());
    EXPECT_NEAR(static_cast<float>(static_cast<const FP16 *>(fp16_cpu.DataPtr())[0]), 1.5f, 8e-2f);
    EXPECT_NEAR(static_cast<float>(static_cast<const BF16 *>(bf16_cpu.DataPtr())[1]), 6.0f, 1.5e-1f);
}

TEST_P(ClipGradNormTest, ErrorOnNonFiniteBeforeScaling) {
    ONLY_CPU();
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    auto grad = MakeTensor(GetDevice(), {std::numeric_limits<float>::quiet_NaN(), 1.0f});
    param->set_grad(grad);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.1f);
    EXPECT_DEATH(optimizer->ClipGradNorm({param}, 1.0f, 2.0f, true), "non-finite");

    auto neg_inf_grad = MakeTensor(GetDevice(), {std::numeric_limits<float>::quiet_NaN(), 1.0f});
    param->set_grad(neg_inf_grad);
    EXPECT_DEATH(optimizer->ClipGradNorm({param}, 1.0f, -std::numeric_limits<float>::infinity(), true), "non-finite");
}

TEST_P(ClipGradNormTest, EmptyGradientIsIgnoredForNegativeInfinity) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{0}, DataType::kFLOAT32, GetDevice());
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{0}, DataType::kFLOAT32, GetDevice());
    param->set_grad(grad);
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{param}, 0.1f);
    auto total = optimizer->ClipGradNorm({param}, 1.0f, -std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(ScalarCPU(total), 0.0f);
}

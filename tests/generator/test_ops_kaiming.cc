#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorKaimingTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::vector<float> ReadCpuFloats(const std::shared_ptr<Tensor> &t) {
    Tensor cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu.NumElements());
    std::memcpy(out.data(), cpu.DataPtr(), out.size() * sizeof(float));
    return out;
}
} // namespace

TEST_P(GeneratorKaimingTest, SameSeedSameWeights) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA random kernels are Phase 2";
    }
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{16, 16}, DataType::kFLOAT32, GetDevice());
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{16, 16}, DataType::kFLOAT32, GetDevice());

    Generator g1(GetDevice());
    Generator g2(GetDevice());
    g1.ManualSeed(2026);
    g2.ManualSeed(2026);
    nn::init::KaimingUniform(t1, /*a=*/0.0f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kLeakyReLU, g1);
    nn::init::KaimingUniform(t2, /*a=*/0.0f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kLeakyReLU, g2);
    EXPECT_EQ(ReadCpuFloats(t1), ReadCpuFloats(t2));
}

TEST_P(GeneratorKaimingTest, FanInBoundsRespected) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA random kernels are Phase 2";
    }
    // weight: [out=8, in=32] -> fan_in = 32, gain(LeakyReLU,a=0) = sqrt(2/(1+0.01^2))
    // std = gain / sqrt(fan_in); bound = sqrt(3) * std
    const int64_t out_dim = 8, in_dim = 32;
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{out_dim, in_dim}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(123);
    nn::init::KaimingUniform(t, /*a=*/0.0f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kLeakyReLU, g);
    const float gain = std::sqrt(2.0f / (1.0f + 0.01f * 0.01f));
    const float bound = std::sqrt(3.0f) * gain / std::sqrt(static_cast<float>(in_dim));
    for (float v : ReadCpuFloats(t)) {
        EXPECT_GE(v, -bound);
        EXPECT_LT(v, bound);
    }
}

INFINI_TRAIN_REGISTER_TEST(GeneratorKaimingTest);

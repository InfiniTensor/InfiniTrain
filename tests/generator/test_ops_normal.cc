#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorNormalOpTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::vector<float> ReadCpuFloats(const std::shared_ptr<Tensor> &t) {
    Tensor cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu.NumElements());
    std::memcpy(out.data(), cpu.DataPtr(), out.size() * sizeof(float));
    return out;
}
} // namespace

TEST_P(GeneratorNormalOpTest, SameSeedSameOutput) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA NormalRandom is Phase 2";
    }
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{16}, DataType::kFLOAT32, GetDevice());
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{16}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(11);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "NormalRandom"}, a, 0.0f, 1.0f, g.impl().get());
    g.ManualSeed(11);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "NormalRandom"}, b, 0.0f, 1.0f, g.impl().get());
    EXPECT_EQ(ReadCpuFloats(a), ReadCpuFloats(b));
}

TEST_P(GeneratorNormalOpTest, MeanCloseToTarget) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA NormalRandom is Phase 2";
    }
    constexpr int kN = 16384;
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{kN}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(31);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "NormalRandom"}, t, 5.0f, 2.0f, g.impl().get());
    auto v = ReadCpuFloats(t);
    double sum = 0;
    for (float x : v) { sum += x; }
    const double mean = sum / kN;
    EXPECT_NEAR(mean, 5.0, 0.1);
}

INFINI_TRAIN_REGISTER_TEST(GeneratorNormalOpTest);

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

class GeneratorUniformOpTest : public infini_train::test::InfiniTrainTest {};

namespace {
std::vector<float> ReadCpuFloats(const std::shared_ptr<Tensor> &t) {
    Tensor cpu = t->To(Device(Device::DeviceType::kCPU, 0));
    std::vector<float> out(cpu.NumElements());
    std::memcpy(out.data(), cpu.DataPtr(), out.size() * sizeof(float));
    return out;
}
} // namespace

TEST_P(GeneratorUniformOpTest, SameSeedSameOutput) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA UniformRandom is Phase 2";
    }
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());

    g.ManualSeed(42);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t1, 0.0f, 1.0f, g.impl().get());
    g.ManualSeed(42);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t2, 0.0f, 1.0f, g.impl().get());

    EXPECT_EQ(ReadCpuFloats(t1), ReadCpuFloats(t2));
}

TEST_P(GeneratorUniformOpTest, ConsecutiveCallsAdvanceState) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA UniformRandom is Phase 2";
    }
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(7);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t1, 0.0f, 1.0f, g.impl().get());
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t2, 0.0f, 1.0f, g.impl().get());
    EXPECT_NE(ReadCpuFloats(t1), ReadCpuFloats(t2));
}

TEST_P(GeneratorUniformOpTest, OutputsWithinRange) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA UniformRandom is Phase 2";
    }
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{1024}, DataType::kFLOAT32, GetDevice());
    Generator g(GetDevice());
    g.ManualSeed(2026);
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, t, -2.0f, 5.0f, g.impl().get());
    for (float v : ReadCpuFloats(t)) {
        EXPECT_GE(v, -2.0f);
        EXPECT_LT(v, 5.0f);
    }
}

INFINI_TRAIN_REGISTER_TEST(GeneratorUniformOpTest);

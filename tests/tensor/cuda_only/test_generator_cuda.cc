#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
std::vector<float> CopyToCPUData(const std::shared_ptr<Tensor> &tensor) {
    auto cpu_tensor = tensor->To(Device());
    auto *impl = core::GetDeviceGuardImpl(tensor->GetDevice().type());
    impl->SynchronizeDevice(tensor->GetDevice());

    if (cpu_tensor.NumElements() == 0) {
        return {};
    }
    const auto *data = static_cast<const float *>(cpu_tensor.DataPtr());
    return std::vector<float>(data, data + cpu_tensor.NumElements());
}

uint64_t CUDASemanticOffset(const std::vector<uint8_t> &state) {
    std::istringstream iss(std::string(state.begin(), state.end()));
    std::string line;
    CHECK(std::getline(iss, line));
    CHECK(std::getline(iss, line));
    CHECK(std::getline(iss, line));
    CHECK(std::getline(iss, line));
    return std::stoull(line);
}
} // namespace

class CUDAGeneratorTensorTest : public ::testing::Test {};

TEST_F(CUDAGeneratorTensorTest, DefaultCUDAGeneratorControlsRand) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    ManualSeedAll(123);
    auto a = CopyToCPUData(nn::function::Rand({2, 3}, cuda_device));

    ManualSeedAll(123);
    auto b = CopyToCPUData(nn::function::Rand({2, 3}, cuda_device));

    ManualSeedAll(456);
    auto c = CopyToCPUData(nn::function::Rand({2, 3}, cuda_device));

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST_F(CUDAGeneratorTensorTest, DefaultCUDAGeneratorControlsRandn) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    ManualSeedAll(123);
    auto a = CopyToCPUData(nn::function::Randn({2, 3}, cuda_device));

    ManualSeedAll(123);
    auto b = CopyToCPUData(nn::function::Randn({2, 3}, cuda_device));

    ManualSeedAll(456);
    auto c = CopyToCPUData(nn::function::Randn({2, 3}, cuda_device));

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST_F(CUDAGeneratorTensorTest, KaimingInitializationUsesDefaultGeneratorReproducibly) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    ManualSeedAll(314159);
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{8, 16}, DataType::kFLOAT32, cuda_device);
    auto a = CopyToCPUData(nn::init::KaimingUniform(first));

    ManualSeedAll(314159);
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{8, 16}, DataType::kFLOAT32, cuda_device);
    auto b = CopyToCPUData(nn::init::KaimingUniform(second));

    EXPECT_EQ(a, b);
}

TEST_F(CUDAGeneratorTensorTest, DefaultCUDAGeneratorAdvancesAcrossRandCalls) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    ManualSeedAll(123);
    auto a = CopyToCPUData(nn::function::Rand({4}, cuda_device));
    auto b = CopyToCPUData(nn::function::Rand({4}, cuda_device));

    EXPECT_NE(a, b);
}

TEST_F(CUDAGeneratorTensorTest, ExplicitCUDAGeneratorControlsRand) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    auto g1 = MakeCUDAGenerator(0, 123);
    auto a = CopyToCPUData(nn::function::Rand({2, 3}, cuda_device, g1));

    auto g2 = MakeCUDAGenerator(0, 123);
    auto b = CopyToCPUData(nn::function::Rand({2, 3}, cuda_device, g2));

    auto g3 = MakeCUDAGenerator(0, 456);
    auto c = CopyToCPUData(nn::function::Rand({2, 3}, cuda_device, g3));

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST_F(CUDAGeneratorTensorTest, CUDAGeneratorStateRestoreReplaysRand) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    auto generator = MakeCUDAGenerator(0, 123);
    (void)CopyToCPUData(nn::function::Rand({4}, cuda_device, generator));
    auto state = generator->GetState();

    auto b = CopyToCPUData(nn::function::Rand({4}, cuda_device, generator));
    generator->SetState(state);
    auto b2 = CopyToCPUData(nn::function::Rand({4}, cuda_device, generator));

    EXPECT_EQ(b, b2);
}

TEST_F(CUDAGeneratorTensorTest, ExplicitCUDAGeneratorDoesNotAdvanceDefaultGenerator) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    ManualSeedAll(777);
    auto default_generator = GetDefaultCUDAGenerator(0);
    auto before = default_generator->GetState();

    auto explicit_generator = MakeCUDAGenerator(0, 123);
    (void)CopyToCPUData(nn::function::Rand({8}, cuda_device, explicit_generator));

    auto after = default_generator->GetState();
    EXPECT_EQ(before, after);
}

TEST_F(CUDAGeneratorTensorTest, TensorRandAdvancesCUDASemanticOffset) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    auto generator = MakeCUDAGenerator(0, 123);
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), 0U);

    (void)CopyToCPUData(nn::function::Rand({2, 3}, cuda_device, generator));
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), 6U);

    (void)CopyToCPUData(nn::function::Randn({4}, cuda_device, generator));
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), 14U);
}

TEST_F(CUDAGeneratorTensorTest, SplitCallsConsumeOneContinuousRandomStream) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    auto split_generator = MakeCUDAGenerator(0, 123);
    auto first = CopyToCPUData(nn::function::Rand({13}, cuda_device, split_generator));
    auto second = CopyToCPUData(nn::function::Rand({19}, cuda_device, split_generator));
    first.insert(first.end(), second.begin(), second.end());

    auto combined_generator = MakeCUDAGenerator(0, 123);
    auto combined = CopyToCPUData(nn::function::Rand({32}, cuda_device, combined_generator));
    EXPECT_EQ(first, combined);
    EXPECT_EQ(CUDASemanticOffset(split_generator->GetState()), 32U);
    EXPECT_EQ(CUDASemanticOffset(combined_generator->GetState()), 32U);

    auto split_normal_generator = MakeCUDAGenerator(0, 456);
    auto normal_first = CopyToCPUData(nn::function::Randn({7}, cuda_device, split_normal_generator));
    auto normal_second = CopyToCPUData(nn::function::Randn({11}, cuda_device, split_normal_generator));
    normal_first.insert(normal_first.end(), normal_second.begin(), normal_second.end());

    auto combined_normal_generator = MakeCUDAGenerator(0, 456);
    auto combined_normal = CopyToCPUData(nn::function::Randn({18}, cuda_device, combined_normal_generator));
    EXPECT_EQ(normal_first, combined_normal);
    EXPECT_EQ(CUDASemanticOffset(split_normal_generator->GetState()), 36U);
    EXPECT_EQ(CUDASemanticOffset(combined_normal_generator->GetState()), 36U);
}

TEST_F(CUDAGeneratorTensorTest, HostAndKernelConsumptionShareOneOffsetStream) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);

    auto mixed_uniform_generator = MakeCUDAGenerator(0, 123);
    std::vector<float> host_uniform(13);
    mixed_uniform_generator->FillUniform(host_uniform, 0.0f, 1.0f);
    auto after_host_uniform = CopyToCPUData(nn::function::Rand({19}, cuda_device, mixed_uniform_generator));

    auto reference_uniform_generator = MakeCUDAGenerator(0, 123);
    (void)reference_uniform_generator->ReserveRandomOffset(13);
    auto reference_uniform = CopyToCPUData(nn::function::Rand({19}, cuda_device, reference_uniform_generator));
    EXPECT_EQ(after_host_uniform, reference_uniform);
    EXPECT_EQ(CUDASemanticOffset(mixed_uniform_generator->GetState()), 32U);

    auto mixed_normal_generator = MakeCUDAGenerator(0, 456);
    std::vector<float> host_normal(7);
    mixed_normal_generator->FillNormal(host_normal, 0.0f, 1.0f);
    auto after_host_normal = CopyToCPUData(nn::function::Randn({11}, cuda_device, mixed_normal_generator));

    auto reference_normal_generator = MakeCUDAGenerator(0, 456);
    (void)reference_normal_generator->ReserveRandomOffset(14);
    auto reference_normal = CopyToCPUData(nn::function::Randn({11}, cuda_device, reference_normal_generator));
    EXPECT_EQ(after_host_normal, reference_normal);
    EXPECT_EQ(CUDASemanticOffset(mixed_normal_generator->GetState()), 36U);
}

TEST_F(CUDAGeneratorTensorTest, ZeroElementRandomCallsDoNotAdvanceState) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);
    auto generator = MakeCUDAGenerator(0, 123);
    auto before = generator->GetState();

    EXPECT_TRUE(CopyToCPUData(nn::function::Rand({0}, cuda_device, generator)).empty());
    EXPECT_TRUE(CopyToCPUData(nn::function::Randn({0}, cuda_device, generator)).empty());
    EXPECT_EQ(before, generator->GetState());
}

TEST_F(CUDAGeneratorTensorTest, ZeroStdNormalReturnsMeanAndAdvancesState) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);
    constexpr int64_t kNumElements = 17;
    auto generator = MakeCUDAGenerator(0, 123);
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{kNumElements}, DataType::kFLOAT32, cuda_device);

    auto values = CopyToCPUData(nn::init::Normal(tensor, 2.5f, 0.0f, generator));

    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value == 2.5f; }));
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), static_cast<uint64_t>(kNumElements) * 2);
}

TEST_F(CUDAGeneratorTensorTest, EqualUniformBoundsReturnConstantAndAdvanceState) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);
    constexpr int64_t kNumElements = 17;
    auto generator = MakeCUDAGenerator(0, 123);
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{kNumElements}, DataType::kFLOAT32, cuda_device);

    auto values = CopyToCPUData(nn::init::Uniform(tensor, 2.5f, 2.5f, generator));

    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value == 2.5f; }));
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), static_cast<uint64_t>(kNumElements));
}

TEST_F(CUDAGeneratorTensorTest, RandomKernelsHaveSaneRangeAndMoments) {
    REQUIRE_MIN_DEVICES(1);
    Device cuda_device(Device::DeviceType::kCUDA, 0);
    constexpr int64_t kSampleCount = 65537;

    auto uniform = CopyToCPUData(nn::function::Rand({kSampleCount}, cuda_device, MakeCUDAGenerator(0, 2026)));
    EXPECT_TRUE(std::all_of(uniform.begin(), uniform.end(), [](float value) { return value >= 0.0f && value < 1.0f; }));
    const double uniform_mean = std::accumulate(uniform.begin(), uniform.end(), 0.0) / uniform.size();
    EXPECT_NEAR(uniform_mean, 0.5, 0.02);

    auto normal = CopyToCPUData(nn::function::Randn({kSampleCount}, cuda_device, MakeCUDAGenerator(0, 2026)));
    EXPECT_TRUE(std::all_of(normal.begin(), normal.end(), [](float value) { return std::isfinite(value); }));
    const double normal_mean = std::accumulate(normal.begin(), normal.end(), 0.0) / normal.size();
    const double squared_sum = std::inner_product(normal.begin(), normal.end(), normal.begin(), 0.0);
    const double normal_variance = squared_sum / normal.size() - normal_mean * normal_mean;
    EXPECT_NEAR(normal_mean, 0.0, 0.05);
    EXPECT_NEAR(normal_variance, 1.0, 0.08);
}

TEST_F(CUDAGeneratorTensorTest, ExplicitGeneratorMayTargetDifferentCUDAIndex) {
    REQUIRE_MIN_DEVICES(2);
    Device cuda1(Device::DeviceType::kCUDA, 1);
    constexpr int64_t kNumElements = 32;

    auto generator = MakeCUDAGenerator(0, 123);
    auto values = CopyToCPUData(nn::function::Rand({kNumElements}, cuda1, generator));

    auto replay_generator = MakeCUDAGenerator(0, 123);
    auto replay = CopyToCPUData(nn::function::Rand({kNumElements}, cuda1, replay_generator));

    EXPECT_EQ(values, replay);
    EXPECT_EQ(generator->GetDevice().index(), 0);
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), static_cast<uint64_t>(kNumElements));
}

TEST_F(CUDAGeneratorTensorTest, DefaultGeneratorsAreIndependentAcrossPhysicalDevices) {
    REQUIRE_MIN_DEVICES(2);
    Device cuda0(Device::DeviceType::kCUDA, 0);
    Device cuda1(Device::DeviceType::kCUDA, 1);

    ManualSeedAll(2026);
    auto generator0 = GetDefaultCUDAGenerator(0);
    auto generator1 = GetDefaultCUDAGenerator(1);

    auto cuda1_before = generator1->GetState();
    auto a0 = CopyToCPUData(nn::function::Rand({32}, cuda0));
    EXPECT_EQ(cuda1_before, generator1->GetState());

    auto cuda0_after = generator0->GetState();
    auto a1 = CopyToCPUData(nn::function::Rand({32}, cuda1));
    EXPECT_EQ(cuda0_after, generator0->GetState());
    EXPECT_EQ(a0, a1);

    auto b0 = CopyToCPUData(nn::function::Rand({32}, cuda0));
    EXPECT_NE(a0, b0);
}

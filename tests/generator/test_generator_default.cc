// Default generator management and ManualSeed tests.

#include "infini_train/include/core/generator.h"
#include "infini_train/include/device.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "infini_train/include/core/cpu_generator.h"
#ifdef USE_CUDA
#include "infini_train/include/core/cuda_generator.h"
#endif

using namespace infini_train;

namespace {

Device CPUDevice() { return Device(Device::DeviceType::kCPU, 0); }

core::Generator MakeCPUGenerator(uint64_t seed) {
    return core::Generator(std::make_shared<core::CPUGeneratorImpl>(seed));
}

std::vector<uint64_t> DrawCPU(core::Generator &generator, int64_t count) {
    auto *impl = generator.Get<core::CPUGeneratorImpl>();
    std::lock_guard<std::mutex> lock(generator.Mutex());
    std::vector<uint64_t> values(count);
    auto &engine = impl->Engine();
    for (auto &value : values) { value = engine(); }
    return values;
}

} // namespace

TEST(GeneratorDefaultTest, DefaultGeneratorIsStable) {
    const auto &g1 = core::detail::DefaultCPUGenerator();
    const auto &g2 = core::detail::GetDefaultGenerator(CPUDevice());
    EXPECT_EQ(g1.UnsafeGetImpl(), g2.UnsafeGetImpl());
}

TEST(GeneratorDefaultTest, GetGeneratorOrDefaultPrefersExplicit) {
    auto explicit_gen = MakeCPUGenerator(123);
    auto *explicit_impl = core::GetGeneratorOrDefault<core::CPUGeneratorImpl>(explicit_gen, CPUDevice());
    EXPECT_EQ(explicit_impl, explicit_gen.UnsafeGetImpl());

    auto *default_impl = core::GetGeneratorOrDefault<core::CPUGeneratorImpl>(std::nullopt, CPUDevice());
    EXPECT_EQ(default_impl, core::detail::DefaultCPUGenerator().UnsafeGetImpl());
}

TEST(GeneratorDefaultTest, GlobalManualSeedResetsCPUDefault) {
    core::ManualSeed(555);
    const auto state = core::detail::DefaultCPUGenerator().GetState();
    auto first = core::detail::DefaultCPUGenerator();
    auto first_values = DrawCPU(first, 8);

    core::ManualSeed(555);
    auto second = core::detail::DefaultCPUGenerator();
    auto second_values = DrawCPU(second, 8);

    EXPECT_EQ(first_values, second_values);

    auto restored = MakeCPUGenerator(1);
    restored.SetState(state);
    EXPECT_EQ(restored.CurrentSeed(), 555u);
}

#ifdef USE_CUDA

TEST(GeneratorDefaultTest, FutureDefaultGeneratorUsesLastManualSeed) {
    core::ManualSeed(2468);
    const auto &cuda_gen = core::detail::DefaultCUDAGenerator(7);
    EXPECT_EQ(cuda_gen.CurrentSeed(), 2468u);
    EXPECT_EQ(cuda_gen.GetDevice(), Device(Device::DeviceType::kCUDA, 7));
}

TEST(GeneratorDefaultTest, ManualSeedResetsExistingCUDADefault) {
    core::ManualSeed(100);
    const auto &cuda5 = core::detail::DefaultCUDAGenerator(5);
    EXPECT_EQ(cuda5.CurrentSeed(), 100u);

    core::ManualSeed(200);
    EXPECT_EQ(cuda5.CurrentSeed(), 200u);
    EXPECT_EQ(cuda5.GetOffset(), 0u);
}

TEST(GeneratorDefaultTest, MultiDeviceIndependence) {
    core::ManualSeed(777);
    const auto &cuda3 = core::detail::DefaultCUDAGenerator(3);
    const auto &cuda4 = core::detail::DefaultCUDAGenerator(4);

    EXPECT_NE(cuda3.UnsafeGetImpl(), cuda4.UnsafeGetImpl());
    EXPECT_EQ(cuda3.GetDevice(), Device(Device::DeviceType::kCUDA, 3));
    EXPECT_EQ(cuda4.GetDevice(), Device(Device::DeviceType::kCUDA, 4));

    {
        auto *impl3 = cuda3.Get<core::CUDAGeneratorImpl>();
        std::lock_guard<std::mutex> lock(impl3->mutex_);
        impl3->PhiloxEngineInputs(42);
    }
    EXPECT_EQ(cuda3.GetOffset(), 42u);
    EXPECT_EQ(cuda4.GetOffset(), 0u);
}

#endif // USE_CUDA

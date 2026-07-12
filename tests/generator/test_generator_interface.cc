#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

namespace infini_train::test {
namespace {

constexpr uint64_t kSeed = 0x12345678ULL;

TEST(GeneratorInterfaceTest, DefaultConstructedGeneratorIsUndefined) {
    Generator generator;
    EXPECT_FALSE(generator.defined());
}

TEST(GeneratorInterfaceTest, CheckGeneratorRejectsUndefinedGenerator) {
    Generator generator;
    EXPECT_THROW(check_generator<core::cpu::CPUGeneratorImpl>(generator), std::invalid_argument);
}

TEST(GeneratorInterfaceTest, CpuStateIsAnOpaqueUint8CpuTensor) {
    Generator generator = core::cpu::createCPUGenerator(kSeed);
    auto state = generator.get_state();

    ASSERT_TRUE(state);
    EXPECT_TRUE(state->GetDevice().IsCPU());
    EXPECT_EQ(state->Dtype(), DataType::kUINT8);
    EXPECT_GT(state->SizeInBytes(), 0u);
}

TEST(GeneratorInterfaceTest, CpuStateRoundTripRestoresSeed) {
    Generator generator = core::cpu::createCPUGenerator(kSeed);
    auto state = generator.get_state();
    generator.set_current_seed(kSeed + 1);

    generator.set_state(*state);

    EXPECT_EQ(generator.current_seed(), kSeed);
}

TEST(GeneratorInterfaceTest, RejectsStateWithWrongDtype) {
    Generator generator = core::cpu::createCPUGenerator(kSeed);
    Tensor invalid_state(std::vector<int64_t>{64}, DataType::kFLOAT32,
                         Device(Device::DeviceType::kCPU, 0));

    EXPECT_DEATH(generator.set_state(invalid_state), "RNG state must be a UINT8 tensor");
}

TEST(GeneratorInterfaceTest, RejectsTruncatedCpuState) {
    Generator generator = core::cpu::createCPUGenerator(kSeed);
    Tensor invalid_state(std::vector<int64_t>{1}, DataType::kUINT8,
                         Device(Device::DeviceType::kCPU, 0));

    EXPECT_DEATH(generator.set_state(invalid_state), "CPU generator state is too small");
}

} // namespace
} // namespace infini_train::test

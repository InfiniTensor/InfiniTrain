#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

namespace infini_train::test {
namespace {

constexpr uint64_t kSeed = 0x12345678ULL;

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

} // namespace
} // namespace infini_train::test

#include <cstdint>
#include <memory>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
using infini_train::test::TensorBytes;

class GeneratorCpuTest : public ::testing::Test {};

TEST_F(GeneratorCpuTest, HighBitsOfCPUSeedAffectTheSequence) {
    const Device cpu;
    constexpr uint64_t low_seed = 17;
    constexpr uint64_t high_seed = low_seed + (uint64_t{1} << 32);
    EXPECT_NE(TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, cpu, CreateGenerator(cpu, low_seed))),
              TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, cpu, CreateGenerator(cpu, high_seed))));
}

TEST_F(GeneratorCpuTest, StateRestoresCachedNormalSequence) {
    const Device cpu;
    auto generator = CreateGenerator(cpu, 4567);
    nn::function::Randn({3}, DataType::kFLOAT32, cpu, generator);
    nn::function::Randn({3}, DataType::kFLOAT64, cpu, generator);
    const auto state_after_odd_normal = generator.get_state();
    const auto normal_float = TensorBytes(nn::function::Randn({127}, DataType::kFLOAT32, cpu, generator));
    const auto normal_double = TensorBytes(nn::function::Randn({127}, DataType::kFLOAT64, cpu, generator));
    const auto state_after_replay = TensorBytes(generator.get_state());

    generator.set_state(*state_after_odd_normal);
    EXPECT_EQ(normal_float, TensorBytes(nn::function::Randn({127}, DataType::kFLOAT32, cpu, generator)));
    EXPECT_EQ(normal_double, TensorBytes(nn::function::Randn({127}, DataType::kFLOAT64, cpu, generator)));
    EXPECT_EQ(state_after_replay, TensorBytes(generator.get_state()));
}

TEST_F(GeneratorCpuTest, ReseedingClearsBothNormalCaches) {
    const Device cpu;
    auto generator = CreateGenerator(cpu, 4567);
    nn::function::Randn({3}, DataType::kFLOAT32, cpu, generator);
    nn::function::Randn({3}, DataType::kFLOAT64, cpu, generator);

    generator.set_current_seed(5678);
    auto fresh = CreateGenerator(cpu, 5678);
    EXPECT_EQ(TensorBytes(generator.get_state()), TensorBytes(fresh.get_state()));
    EXPECT_EQ(TensorBytes(nn::function::Randn({127}, DataType::kFLOAT32, cpu, generator)),
              TensorBytes(nn::function::Randn({127}, DataType::kFLOAT32, cpu, fresh)));
    EXPECT_EQ(TensorBytes(nn::function::Randn({127}, DataType::kFLOAT64, cpu, generator)),
              TensorBytes(nn::function::Randn({127}, DataType::kFLOAT64, cpu, fresh)));
    EXPECT_EQ(TensorBytes(generator.get_state()), TensorBytes(fresh.get_state()));
}

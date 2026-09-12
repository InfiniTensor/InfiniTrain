#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

#include "infini_train/include/device.h"
#include "tests/common/test_utils.h"

using namespace infini_train;
using infini_train::test::TensorBytes;

class GeneratorCoreTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorCoreTest, PublicInterfaceCopyCloneAndSeed) {
    const Device device = GetDevice();
    auto generator = CreateGenerator(device);
    EXPECT_EQ(generator.device(), device);
    EXPECT_EQ(generator.current_seed(), Generator::kDefaultSeed);

    generator.set_current_seed(1234);
    EXPECT_EQ(generator.current_seed(), 1234U);
    const auto first = TensorBytes(nn::function::Rand({128}, DataType::kFLOAT32, device, generator));
    generator.set_current_seed(1234);
    EXPECT_EQ(first, TensorBytes(nn::function::Rand({128}, DataType::kFLOAT32, device, generator)));

    generator.set_current_seed(2345);
    Generator alias = generator;
    EXPECT_EQ(alias, generator);
    const auto state_before = TensorBytes(alias.get_state());
    nn::function::Rand({32}, DataType::kFLOAT32, device, generator);
    EXPECT_NE(state_before, TensorBytes(alias.get_state()));

    auto clone = generator.Clone();
    EXPECT_NE(clone, generator);
    EXPECT_EQ(TensorBytes(clone.get_state()), TensorBytes(generator.get_state()));
    clone.set_current_seed(3456);
    EXPECT_EQ(generator.current_seed(), 2345U);
    EXPECT_EQ(clone.current_seed(), 3456U);

    const uint64_t generated_seed = clone.Seed();
    EXPECT_EQ(clone.current_seed(), generated_seed);
}

TEST_P(GeneratorCoreTest, StateRestoresUniformSequence) {
    const Device device = GetDevice();
    auto generator = CreateGenerator(device, 4567);
    const auto state = generator.get_state();
    EXPECT_EQ(state->Dtype(), DataType::kUINT8);
    EXPECT_EQ(state->GetDevice(), Device());

    const auto uniform = TensorBytes(nn::function::Rand({129}, DataType::kFLOAT32, device, generator));
    generator.set_state(*state);
    EXPECT_EQ(uniform, TensorBytes(nn::function::Rand({129}, DataType::kFLOAT32, device, generator)));
}

TEST_P(GeneratorCoreTest, StateRejectsMalformedInput) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const Device device = GetDevice();
    auto generator = CreateGenerator(device, 4601);
    auto wrong_dtype = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, Device());
    EXPECT_DEATH(generator.set_state(*wrong_dtype), "UINT8");

    const auto valid_state = generator.get_state();
    auto truncated = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(valid_state->SizeInBytes() - 1)}, DataType::kUINT8, Device());
    EXPECT_DEATH(generator.set_state(*truncated), "Check failed");
}

TEST_P(GeneratorCoreTest, DefaultExplicitAndUndefinedPathsHaveExpectedState) {
    const Device device = GetDevice();
    constexpr uint64_t seed = 7890;

    ManualSeed(seed);
    const auto *first_default = &GetDefaultGenerator(device);
    const auto *second_default = &GetDefaultGenerator(device);
    EXPECT_EQ(first_default, second_default);
    EXPECT_EQ(first_default->device(), device);

    const auto default_state = TensorBytes(first_default->get_state());
    const auto explicit_result
        = TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device, CreateGenerator(device, seed)));
    EXPECT_EQ(default_state, TensorBytes(first_default->get_state()));
    EXPECT_EQ(explicit_result, TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device)));

    ManualSeed(seed + 1);
    const auto default_result = TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device));
    ManualSeed(seed + 1);
    std::optional<Generator> undefined = std::nullopt;
    EXPECT_EQ(default_result, TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device, undefined)));
}

TEST_P(GeneratorCoreTest, InitializerEntryPointsUseExplicitGenerator) {
    const Device device = GetDevice();
    auto make_tensor = [&] {
        return std::make_shared<Tensor>(std::vector<int64_t>{16, 16}, DataType::kFLOAT32, device);
    };

    auto normal_a = make_tensor();
    auto normal_b = make_tensor();
    nn::init::Normal(normal_a, 1.5f, 0.5f, CreateGenerator(device, 8101));
    nn::init::Normal(normal_b, 1.5f, 0.5f, CreateGenerator(device, 8101));
    EXPECT_EQ(TensorBytes(normal_a), TensorBytes(normal_b));

    auto uniform_a = make_tensor();
    auto uniform_b = make_tensor();
    nn::init::Uniform(uniform_a, -2.0f, 3.0f, CreateGenerator(device, 8102));
    uniform_b->Uniform(-2.0f, 3.0f, CreateGenerator(device, 8102));
    EXPECT_EQ(TensorBytes(uniform_a), TensorBytes(uniform_b));

    auto kaiming_a = make_tensor();
    auto kaiming_b = make_tensor();
    nn::init::KaimingUniform(kaiming_a, 0.1f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kLeakyReLU,
                             CreateGenerator(device, 8103));
    nn::init::KaimingUniform(kaiming_b, 0.1f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kLeakyReLU,
                             CreateGenerator(device, 8103));
    EXPECT_EQ(TensorBytes(kaiming_a), TensorBytes(kaiming_b));
}

INFINI_TRAIN_REGISTER_TEST(GeneratorCoreTest);

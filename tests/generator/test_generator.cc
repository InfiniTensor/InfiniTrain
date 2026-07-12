#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"
#if defined(USE_CUDA)
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#endif
#include "tests/common/test_utils.h"

namespace infini_train::test {
namespace {

constexpr uint64_t kSeed = 0x12345678ULL;
constexpr int64_t kElements = 4096;

Generator CreateGenerator(Device device, uint64_t seed) {
    if (device.IsCPU()) {
        return core::cpu::createCPUGenerator(seed);
    }
#if defined(USE_CUDA)
    return core::cuda::createCUDAGenerator(device.index(), seed);
#else
    (void)seed;
    throw std::runtime_error("CUDA support is disabled");
#endif
}

const Generator &GetDefaultGenerator(Device device) {
    if (device.IsCPU()) {
        return core::cpu::getDefaultCPUGenerator();
    }
#if defined(USE_CUDA)
    return core::cuda::getDefaultCUDAGenerator(device.index());
#else
    throw std::runtime_error("CUDA support is disabled");
#endif
}

void Synchronize(Device device) {
    core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
}

std::vector<float> ToHostVector(const std::shared_ptr<Tensor> &tensor) {
    const Device device = tensor->GetDevice();
    Synchronize(device);
    Tensor cpu = tensor->To(Device(Device::DeviceType::kCPU, 0));
    Synchronize(device);
    const auto *data = static_cast<const float *>(cpu.DataPtr());
    return std::vector<float>(data, data + cpu.NumElements());
}

std::vector<uint8_t> StateBytes(const Generator &generator) {
    auto state = generator.get_state();
    const auto *data = static_cast<const uint8_t *>(state->DataPtr());
    return std::vector<uint8_t>(data, data + state->SizeInBytes());
}

std::shared_ptr<Tensor> MakeTensor(Device device) {
    return std::make_shared<Tensor>(std::vector<int64_t>{kElements}, DataType::kFLOAT32, device);
}

class GeneratorTest : public InfiniTrainTest {};

TEST_P(GeneratorTest, SupportsDeviceAndSeedInterface) {
    const Device device = GetDevice();
    Generator generator = CreateGenerator(device, kSeed);

    EXPECT_TRUE(generator.defined());
    EXPECT_EQ(generator.device(), device);
    EXPECT_EQ(generator.current_seed(), kSeed);
    generator.set_current_seed(kSeed + 1);
    EXPECT_EQ(generator.current_seed(), kSeed + 1);
}

TEST_P(GeneratorTest, UniformIsReproducibleForSameSeed) {
    Generator first_generator = CreateGenerator(GetDevice(), kSeed);
    Generator second_generator = CreateGenerator(GetDevice(), kSeed);
    auto first = MakeTensor(GetDevice());
    auto second = MakeTensor(GetDevice());

    nn::init::Uniform(first, -3.0f, 7.0f, first_generator);
    nn::init::Uniform(second, -3.0f, 7.0f, second_generator);

    EXPECT_EQ(ToHostVector(first), ToHostVector(second));
}

TEST_P(GeneratorTest, NormalIsReproducibleForSameSeed) {
    Generator first_generator = CreateGenerator(GetDevice(), kSeed);
    Generator second_generator = CreateGenerator(GetDevice(), kSeed);
    auto first = MakeTensor(GetDevice());
    auto second = MakeTensor(GetDevice());

    nn::init::Normal(first, 2.0f, 0.5f, first_generator);
    nn::init::Normal(second, 2.0f, 0.5f, second_generator);

    EXPECT_EQ(ToHostVector(first), ToHostVector(second));
}

TEST_P(GeneratorTest, RandAndRandnSupportExplicitAndDefaultGenerators) {
    const Device device = GetDevice();
    Generator first_generator = CreateGenerator(device, kSeed);
    Generator second_generator = CreateGenerator(device, kSeed);

    auto first = nn::function::Rand({17, 19}, DataType::kFLOAT32, device, first_generator);
    auto second = nn::function::Rand({17, 19}, DataType::kFLOAT32, device, second_generator);
    EXPECT_EQ(ToHostVector(first), ToHostVector(second));

    manual_seed(kSeed);
    auto first_default = nn::function::Randn({17, 19}, DataType::kFLOAT32, device);
    manual_seed(kSeed);
    auto second_default = nn::function::Randn({17, 19}, DataType::kFLOAT32, device);
    EXPECT_EQ(ToHostVector(first_default), ToHostVector(second_default));
}

TEST_P(GeneratorTest, DropoutMaskIsReproducibleForSameSeed) {
    const Device device = GetDevice();
    auto input = MakeTensor(device);
    input->Fill(1.0f);

    manual_seed(kSeed);
    auto first = nn::function::Dropout(input, 0.25, true);
    manual_seed(kSeed);
    auto second = nn::function::Dropout(input, 0.25, true);
    manual_seed(kSeed + 1);
    auto different_seed = nn::function::Dropout(input, 0.25, true);

    EXPECT_EQ(ToHostVector(first), ToHostVector(second));
    EXPECT_NE(ToHostVector(first), ToHostVector(different_seed));
}

TEST_P(GeneratorTest, ConsecutiveCallsAdvanceSequence) {
    Generator generator = CreateGenerator(GetDevice(), kSeed);
    auto first = MakeTensor(GetDevice());
    auto second = MakeTensor(GetDevice());

    nn::init::Uniform(first, 0.0f, 1.0f, generator);
    nn::init::Uniform(second, 0.0f, 1.0f, generator);

    EXPECT_NE(ToHostVector(first), ToHostVector(second));
}

TEST_P(GeneratorTest, DifferentSeedsProduceDifferentSequences) {
    Generator first_generator = CreateGenerator(GetDevice(), kSeed);
    Generator second_generator = CreateGenerator(GetDevice(), kSeed + 1);
    auto first = MakeTensor(GetDevice());
    auto second = MakeTensor(GetDevice());

    nn::init::Uniform(first, 0.0f, 1.0f, first_generator);
    nn::init::Uniform(second, 0.0f, 1.0f, second_generator);

    EXPECT_NE(ToHostVector(first), ToHostVector(second));
}

TEST_P(GeneratorTest, SameSeedAndCallOrderReproduceResults) {
    const Device device = GetDevice();
    auto first_uniform = MakeTensor(device);
    auto second_uniform = MakeTensor(device);
    auto first_normal = MakeTensor(device);
    auto second_normal = MakeTensor(device);
    auto input = MakeTensor(device);
    input->Fill(1.0f);

    manual_seed(kSeed);
    nn::init::Uniform(first_uniform);
    auto first_dropout = nn::function::Dropout(input, 0.5, true);
    nn::init::Normal(first_normal);

    manual_seed(kSeed);
    nn::init::Uniform(second_uniform);
    auto second_dropout = nn::function::Dropout(input, 0.5, true);
    nn::init::Normal(second_normal);

    EXPECT_EQ(ToHostVector(first_uniform), ToHostVector(second_uniform));
    EXPECT_EQ(ToHostVector(first_dropout), ToHostVector(second_dropout));
    EXPECT_EQ(ToHostVector(first_normal), ToHostVector(second_normal));
}

TEST_P(GeneratorTest, StateRestoreReplaysUniformSequence) {
    Generator generator = CreateGenerator(GetDevice(), kSeed);
    auto prefix = MakeTensor(GetDevice());
    auto expected = MakeTensor(GetDevice());
    auto actual = MakeTensor(GetDevice());

    nn::init::Uniform(prefix, 0.0f, 1.0f, generator);
    auto state = generator.get_state();
    nn::init::Uniform(expected, 0.0f, 1.0f, generator);
    generator.set_state(*state);
    nn::init::Uniform(actual, 0.0f, 1.0f, generator);

    EXPECT_EQ(ToHostVector(expected), ToHostVector(actual));
}

TEST_P(GeneratorTest, StateRestoreReplaysNormalSequence) {
    Generator generator = CreateGenerator(GetDevice(), kSeed);
    auto prefix = MakeTensor(GetDevice());
    auto expected = MakeTensor(GetDevice());
    auto actual = MakeTensor(GetDevice());

    nn::init::Normal(prefix, 0.0f, 1.0f, generator);
    auto state = generator.get_state();
    nn::init::Normal(expected, 0.0f, 1.0f, generator);
    generator.set_state(*state);
    nn::init::Normal(actual, 0.0f, 1.0f, generator);

    EXPECT_EQ(ToHostVector(expected), ToHostVector(actual));
}

TEST_P(GeneratorTest, ExplicitGeneratorDoesNotAdvanceDefaultGenerator) {
    const Device device = GetDevice();
    manual_seed(kSeed);
    const Generator &default_generator = GetDefaultGenerator(device);
    const std::vector<uint8_t> before = StateBytes(default_generator);
    Generator explicit_generator = CreateGenerator(device, kSeed + 1);

    auto tensor = MakeTensor(device);
    nn::init::Uniform(tensor, 0.0f, 1.0f, explicit_generator);

    EXPECT_EQ(before, StateBytes(default_generator));
}

TEST_P(GeneratorTest, MissingGeneratorUsesAndAdvancesDefaultGenerator) {
    const Device device = GetDevice();
    manual_seed(kSeed);
    const Generator &default_generator = GetDefaultGenerator(device);
    const std::vector<uint8_t> before = StateBytes(default_generator);

    auto tensor = MakeTensor(device);
    nn::init::Uniform(tensor);

    EXPECT_NE(before, StateBytes(default_generator));
}

TEST_P(GeneratorTest, RepeatedDefaultGeneratorLookupSharesState) {
    const Device device = GetDevice();
    const Generator &first = GetDefaultGenerator(device);
    const Generator &second = GetDefaultGenerator(device);

    EXPECT_EQ(first, second);
}

TEST_P(GeneratorTest, GlobalManualSeedReproducesDefaultGeneratorResults) {
    const Device device = GetDevice();
    auto first = MakeTensor(device);
    auto second = MakeTensor(device);

    manual_seed(kSeed);
    EXPECT_EQ(GetDefaultGenerator(device).current_seed(), kSeed);
    nn::init::Uniform(first);
    manual_seed(kSeed);
    nn::init::Uniform(second);

    EXPECT_EQ(ToHostVector(first), ToHostVector(second));
}

TEST_P(GeneratorTest, KaimingUniformUsesExplicitGenerator) {
    Generator first_generator = CreateGenerator(GetDevice(), kSeed);
    Generator second_generator = CreateGenerator(GetDevice(), kSeed);
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{64, 32}, DataType::kFLOAT32, GetDevice());
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{64, 32}, DataType::kFLOAT32, GetDevice());

    nn::init::KaimingUniform(first, 0.0f, nn::init::KaimingMode::kFanIn,
                             nn::init::NonLinearityType::kReLU, first_generator);
    nn::init::KaimingUniform(second, 0.0f, nn::init::KaimingMode::kFanIn,
                             nn::init::NonLinearityType::kReLU, second_generator);

    EXPECT_EQ(ToHostVector(first), ToHostVector(second));
}

INFINI_TRAIN_REGISTER_TEST(GeneratorTest);

} // namespace
} // namespace infini_train::test

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {
std::vector<float> TensorData(const std::shared_ptr<Tensor> &tensor) {
    CHECK(tensor->GetDevice().IsCPU());
    if (tensor->NumElements() == 0) {
        return {};
    }
    const auto *data = static_cast<const float *>(tensor->DataPtr());
    return std::vector<float>(data, data + tensor->NumElements());
}

bool SameValues(const std::vector<float> &a, const std::vector<float> &b) { return a == b; }

bool DifferentValues(const std::vector<float> &a, const std::vector<float> &b) { return a != b; }

uint64_t HashValues(uint64_t hash, const std::vector<float> &values) {
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    for (float value : values) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (bits >> shift) & 0xffU;
            hash *= kFnvPrime;
        }
    }
    return hash;
}

std::string SerializedState(const std::vector<uint8_t> &state) { return std::string(state.begin(), state.end()); }

std::vector<uint8_t> StateBytes(const std::string &state) { return std::vector<uint8_t>(state.begin(), state.end()); }

uint64_t CUDASemanticOffset(const std::vector<uint8_t> &state) {
    std::istringstream iss(SerializedState(state));
    std::string line;
    CHECK(std::getline(iss, line));
    CHECK(std::getline(iss, line));
    CHECK(std::getline(iss, line));
    CHECK(std::getline(iss, line));
    return std::stoull(line);
}

std::vector<uint8_t> ReplaceStateLine(const std::vector<uint8_t> &state, size_t line_index,
                                      const std::string &replacement) {
    std::string serialized = SerializedState(state);
    size_t offset_begin = 0;
    for (size_t i = 0; i < line_index; ++i) {
        offset_begin = serialized.find('\n', offset_begin);
        CHECK_NE(offset_begin, std::string::npos);
        ++offset_begin;
    }
    size_t offset_end = serialized.find('\n', offset_begin);
    if (offset_end == std::string::npos) {
        offset_end = serialized.size();
    }
    serialized.replace(offset_begin, offset_end - offset_begin, replacement);
    return StateBytes(serialized);
}

std::vector<uint8_t> ReplaceCPUEngineToken(const std::vector<uint8_t> &state, size_t token_index,
                                           const std::string &replacement) {
    const std::string serialized = SerializedState(state);
    size_t engine_begin = serialized.find('\n');
    CHECK_NE(engine_begin, std::string::npos);
    engine_begin = serialized.find('\n', engine_begin + 1);
    CHECK_NE(engine_begin, std::string::npos);
    ++engine_begin;

    std::istringstream engine_stream(serialized.substr(engine_begin));
    std::vector<std::string> tokens;
    for (std::string token; engine_stream >> token;) { tokens.push_back(token); }
    CHECK_LT(token_index, tokens.size());
    tokens[token_index] = replacement;

    std::ostringstream rebuilt;
    rebuilt << serialized.substr(0, engine_begin);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) {
            rebuilt << ' ';
        }
        rebuilt << tokens[i];
    }
    return StateBytes(rebuilt.str());
}

std::vector<uint8_t> MakeAllZeroCPUEngineState(const std::vector<uint8_t> &state) {
    const std::string serialized = SerializedState(state);
    size_t engine_begin = serialized.find('\n');
    CHECK_NE(engine_begin, std::string::npos);
    engine_begin = serialized.find('\n', engine_begin + 1);
    CHECK_NE(engine_begin, std::string::npos);
    ++engine_begin;

    std::ostringstream rebuilt;
    rebuilt << serialized.substr(0, engine_begin);
    for (size_t i = 0; i < std::mt19937::state_size; ++i) {
        if (i != 0) {
            rebuilt << ' ';
        }
        rebuilt << '0';
    }
    rebuilt << ' ' << std::mt19937::state_size;
    return StateBytes(rebuilt.str());
}

std::vector<uint8_t> ReplaceCUDASemanticOffset(const std::vector<uint8_t> &state, const std::string &offset) {
    return ReplaceStateLine(state, 3, offset);
}
} // namespace

TEST(CPUGeneratorTest, InterfaceAndStateRoundTrip) {
    auto generator = MakeCPUGenerator(123);

    EXPECT_EQ(generator->InitialSeed(), 123U);
    EXPECT_TRUE(generator->GetDevice().IsCPU());

    auto state = generator->GetState();
    EXPECT_FALSE(state.empty());

    generator->ManualSeed(456);
    EXPECT_EQ(generator->InitialSeed(), 456U);

    const auto generated_seed = generator->Seed();
    EXPECT_EQ(generator->InitialSeed(), generated_seed);

    generator->SetState(state);
    EXPECT_EQ(generator->InitialSeed(), 123U);
}

TEST(CPUGeneratorTest, ProcessDefaultSeedIsSharedByInitialDefaultGenerators) {
    auto cpu = GetDefaultCPUGenerator();
    auto cuda = GetDefaultCUDAGenerator(0);

    EXPECT_EQ(cpu->InitialSeed(), cuda->InitialSeed());
}

TEST(CPUGeneratorTest, RandUsesManualSeedForReproducibility) {
    ManualSeed(123);
    auto a = TensorData(nn::function::Rand({2, 3}));

    ManualSeed(123);
    auto b = TensorData(nn::function::Rand({2, 3}));

    ManualSeed(456);
    auto c = TensorData(nn::function::Rand({2, 3}));

    EXPECT_TRUE(SameValues(a, b));
    EXPECT_TRUE(DifferentValues(a, c));
}

TEST(CPUGeneratorTest, RandnUsesManualSeedForReproducibility) {
    ManualSeed(123);
    auto a = TensorData(nn::function::Randn({2, 3}));

    ManualSeed(123);
    auto b = TensorData(nn::function::Randn({2, 3}));

    ManualSeed(456);
    auto c = TensorData(nn::function::Randn({2, 3}));

    EXPECT_TRUE(SameValues(a, b));
    EXPECT_TRUE(DifferentValues(a, c));
}

TEST(CPUGeneratorTest, CrossProcessReproducibilityDigest) {
    ManualSeed(20260712);
    auto parameter = std::make_shared<Tensor>(std::vector<int64_t>{8, 16}, DataType::kFLOAT32, Device());
    auto initialized = TensorData(nn::init::KaimingUniform(parameter));
    auto uniform = TensorData(nn::function::Rand({31}));
    auto normal = TensorData(nn::function::Randn({29}));

    uint64_t digest = 1469598103934665603ULL;
    digest = HashValues(digest, initialized);
    digest = HashValues(digest, uniform);
    digest = HashValues(digest, normal);
    std::cout << "GENERATOR_REPRODUCIBILITY_DIGEST=" << std::hex << digest << std::dec << std::endl;
}

TEST(CPUGeneratorTest, ModelLevelInitializationAndRandomSequenceReplay) {
    ManualSeed(2026);
    auto parameter = std::make_shared<Tensor>(std::vector<int64_t>{4, 8}, DataType::kFLOAT32, Device());
    auto initialized = TensorData(nn::init::KaimingUniform(parameter));
    auto state = GetDefaultCPUGenerator()->GetState();
    auto uniform = TensorData(nn::function::Rand({17}));
    auto normal = TensorData(nn::function::Randn({19}));

    GetDefaultCPUGenerator()->SetState(state);
    EXPECT_EQ(uniform, TensorData(nn::function::Rand({17})));
    EXPECT_EQ(normal, TensorData(nn::function::Randn({19})));

    ManualSeed(2026);
    auto replay_parameter = std::make_shared<Tensor>(std::vector<int64_t>{4, 8}, DataType::kFLOAT32, Device());
    EXPECT_EQ(initialized, TensorData(nn::init::KaimingUniform(replay_parameter)));
}

TEST(CPUGeneratorTest, ManualSeedUsesHighBitsOfUint64Seed) {
    constexpr uint64_t low_seed = 123;
    constexpr uint64_t high_seed = low_seed + (1ULL << 32);

    auto cpu_low = MakeCPUGenerator(low_seed);
    auto cpu_high = MakeCPUGenerator(high_seed);

    auto cpu_a = TensorData(nn::function::Rand({8}, Device(), cpu_low));
    auto cpu_b = TensorData(nn::function::Rand({8}, Device(), cpu_high));

    EXPECT_EQ(cpu_high->InitialSeed(), high_seed);
    EXPECT_TRUE(DifferentValues(cpu_a, cpu_b));

    auto cuda_low = MakeCUDAGenerator(0, low_seed);
    auto cuda_high = MakeCUDAGenerator(0, high_seed);
    std::vector<float> cuda_a(8);
    std::vector<float> cuda_b(8);
    cuda_low->FillUniform(cuda_a, 0.0f, 1.0f);
    cuda_high->FillUniform(cuda_b, 0.0f, 1.0f);

    EXPECT_EQ(cuda_high->InitialSeed(), high_seed);
    EXPECT_TRUE(DifferentValues(cuda_a, cuda_b));
}

TEST(CPUGeneratorTest, StateRestoreReplaysSequence) {
    auto generator = MakeCPUGenerator(123);

    (void)TensorData(nn::function::Rand({4}, Device(), generator));
    auto state = generator->GetState();

    auto b = TensorData(nn::function::Rand({4}, Device(), generator));
    generator->SetState(state);
    auto b2 = TensorData(nn::function::Rand({4}, Device(), generator));

    EXPECT_TRUE(SameValues(b, b2));
}

TEST(CPUGeneratorTest, SetStateRejectsCorruptedState) {
    auto generator = MakeCPUGenerator(123);
    std::vector<uint8_t> corrupted_state = {'n', 'o', 't', '-', 'a', '-', 's', 't', 'a', 't', 'e'};

    EXPECT_DEATH(generator->SetState(corrupted_state), "Invalid CPU generator state header");
}

TEST(CPUGeneratorTest, SetStateRejectsEmptyAndTruncatedState) {
    auto cpu_generator = MakeCPUGenerator(123);
    EXPECT_DEATH(cpu_generator->SetState({}), "Invalid CPU generator state header");

    auto cpu_state = cpu_generator->GetState();
    cpu_state.resize(cpu_state.size() / 2);
    EXPECT_DEATH(cpu_generator->SetState(cpu_state), "Invalid CPU generator engine state");

    auto cuda_generator = MakeCUDAGenerator(0, 123);
    auto cuda_state = cuda_generator->GetState();
    const auto offset_separator = SerializedState(cuda_state).rfind('\n', cuda_state.size() - 2);
    ASSERT_NE(offset_separator, std::string::npos);
    cuda_state.resize(offset_separator + 1);
    EXPECT_DEATH(cuda_generator->SetState(cuda_state), "Invalid CUDA generator offset in state");
}

TEST(CPUGeneratorTest, SetStateRejectsTrailingStateData) {
    auto cpu_generator = MakeCPUGenerator(123);
    auto cpu_state = cpu_generator->GetState();
    const std::string trailing_data = " trailing-junk";
    cpu_state.insert(cpu_state.end(), trailing_data.begin(), trailing_data.end());

    EXPECT_DEATH(cpu_generator->SetState(cpu_state), "trailing data");

    auto cuda_generator = MakeCUDAGenerator(0, 123);
    auto cuda_state = cuda_generator->GetState();
    cuda_state.insert(cuda_state.end(), trailing_data.begin(), trailing_data.end());

    EXPECT_DEATH(cuda_generator->SetState(cuda_state), "trailing data");
}

TEST(CPUGeneratorTest, SetStateRejectsTrailingWhitespace) {
    auto cpu_generator = MakeCPUGenerator(123);
    auto cpu_state = cpu_generator->GetState();
    cpu_state.push_back(' ');
    EXPECT_DEATH(cpu_generator->SetState(cpu_state), "trailing data");

    auto cuda_generator = MakeCUDAGenerator(0, 123);
    auto cuda_state = cuda_generator->GetState();
    cuda_state.push_back('\n');
    EXPECT_DEATH(cuda_generator->SetState(cuda_state), "trailing data");
}

TEST(CPUGeneratorTest, SetStateRejectsWrongBackendState) {
    auto cpu_generator = MakeCPUGenerator(123);
    auto cuda_generator = MakeCUDAGenerator(0, 123);

    EXPECT_DEATH(cpu_generator->SetState(cuda_generator->GetState()), "Invalid CPU generator state header");
    EXPECT_DEATH(cuda_generator->SetState(cpu_generator->GetState()), "Invalid CUDA generator state header");
}

TEST(CPUGeneratorTest, SetStateRejectsNegativeUnsignedFields) {
    auto cpu_generator = MakeCPUGenerator(123);
    auto negative_cpu_seed = ReplaceStateLine(cpu_generator->GetState(), 1, "-1");
    EXPECT_DEATH(cpu_generator->SetState(negative_cpu_seed), "Invalid CPU generator seed in state");

    auto cuda_generator = MakeCUDAGenerator(0, 123);
    auto negative_cuda_seed = ReplaceStateLine(cuda_generator->GetState(), 1, "-1");
    EXPECT_DEATH(cuda_generator->SetState(negative_cuda_seed), "Invalid CUDA generator seed in state");

    auto negative_cuda_offset = ReplaceStateLine(cuda_generator->GetState(), 3, "-1");
    EXPECT_DEATH(cuda_generator->SetState(negative_cuda_offset), "Invalid CUDA generator offset in state");
}

TEST(CPUGeneratorTest, CPUSetStateRejectsInvalidEngineStructure) {
    auto generator = MakeCPUGenerator(123);
    const auto state = generator->GetState();

    auto oversized_word = ReplaceCPUEngineToken(state, 0, "4294967296");
    EXPECT_DEATH(generator->SetState(oversized_word), "Invalid CPU generator engine word in state");

    auto invalid_position
        = ReplaceCPUEngineToken(state, std::mt19937::state_size, std::to_string(std::mt19937::state_size + 1));
    EXPECT_DEATH(generator->SetState(invalid_position), "Invalid CPU generator engine position in state");

    auto all_zero = MakeAllZeroCPUEngineState(state);
    EXPECT_DEATH(generator->SetState(all_zero), "Invalid CPU generator engine state: all-zero state");
}

TEST(CPUGeneratorTest, CUDAGeneratorSetStateRejectsInvalidSourceDeviceIndex) {
    auto generator = MakeCUDAGenerator(0, 123);
    auto negative_index = ReplaceStateLine(generator->GetState(), 2, "-1");
    EXPECT_DEATH(generator->SetState(negative_index), "Invalid CUDA generator device index in state");

    auto oversized_index = ReplaceStateLine(generator->GetState(), 2, "128");
    EXPECT_DEATH(generator->SetState(oversized_index), "Invalid CUDA generator device index in state");
}

TEST(CPUGeneratorTest, CUDAGeneratorSetStateFromDifferentDeviceReplaysSequence) {
    auto cuda0 = MakeCUDAGenerator(0, 123);
    std::vector<float> ignored(5);
    std::vector<float> expected(11);
    std::vector<float> actual(11);
    cuda0->FillUniform(ignored, 0.0f, 1.0f);
    auto state = cuda0->GetState();
    cuda0->FillUniform(expected, 0.0f, 1.0f);

    auto cuda1 = MakeCUDAGenerator(1, 999);
    cuda1->SetState(state);
    cuda1->FillUniform(actual, 0.0f, 1.0f);

    EXPECT_EQ(expected, actual);
    EXPECT_EQ(cuda1->InitialSeed(), 123U);
    EXPECT_EQ(cuda1->GetDevice().index(), 1);
}

TEST(CPUGeneratorTest, DefaultGeneratorAdvancesAcrossCalls) {
    ManualSeed(123);
    auto a = TensorData(nn::function::Rand({4}));
    auto b = TensorData(nn::function::Rand({4}));

    ManualSeed(123);
    auto a2 = TensorData(nn::function::Rand({4}));

    EXPECT_TRUE(DifferentValues(a, b));
    EXPECT_TRUE(SameValues(a, a2));
}

TEST(CPUGeneratorTest, ExplicitGeneratorDoesNotAdvanceDefaultGenerator) {
    ManualSeed(777);
    auto default_generator = GetDefaultCPUGenerator();
    auto before = default_generator->GetState();

    auto explicit_generator = MakeCPUGenerator(123);
    (void)TensorData(nn::function::Rand({8}, Device(), explicit_generator));

    auto after = default_generator->GetState();
    EXPECT_EQ(before, after);
}

TEST(CPUGeneratorTest, TensorUniformAcceptsExplicitGenerator) {
    auto g1 = MakeCPUGenerator(999);
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device());
    auto a = TensorData(t1->Uniform(-1.0f, 1.0f, g1));

    auto g2 = MakeCPUGenerator(999);
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device());
    auto b = TensorData(t2->Uniform(-1.0f, 1.0f, g2));

    EXPECT_TRUE(SameValues(a, b));
    EXPECT_TRUE(std::all_of(a.begin(), a.end(), [](float x) { return x >= -1.0f && x <= 1.0f; }));
}

TEST(CPUGeneratorTest, ZeroElementRandomCallsDoNotAdvanceState) {
    auto generator = MakeCPUGenerator(123);
    auto before = generator->GetState();
    EXPECT_EQ(TensorData(nn::function::Rand({0}, Device(), generator)).size(), 0U);
    EXPECT_EQ(TensorData(nn::function::Randn({0}, Device(), generator)).size(), 0U);
    EXPECT_EQ(before, generator->GetState());
}

TEST(CPUGeneratorTest, RandomDistributionsRejectInvalidParameters) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, Device());
    EXPECT_DEATH((void)nn::init::Uniform(tensor, 2.0f, 1.0f), "Check failed: from <= to");
    EXPECT_DEATH((void)nn::init::Normal(tensor, 0.0f, -0.1f), "Check failed: std >= 0.0f");
}

TEST(CPUGeneratorTest, EmptyRandomCallsStillValidateParametersAndBackend) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{0}, DataType::kFLOAT32, Device());
    EXPECT_DEATH((void)nn::init::Normal(tensor, 0.0f, -0.1f), "Check failed: std >= 0.0f");

    auto cuda_generator = MakeCUDAGenerator(0, 123);
    EXPECT_DEATH((void)nn::init::Uniform(tensor, 0.0f, 1.0f, cuda_generator),
                 "Generator backend must match tensor device backend");
}

TEST(CPUGeneratorTest, UniformRejectsNonFiniteAndOverflowingBoundsEvenWhenEmpty) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{0}, DataType::kFLOAT32, Device());
    const float infinity = std::numeric_limits<float>::infinity();
    const float max = std::numeric_limits<float>::max();
    EXPECT_DEATH((void)nn::init::Uniform(tensor, 0.0f, infinity), "Uniform upper bound must be finite");
    EXPECT_DEATH((void)nn::init::Uniform(tensor, -max, max), "Uniform bounds range exceeds float maximum");

    std::vector<float> empty;
    auto cpu_generator = MakeCPUGenerator(123);
    EXPECT_DEATH(cpu_generator->FillUniform(empty, 0.0f, infinity), "Uniform upper bound must be finite");

    auto cuda_generator = MakeCUDAGenerator(0, 123);
    EXPECT_DEATH(cuda_generator->FillUniform(empty, -max, max), "Uniform bounds range exceeds float maximum");
}

TEST(CPUGeneratorTest, EqualUniformBoundsReturnConstantAndAdvanceState) {
    constexpr size_t kNumElements = 17;
    auto generator = MakeCPUGenerator(123);
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{kNumElements}, DataType::kFLOAT32, Device());
    const auto before = generator->GetState();

    auto values = TensorData(nn::init::Uniform(tensor, 2.5f, 2.5f, generator));

    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value == 2.5f; }));
    EXPECT_NE(before, generator->GetState());
}

TEST(CPUGeneratorTest, RandomInitializationRejectsNonFloat32Tensors) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT16, Device());
    EXPECT_DEATH((void)nn::init::Uniform(tensor), "Random uniform currently supports float32 tensors");
    EXPECT_DEATH((void)nn::init::Normal(tensor), "Random normal currently supports float32 tensors");
}

TEST(CPUGeneratorTest, ZeroStdNormalReturnsMeanAndAdvancesState) {
    constexpr size_t kNumElements = 17;
    auto zero_std_generator = MakeCPUGenerator(123);
    auto reference_generator = MakeCPUGenerator(123);
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{kNumElements}, DataType::kFLOAT32, Device());

    auto values = TensorData(nn::init::Normal(tensor, 2.5f, 0.0f, zero_std_generator));
    std::vector<float> ignored(kNumElements);
    reference_generator->FillNormal(ignored, 0.0f, 1.0f);

    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value == 2.5f; }));
    EXPECT_EQ(zero_std_generator->GetState(), reference_generator->GetState());
}

TEST(CPUGeneratorTest, CUDAHostZeroStdNormalReturnsMeanAndAdvancesOffset) {
    constexpr size_t kNumElements = 17;
    auto generator = MakeCUDAGenerator(0, 123);
    std::vector<float> values(kNumElements);

    generator->FillNormal(values, 2.5f, 0.0f);

    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value == 2.5f; }));
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), kNumElements * 2);
}

TEST(CPUGeneratorTest, LargeRandomTensorSpansManyBlocksWorthOfValues) {
    auto values = TensorData(nn::function::Rand({65537}, Device(), MakeCPUGenerator(123)));
    EXPECT_EQ(values.size(), 65537U);
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value >= 0.0f && value <= 1.0f; }));
}

TEST(CPUGeneratorTest, KaimingUniformUsesExplicitGeneratorReproducibly) {
    auto g1 = MakeCPUGenerator(314159);
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{4, 8}, DataType::kFLOAT32, Device());
    auto a = TensorData(
        nn::init::KaimingUniform(t1, 0.0f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kReLU, g1));

    auto g2 = MakeCPUGenerator(314159);
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{4, 8}, DataType::kFLOAT32, Device());
    auto b = TensorData(
        nn::init::KaimingUniform(t2, 0.0f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kReLU, g2));

    EXPECT_EQ(a, b);
}

TEST(CPUGeneratorTest, ExplicitGeneratorBackendMustMatchTensorDeviceBackend) {
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device());
    auto cuda_generator = MakeCUDAGenerator(0, 123);

    EXPECT_DEATH((void)tensor->Uniform(0.0f, 1.0f, cuda_generator),
                 "Generator backend must match tensor device backend");
}

TEST(CPUGeneratorTest, DefaultCUDAGeneratorsArePerDevice) {
    ManualSeedAll(321);
    auto cuda0 = GetDefaultCUDAGenerator(0);
    auto cuda1 = GetDefaultCUDAGenerator(1);

    EXPECT_TRUE(cuda0->GetDevice().IsCUDA());
    EXPECT_TRUE(cuda1->GetDevice().IsCUDA());
    EXPECT_EQ(cuda0->GetDevice().index(), 0);
    EXPECT_EQ(cuda1->GetDevice().index(), 1);
    EXPECT_EQ(cuda0->InitialSeed(), 321U);
    EXPECT_EQ(cuda1->InitialSeed(), 321U);
    EXPECT_NE(cuda0->GetState(), cuda1->GetState());

    std::vector<float> buffer(4);
    auto cuda1_before = cuda1->GetState();
    cuda0->FillUniform(buffer, 0.0f, 1.0f);
    EXPECT_EQ(cuda1_before, cuda1->GetState());
    EXPECT_EQ(cuda0, GetDefaultCUDAGenerator(0));
}

TEST(CPUGeneratorTest, GetDefaultGeneratorDispatchesByDevice) {
    ManualSeedAll(654);

    auto cpu = GetDefaultGenerator(Device());
    auto cuda0 = GetDefaultGenerator(Device(Device::DeviceType::kCUDA, 0));

    EXPECT_EQ(cpu, GetDefaultCPUGenerator());
    EXPECT_EQ(cuda0, GetDefaultCUDAGenerator(0));
    EXPECT_TRUE(cpu->GetDevice().IsCPU());
    EXPECT_TRUE(cuda0->GetDevice().IsCUDA());
    EXPECT_EQ(cuda0->GetDevice().index(), 0);
    EXPECT_EQ(cpu->InitialSeed(), 654U);
    EXPECT_EQ(cuda0->InitialSeed(), 654U);
}

TEST(CPUGeneratorTest, ManualSeedAllSeedsExistingAndFutureDefaultGenerators) {
    auto cuda0 = GetDefaultCUDAGenerator(0);
    cuda0->ManualSeed(123);

    ManualSeedAll(888);

    EXPECT_EQ(GetDefaultCPUGenerator()->InitialSeed(), 888U);
    EXPECT_EQ(cuda0->InitialSeed(), 888U);
    EXPECT_EQ(GetDefaultCUDAGenerator(2)->InitialSeed(), 888U);
}

TEST(CPUGeneratorTest, ManualSeedAliasesManualSeedAll) {
    auto cuda0 = GetDefaultCUDAGenerator(0);
    cuda0->ManualSeed(123);

    ManualSeed(999);

    EXPECT_EQ(GetDefaultCPUGenerator()->InitialSeed(), 999U);
    EXPECT_EQ(cuda0->InitialSeed(), 999U);
    EXPECT_EQ(GetDefaultCUDAGenerator(3)->InitialSeed(), 999U);
}

TEST(CPUGeneratorTest, CUDAGeneratorStateRestoreReplaysSequence) {
    auto generator = MakeCUDAGenerator(0, 123);
    std::vector<float> ignored(4);
    std::vector<float> b(4);
    std::vector<float> b2(4);

    generator->FillUniform(ignored, 0.0f, 1.0f);
    auto state = generator->GetState();
    generator->FillUniform(b, 0.0f, 1.0f);
    generator->SetState(state);
    generator->FillUniform(b2, 0.0f, 1.0f);

    EXPECT_TRUE(SameValues(b, b2));
}

TEST(CPUGeneratorTest, CUDAGeneratorStateSerializesSemanticOffset) {
    auto generator = MakeCUDAGenerator(0, 123);
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), 0U);

    std::vector<float> uniform(5);
    generator->FillUniform(uniform, 0.0f, 1.0f);
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), 5U);

    std::vector<float> normal(7);
    generator->FillNormal(normal, 0.0f, 1.0f);
    auto state = generator->GetState();
    EXPECT_EQ(CUDASemanticOffset(state), 19U);

    generator->ManualSeed(456);
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), 0U);

    generator->SetState(state);
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), 19U);
}

TEST(CPUGeneratorTest, CUDAGeneratorSetStateRejectsCorruptedOffset) {
    auto generator = MakeCUDAGenerator(0, 123);
    auto state = ReplaceCUDASemanticOffset(generator->GetState(), "not-an-offset");

    EXPECT_DEATH(generator->SetState(state), "Invalid CUDA generator offset in state");
}

TEST(CPUGeneratorTest, CUDAGeneratorRejectsOffsetOverflow) {
    auto generator = MakeCUDAGenerator(0, 123);
    EXPECT_EQ(generator->ReserveRandomOffset(std::numeric_limits<uint64_t>::max()).second, 0U);
    EXPECT_DEATH((void)generator->ReserveRandomOffset(1), "CUDA generator offset overflow");
}

TEST(CPUGeneratorTest, CUDAGeneratorOffsetReservationsAreThreadSafeAndDisjoint) {
    constexpr size_t kThreadCount = 8;
    constexpr size_t kReservationsPerThread = 100;
    constexpr uint64_t kIncrement = 7;
    auto generator = MakeCUDAGenerator(0, 123);
    std::vector<uint64_t> offsets(kThreadCount * kReservationsPerThread);
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);

    for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        threads.emplace_back([&, thread_index]() {
            for (size_t i = 0; i < kReservationsPerThread; ++i) {
                const auto [seed, offset] = generator->ReserveRandomOffset(kIncrement);
                EXPECT_EQ(seed, 123U);
                offsets[thread_index * kReservationsPerThread + i] = offset;
            }
        });
    }
    for (auto &thread : threads) { thread.join(); }

    std::sort(offsets.begin(), offsets.end());
    for (size_t i = 0; i < offsets.size(); ++i) { EXPECT_EQ(offsets[i], static_cast<uint64_t>(i) * kIncrement); }
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), offsets.size() * kIncrement);
}

TEST(CPUGeneratorTest, DefaultCUDAGeneratorIsThreadSafeUnderConcurrentConsumption) {
    constexpr size_t kThreadCount = 8;
    constexpr size_t kReservationsPerThread = 50;
    constexpr uint64_t kIncrement = 3;
    ManualSeedAll(4242);
    auto generator = GetDefaultCUDAGenerator(0);
    std::vector<uint64_t> offsets(kThreadCount * kReservationsPerThread);
    std::vector<std::thread> threads;

    for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        threads.emplace_back([&, thread_index]() {
            for (size_t i = 0; i < kReservationsPerThread; ++i) {
                offsets[thread_index * kReservationsPerThread + i] = generator->ReserveRandomOffset(kIncrement).second;
            }
        });
    }
    for (auto &thread : threads) { thread.join(); }

    std::sort(offsets.begin(), offsets.end());
    for (size_t i = 0; i < offsets.size(); ++i) { EXPECT_EQ(offsets[i], static_cast<uint64_t>(i) * kIncrement); }
    EXPECT_EQ(CUDASemanticOffset(generator->GetState()), offsets.size() * kIncrement);
}

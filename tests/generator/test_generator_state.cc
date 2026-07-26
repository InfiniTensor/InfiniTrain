#include <cstdint>
#include <locale>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "tests/generator/generator_test_utils.h"

namespace infini_train::test {
namespace {

std::vector<uint8_t> ReplaceStateHeader(const std::vector<uint8_t> &state, const std::string &header) {
    std::string serialized(state.begin(), state.end());
    const auto newline = serialized.find('\n');
    CHECK_NE(newline, std::string::npos);
    serialized.replace(0, newline, header);
    return {serialized.begin(), serialized.end()};
}

class GroupedNumericPunct : public std::numpunct<char> {
protected:
    char do_thousands_sep() const override { return ','; }
    std::string do_grouping() const override { return "\3"; }
};

class ScopedGroupedNumericLocale {
public:
    ScopedGroupedNumericLocale() : original_(std::locale()) {
        std::locale::global(std::locale(std::locale::classic(), new GroupedNumericPunct));
    }

    ~ScopedGroupedNumericLocale() { std::locale::global(original_); }

    ScopedGroupedNumericLocale(const ScopedGroupedNumericLocale &) = delete;
    ScopedGroupedNumericLocale &operator=(const ScopedGroupedNumericLocale &) = delete;

private:
    std::locale original_;
};

} // namespace

TEST_P(GeneratorDeviceTest, StateRestoreReplaysSequence) {
    const auto device = GetDevice();
    auto generator = MakeGenerator(device, 123);
    (void)nn::function::Rand({7}, device, generator);
    const auto state = generator->GetState();

    const auto expected = CopyToCPUData(nn::function::Randn({11}, device, generator));
    generator->ManualSeed(999);
    generator->SetState(state);
    EXPECT_EQ(generator->InitialSeed(), 123U);
    EXPECT_EQ(expected, CopyToCPUData(nn::function::Randn({11}, device, generator)));
}

TEST_P(GeneratorDeviceTest, StateRoundTripIgnoresGlobalNumericLocale) {
    const auto device = GetDevice();
    auto generator = MakeGenerator(device, 1234567890123ULL);
    (void)nn::function::Randn({1}, device, generator);

    std::vector<uint8_t> state;
    {
        ScopedGroupedNumericLocale locale;
        state = generator->GetState();
    }
    const auto expected = CopyToCPUData(nn::function::Randn({8}, device, generator));
    {
        ScopedGroupedNumericLocale locale;
        generator->SetState(state);
    }

    EXPECT_EQ(expected, CopyToCPUData(nn::function::Randn({8}, device, generator)));
}

TEST_P(GeneratorDeviceTest, RandomStreamsSurviveCallPartitioningAndStateRestore) {
    constexpr int64_t kTotalSize = 31;
    const auto device = GetDevice();

    for (const bool normal : {false, true}) {
        SCOPED_TRACE(normal ? "normal" : "uniform");
        const auto random_values = [&](int64_t size, const std::shared_ptr<Generator> &generator) {
            auto tensor = normal ? nn::function::Randn({size}, device, generator)
                                 : nn::function::Rand({size}, device, generator);
            return CopyToCPUData(tensor);
        };

        for (const int64_t first_size : {1, 2, 3, 16, 17, 30}) {
            SCOPED_TRACE(first_size);
            auto whole_generator = MakeGenerator(device, 123);
            auto split_generator = MakeGenerator(device, 123);

            const auto whole = random_values(kTotalSize, whole_generator);
            auto split = random_values(first_size, split_generator);
            const auto cached_state = split_generator->GetState();
            const auto tail = random_values(kTotalSize - first_size, split_generator);
            split.insert(split.end(), tail.begin(), tail.end());

            EXPECT_EQ(whole, split);
            EXPECT_EQ(whole_generator->GetState(), split_generator->GetState());

            split_generator->SetState(cached_state);
            EXPECT_EQ(tail, random_values(kTotalSize - first_size, split_generator));
        }
    }
}

TEST_P(GeneratorDeviceTest, StateChangesAfterRandomConsumption) {
    const auto device = GetDevice();
    auto generator = MakeGenerator(device, 123);
    const auto before = generator->GetState();
    (void)nn::function::Rand({7}, device, generator);
    EXPECT_NE(before, generator->GetState());
}

TEST_P(GeneratorDeviceTest, RejectsEmptyTruncatedAndWrongBackendState) {
    const auto device = GetDevice();
    auto generator = MakeGenerator(device, 123);
    const auto state = generator->GetState();
    ASSERT_GT(state.size(), 1U);

    EXPECT_DEATH(generator->SetState({}), "Invalid .* generator state");
    EXPECT_DEATH(generator->SetState(std::vector<uint8_t>(state.begin(), state.begin() + state.size() / 2)),
                 "Invalid .* generator");
    EXPECT_DEATH(generator->SetState(ReplaceStateHeader(state, "not a generator state")),
                 "Invalid .* generator state header");

    auto with_trailing_data = state;
    with_trailing_data.push_back('x');
    EXPECT_DEATH(generator->SetState(with_trailing_data), "Invalid .* generator");
}

TEST_P(GeneratorDeviceTest, RejectsStateFromDifferentBackend) {
    const auto device = GetDevice();
    auto generator = MakeGenerator(device, 123);
    auto other = device.IsCPU() ? MakeCUDAGenerator(0, 123) : MakeCPUGenerator(123);
    EXPECT_DEATH(generator->SetState(other->GetState()), "Invalid .* generator state header");
}

TEST_P(GeneratorDeviceTest, UsesAllSeedBits) {
    const auto device = GetDevice();
    constexpr uint64_t low_seed = 7;
    constexpr uint64_t high_seed = (uint64_t{1} << 63) | low_seed;
    auto low = MakeGenerator(device, low_seed);
    auto high = MakeGenerator(device, high_seed);

    EXPECT_NE(CopyToCPUData(nn::function::Rand({16}, device, low)),
              CopyToCPUData(nn::function::Rand({16}, device, high)));
}

} // namespace infini_train::test

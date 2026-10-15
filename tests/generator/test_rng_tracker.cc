#include <array>
#include <barrier>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/rng_tracker.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

using namespace infini_train;
using infini_train::test::TensorBytes;

namespace {

using Samples = std::vector<std::vector<uint8_t>>;

Samples SampleScript(const Device &device, std::optional<Generator> generator = std::nullopt) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{33}, DataType::kFLOAT32, device);
    input->Fill(1.0f);
    return {
        TensorBytes(nn::function::Rand({33}, DataType::kFLOAT32, device, generator)),
        TensorBytes(nn::function::Randn({33}, DataType::kFLOAT32, device, generator)),
        TensorBytes(nn::function::Randn({33}, DataType::kFLOAT64, device, generator)),
        TensorBytes(nn::function::Dropout(input, 0.25, true, generator)),
    };
}

std::array<Samples, 2> InterleavedSamples(const Device &device, bool reverse) {
    std::array<Samples, 2> results;
    std::array<std::exception_ptr, 2> errors;
    std::barrier ready(3);
    std::binary_semaphore first_gate(0);
    std::binary_semaphore second_gate(0);
    std::array<std::binary_semaphore *, 2> gates{&first_gate, &second_gate};
    std::binary_semaphore finished(0);

    auto worker = [&](int index) {
        std::unique_ptr<RngScope> scope;
        try {
            scope = std::make_unique<RngScope>(9100 + index);
        } catch (...) { errors[index] = std::current_exception(); }
        ready.arrive_and_wait();
        for (int step = 0; step < 3; ++step) {
            gates[index]->acquire();
            if (!errors[index]) {
                try {
                    results[index].push_back(TensorBytes(nn::function::Rand({17}, DataType::kFLOAT32, device)));
                } catch (...) { errors[index] = std::current_exception(); }
            }
            finished.release();
        }
    };

    std::thread first(worker, 0);
    std::thread second(worker, 1);
    ready.arrive_and_wait();
    for (int step = 0; step < 3; ++step) {
        for (int order = 0; order < 2; ++order) {
            gates[reverse ? 1 - order : order]->release();
            finished.acquire();
        }
    }
    first.join();
    second.join();
    for (const auto &error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return results;
}

} // namespace

class RngTrackerTest : public infini_train::test::InfiniTrainTest {};

TEST_P(RngTrackerTest, IndependentTrackersRetainTheirOwnSequences) {
    const Device device = GetDevice();
    RngTracker first(9001);
    RngTracker second(9001);
    auto reference = CreateGenerator(device, 9001);
    const auto expected_first = SampleScript(device, reference);
    const auto expected_second = SampleScript(device, reference);
    std::optional<Generator> first_generator;
    {
        RngScope scope(first);
        first_generator = GetDefaultGenerator(device);
        EXPECT_EQ(expected_first, SampleScript(device));
        EXPECT_EQ(expected_second, SampleScript(device));
    }
    {
        RngScope scope(second);
        EXPECT_NE(*first_generator, GetDefaultGenerator(device));
        EXPECT_EQ(expected_first, SampleScript(device));
    }
    {
        RngScope scope(second);
        EXPECT_EQ(expected_second, SampleScript(device));
    }
}

TEST_P(RngTrackerTest, OwningScopesStartFreshAndLeaveGlobalStateUntouched) {
    const Device device = GetDevice();
    ManualSeed(9002);
    Generator global = GetDefaultGenerator(device);
    const auto before = TensorBytes(global.get_state());
    const auto expected = SampleScript(device, CreateGenerator(device));
    for (int attempt = 0; attempt < 2; ++attempt) {
        RngScope scope;
        EXPECT_NE(global, GetDefaultGenerator(device));
        EXPECT_EQ(expected, SampleScript(device));
        ManualSeed(9003);
        EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9003)), SampleScript(device));
    }
    EXPECT_EQ(global, GetDefaultGenerator(device));
    EXPECT_EQ(before, TensorBytes(global.get_state()));
    {
        RngScope scope(9004);
        EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9004)), SampleScript(device));
    }
    EXPECT_EQ(before, TensorBytes(global.get_state()));
}

TEST_P(RngTrackerTest, NestedScopesRestoreSelectionAfterExceptions) {
    const Device device = GetDevice();
    RngTracker training(9005);
    RngTracker evaluation(9006);
    auto reference = CreateGenerator(device, 9005);
    RngScope outer(training);
    Generator outer_generator = GetDefaultGenerator(device);
    EXPECT_EQ(SampleScript(device, reference), SampleScript(device));
    EXPECT_THROW(
        {
            RngScope inner(evaluation);
            EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9006)), SampleScript(device));
            throw std::runtime_error("leave evaluation");
        },
        std::runtime_error);
    EXPECT_EQ(outer_generator, GetDefaultGenerator(device));
    EXPECT_EQ(SampleScript(device, reference), SampleScript(device));
}

TEST_P(RngTrackerTest, ManualSeedAffectsOnlyTheSelectedStream) {
    const Device device = GetDevice();
    RngTracker tracker(9007);
    tracker.Add("evaluation", 9008);
    RngScope outer(tracker);
    auto reference = CreateGenerator(device, 9007);
    EXPECT_EQ(SampleScript(device, reference), SampleScript(device));
    const auto default_state = TensorBytes(GetDefaultGenerator(device).get_state());
    {
        RngScope inner(tracker, "evaluation");
        SampleScript(device);
        ManualSeed(9009);
        EXPECT_EQ(GetDefaultGenerator(device).current_seed(), 9009U);
        EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9009)), SampleScript(device));
        ManualSeed(9009);
        EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9009)), SampleScript(device));
    }
    EXPECT_EQ(default_state, TensorBytes(GetDefaultGenerator(device).get_state()));
    EXPECT_EQ(SampleScript(device, reference), SampleScript(device));

    tracker.ManualSeed("evaluation", 9010);
    {
        RngScope inner(tracker, "evaluation");
        EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9010)), SampleScript(device));
    }
    tracker.ManualSeed(9011);
    EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9011)), SampleScript(device));
}

TEST_P(RngTrackerTest, SeedAndSnapshotsCoverLazilyCreatedDevices) {
    const Device device = GetDevice();
    RngTracker tracker(9012);
    RngScope scope(tracker);
    ManualSeed(9013);
    const auto snapshot = tracker.GetState();
    const auto expected = SampleScript(device, CreateGenerator(device, 9013));
    EXPECT_EQ(expected, SampleScript(device));
    ManualSeed(9014);
    SampleScript(device);
    tracker.SetState(*snapshot);
    EXPECT_EQ(expected, SampleScript(device));
    tracker.SetState(*snapshot);
    EXPECT_EQ(expected, SampleScript(device));
}

TEST_P(RngTrackerTest, CopiedStateRestoresAllStreamsAfterTrackerDestruction) {
    const Device device = GetDevice();
    const std::string evaluation_name = "evaluation";
    std::shared_ptr<Tensor> exported;
    Samples expected_default;
    Samples expected_evaluation;
    {
        RngTracker original(9030);
        original.Add(evaluation_name, 9031);
        original.Add("unused", 9032);
        RngScope outer(original);
        EXPECT_EQ(GetDefaultGenerator(device).current_seed(), 9030U);
        SampleScript(device);
        {
            RngScope inner(original, evaluation_name);
            EXPECT_EQ(GetDefaultGenerator(device).current_seed(), 9031U);
            SampleScript(device);
        }
        exported = original.GetState();
        expected_default = SampleScript(device);
        RngScope inner(original, evaluation_name);
        expected_evaluation = SampleScript(device);
    }
    ASSERT_NE(exported, nullptr);
    EXPECT_TRUE(exported->GetDevice().IsCPU());
    EXPECT_EQ(exported->Dtype(), DataType::kUINT8);
    ASSERT_EQ(exported->Dims().size(), 1U);
    EXPECT_GT(exported->NumElements(), 0U);

    auto loaded = std::make_shared<Tensor>(exported->Dims(), exported->Dtype(), Device());
    loaded->CopyFrom(*exported);
    EXPECT_EQ(TensorBytes(loaded), TensorBytes(exported));

    RngTracker restored(0);
    restored.SetState(*loaded);
    EXPECT_EQ(TensorBytes(restored.GetState()), TensorBytes(exported));
    loaded->Fill(0);
    EXPECT_TRUE(restored.Contains(evaluation_name));
    EXPECT_TRUE(restored.Contains("unused"));
    RngScope outer(restored);
    EXPECT_EQ(expected_default, SampleScript(device));
    {
        RngScope inner(restored, evaluation_name);
        EXPECT_EQ(expected_evaluation, SampleScript(device));
    }
    {
        RngScope inner(restored, "unused");
        EXPECT_EQ(GetDefaultGenerator(device).current_seed(), 9032U);
        EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9032)), SampleScript(device));
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
        restored.SetState(*exported);
        EXPECT_EQ(expected_default, SampleScript(device));
        RngScope inner(restored, evaluation_name);
        EXPECT_EQ(expected_evaluation, SampleScript(device));
    }
}

TEST_P(RngTrackerTest, InvalidSerializedStatePreservesLiveGenerators) {
    const Device device = GetDevice();
    RngTracker tracker(9033);
    RngScope scope(tracker);
    SampleScript(device);
    const Generator selected = GetDefaultGenerator(device);
    const auto before = TensorBytes(selected.get_state());
    const auto valid = tracker.GetState();
    const auto bytes = TensorBytes(valid);
    auto reference = selected.Clone();
    auto expect_rejected = [&](const Tensor &state) {
        EXPECT_THROW(tracker.SetState(state), std::invalid_argument);
        EXPECT_EQ(selected, GetDefaultGenerator(device));
        EXPECT_EQ(before, TensorBytes(selected.get_state()));
        EXPECT_EQ(bytes, TensorBytes(tracker.GetState()));
    };
    expect_rejected(Tensor({1}, DataType::kFLOAT32, Device()));
    expect_rejected(Tensor(*valid, 0, {static_cast<int64_t>(valid->NumElements() - 1)}));
    EXPECT_EQ(SampleScript(device, reference), SampleScript(device));
}

TEST_P(RngTrackerTest, InvalidStreamOperationsPreserveTheBinding) {
    const Device device = GetDevice();
    RngTracker tracker(9017);
    EXPECT_TRUE(tracker.Contains("default"));
    EXPECT_FALSE(tracker.Contains("missing"));
    EXPECT_THROW(tracker.Add("", 1), std::invalid_argument);
    EXPECT_THROW(tracker.Add("default", 1), std::invalid_argument);
    EXPECT_THROW(tracker.ManualSeed("missing", 1), std::invalid_argument);
    const auto snapshot = tracker.GetState();
    tracker.Add("later", 9018);
    EXPECT_TRUE(tracker.Contains("later"));
    RngScope outer(tracker);
    Generator selected = GetDefaultGenerator(device);
    EXPECT_THROW(RngScope missing(tracker, "missing"), std::invalid_argument);
    EXPECT_EQ(selected, GetDefaultGenerator(device));
    {
        RngScope inner(tracker, "later");
        EXPECT_THROW(tracker.SetState(*snapshot), std::invalid_argument);
        EXPECT_EQ(GetDefaultGenerator(device).current_seed(), 9018U);
        RngScope other(9019);
        EXPECT_THROW(tracker.SetState(*snapshot), std::invalid_argument);
    }
    tracker.SetState(*snapshot);
    EXPECT_FALSE(tracker.Contains("later"));
    EXPECT_EQ(GetDefaultGenerator(device).current_seed(), 9017U);
}

TEST_P(RngTrackerTest, ExplicitGeneratorTakesPriorityOverTheScope) {
    const Device device = GetDevice();
    RngScope scope(9019);
    const auto before = TensorBytes(GetDefaultGenerator(device).get_state());
    auto explicit_generator = CreateGenerator(device, 9020);
    EXPECT_NE(explicit_generator, GetDefaultGenerator(device));
    EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9020)), SampleScript(device, explicit_generator));
    EXPECT_EQ(before, TensorBytes(GetDefaultGenerator(device).get_state()));
    const auto explicit_state = TensorBytes(explicit_generator.get_state());
    ManualSeed(9022);
    EXPECT_EQ(explicit_state, TensorBytes(explicit_generator.get_state()));
    EXPECT_EQ(SampleScript(device, CreateGenerator(device, 9022)), SampleScript(device));
}

TEST_P(RngTrackerTest, ChildThreadsDoNotInheritTheParentScope) {
    const Device device = GetDevice();
    const Generator global = GetDefaultGenerator(device);
    RngScope parent(9023);
    const Generator scoped = GetDefaultGenerator(device);
    EXPECT_NE(global, scoped);
    auto child = std::async(std::launch::async, [&] { return GetDefaultGenerator(device); });
    EXPECT_EQ(global, child.get());
    EXPECT_EQ(scoped, GetDefaultGenerator(device));
}

TEST_P(RngTrackerTest, ReversingThreadInterleavingPreservesEachSequence) {
    const Device device = GetDevice();
    const auto forward = InterleavedSamples(device, false);
    const auto reverse = InterleavedSamples(device, true);
    EXPECT_EQ(forward, reverse);
    for (int index = 0; index < 2; ++index) {
        auto reference = CreateGenerator(device, 9100 + index);
        for (int step = 0; step < 3; ++step) {
            EXPECT_EQ(forward[index][step],
                      TensorBytes(nn::function::Rand({17}, DataType::kFLOAT32, device, reference)));
        }
    }
}

TEST_P(RngTrackerTest, IndependentTrackersCanSampleConcurrently) {
    const Device device = GetDevice();
    std::barrier ready(2);
    auto worker = [&](uint64_t seed) {
        RngScope scope(seed);
        ready.arrive_and_wait();
        return SampleScript(device);
    };
    auto first = std::async(std::launch::async, worker, 9120);
    auto second = std::async(std::launch::async, worker, 9121);
    const auto first_samples = first.get();
    const auto second_samples = second.get();
    EXPECT_EQ(first_samples, SampleScript(device, CreateGenerator(device, 9120)));
    EXPECT_EQ(second_samples, SampleScript(device, CreateGenerator(device, 9121)));
}

TEST_P(RngTrackerTest, TrackerRejectsConcurrentBindingAndCanMoveBetweenThreads) {
    const Device device = GetDevice();
    RngTracker tracker(9021);
    auto reference = CreateGenerator(device, 9021);
    {
        RngScope outer(tracker);
        EXPECT_EQ(SampleScript(device, reference), SampleScript(device));
        {
            RngScope inner(tracker);
            EXPECT_EQ(SampleScript(device, reference), SampleScript(device));
        }
        auto conflicting_scope = std::async(std::launch::async, [&] {
            try {
                RngScope scope(tracker);
            } catch (const std::logic_error &) { return true; }
            return false;
        });
        EXPECT_TRUE(conflicting_scope.get());
        auto conflicting_seed = std::async(std::launch::async, [&] {
            try {
                tracker.ManualSeed(0);
            } catch (const std::logic_error &) { return true; }
            return false;
        });
        EXPECT_TRUE(conflicting_seed.get());
    }
    auto next_thread = std::async(std::launch::async, [&] {
        RngScope scope(tracker);
        return SampleScript(device);
    });
    EXPECT_EQ(SampleScript(device, reference), next_thread.get());
}

INFINI_TRAIN_REGISTER_TEST(RngTrackerTest);

#include <array>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/rng_tracker.h"

#include "tests/common/test_utils.h"

namespace infini_train {
namespace {

using nn::parallel::ColumnParallelLinear;
using nn::parallel::kTensorParallelInitRngName;
using nn::parallel::RowParallelLinear;
using test::TensorBytes;

struct ModelParameters {
    std::vector<uint8_t> embedding;
    std::vector<uint8_t> column_weight;
    std::vector<uint8_t> column_bias;
    std::vector<uint8_t> row_weight;
    std::vector<uint8_t> row_bias;

    bool operator==(const ModelParameters &) const = default;
};

ModelParameters InitializeModel() {
    nn::Embedding embedding(16, 8);
    ColumnParallelLinear column(8, 16, true, false, false, false);
    RowParallelLinear row(16, 8, true, true, true, false);
    return {TensorBytes(embedding.parameter(nn::Embedding::kParamWeightName)),
            TensorBytes(column.parameter(ColumnParallelLinear::kParamWeightName)),
            TensorBytes(column.parameter(ColumnParallelLinear::kParamBiasName)),
            TensorBytes(row.parameter(RowParallelLinear::kParamWeightName)),
            TensorBytes(row.parameter(RowParallelLinear::kParamBiasName))};
}

std::array<ModelParameters, 4> InitializeRanks(const std::array<int, 4> &order) {
    std::array<ModelParameters, 4> parameters;
    std::array<std::promise<void>, 4> start;
    std::array<std::promise<void>, 4> done;
    std::array<std::thread, 4> threads;
    for (int rank = 0; rank < 4; ++rank) {
        threads[rank] = std::thread([&, rank] {
            RngTracker tracker(42);
            tracker.Add(kTensorParallelInitRngName, 42 + 2718 + rank % 2);
            RngScope scope(tracker);
            start[rank].get_future().wait();
            parameters[rank] = InitializeModel();
            done[rank].set_value();
        });
    }
    for (int rank : order) {
        start[rank].set_value();
        done[rank].get_future().wait();
    }
    for (auto &thread : threads) { thread.join(); }
    return parameters;
}

TEST(RngTensorParallelTest, ShardsDifferAndReplicatedParametersMatch) {
    const auto parameters = InitializeRanks({0, 1, 2, 3});
    EXPECT_EQ(parameters[0], parameters[2]);
    EXPECT_EQ(parameters[1], parameters[3]);
    EXPECT_EQ(parameters[0].embedding, parameters[1].embedding);
    EXPECT_EQ(parameters[0].row_bias, parameters[1].row_bias);
    EXPECT_NE(parameters[0].column_weight, parameters[1].column_weight);
    EXPECT_NE(parameters[0].column_bias, parameters[1].column_bias);
    EXPECT_NE(parameters[0].row_weight, parameters[1].row_weight);
}

TEST(RngTensorParallelTest, ThreadOrderDoesNotChangeInitialization) {
    const auto forward = InitializeRanks({0, 1, 2, 3});
    const auto reverse = InitializeRanks({3, 2, 1, 0});
    EXPECT_EQ(forward, reverse);
}

TEST(RngTensorParallelTest, ConsecutiveLayersContinueTheShardedStream) {
    RngTracker tracker(42);
    tracker.Add(kTensorParallelInitRngName, 42 + 2718);
    RngScope scope(tracker);
    ColumnParallelLinear first(8, 16, true, false, false, false);
    ColumnParallelLinear second(8, 16, true, false, false, false);
    EXPECT_NE(TensorBytes(first.parameter(ColumnParallelLinear::kParamWeightName)),
              TensorBytes(second.parameter(ColumnParallelLinear::kParamWeightName)));
}

TEST(RngTensorParallelTest, MissingNamedStreamUsesTheCurrentDefaultStream) {
    ModelParameters first;
    {
        RngScope scope(42);
        first = InitializeModel();
    }
    ModelParameters replay;
    {
        RngScope scope(42);
        replay = InitializeModel();
    }
    EXPECT_EQ(first, replay);
}

TEST(RngTensorParallelTest, UnscopedInitializationUsesTheProcessDefault) {
    const auto &generator = GetDefaultGenerator(Device());
    generator.set_current_seed(42);
    const auto first = InitializeModel();
    generator.set_current_seed(42);
    EXPECT_EQ(first, InitializeModel());
}

} // namespace
} // namespace infini_train

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    infini_train::nn::parallel::global::InitAllEnv(4, 2, false, 1, 1);
    return RUN_ALL_TESTS();
}

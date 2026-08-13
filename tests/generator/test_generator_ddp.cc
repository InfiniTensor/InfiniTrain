#include <algorithm>
#include <barrier>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"
#include "tests/generator/generator_test_utils.h"

namespace infini_train::test {
namespace {

constexpr int kNumRanks = 2;
constexpr char kRankBufferName[] = "rank_buffer";

class StatefulLinear : public nn::Linear {
public:
    explicit StatefulLinear(float rank_value) : nn::Linear(4, 3, true, Device()) {
        parameters_[nn::Linear::kParamBiasName]->set_requires_grad(false);
        auto rank_buffer = std::make_shared<Tensor>(std::vector<int64_t>{5}, DataType::kFLOAT32, Device());
        rank_buffer->Fill(rank_value);
        buffers_[kRankBufferName] = std::move(rank_buffer);
    }
};

std::vector<float> FlattenState(const std::shared_ptr<nn::Module> &module) {
    auto state = module->StateDict();
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> named_tensors(state.begin(), state.end());
    std::sort(named_tensors.begin(), named_tensors.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::vector<float> values;
    for (const auto &[_, tensor] : named_tensors) {
        auto tensor_values = CopyToCPUData(tensor);
        values.insert(values.end(), tensor_values.begin(), tensor_values.end());
    }
    return values;
}

void EnsureDataParallelProcessGroup(const nn::parallel::Rank &rank, Device device) {
    auto *factory = nn::parallel::ProcessGroupFactory::Instance(device.type());
    factory->GetOrCreate(nn::parallel::GetDataParallelProcessGroupName(rank.GlobalRank()),
                         nn::parallel::GetDataParallelGroupRanks(rank.GlobalRank()));
}

} // namespace

TEST(GeneratorDDPTest, SynchronizesCPUInitializedStateBeforeFirstForward) {
    REQUIRE_MIN_DEVICES(2);
    ManualSeedAll(2026);
    std::vector<std::vector<float>> pre_sync_state(kNumRanks);
    std::vector<std::vector<float>> pre_sync_frozen_bias(kNumRanks);
    std::vector<std::vector<float>> synchronized_state(kNumRanks);
    std::vector<std::vector<float>> synchronized_buffers(kNumRanks);
    std::vector<std::vector<float>> forward_outputs(kNumRanks);
    std::vector<Tensor *> pre_sync_weights(kNumRanks);
    std::vector<Tensor *> synchronized_weights(kNumRanks);
    std::vector<std::shared_ptr<nn::Module>> synchronized_modules(kNumRanks);
    std::barrier models_ready(kNumRanks);
    std::vector<std::thread> threads;

    for (int thread_rank = 0; thread_rank < kNumRanks; ++thread_rank) {
        threads.emplace_back([&, thread_rank]() {
            nn::parallel::Rank rank(0, thread_rank, 1, kNumRanks);
            nn::parallel::global::thread_global_rank = rank.GlobalRank();
            const Device device(Device::DeviceType::kCUDA, thread_rank);

            auto module = std::make_shared<StatefulLinear>(static_cast<float>(thread_rank + 1));
            module->To(device);
            auto state = module->StateDict();
            pre_sync_state[thread_rank] = FlattenState(module);
            pre_sync_frozen_bias[thread_rank] = CopyToCPUData(state.at(nn::Linear::kParamBiasName));
            pre_sync_weights[thread_rank] = state.at(nn::Linear::kParamWeightName).get();
            models_ready.arrive_and_wait();

            EnsureDataParallelProcessGroup(rank, device);

            nn::parallel::DistributedDataParallelConfig config;
            auto ddp = std::make_shared<nn::parallel::DistributedDataParallel>(module, rank, config);
            state = module->StateDict();
            synchronized_state[thread_rank] = FlattenState(module);
            synchronized_buffers[thread_rank] = CopyToCPUData(state.at(kRankBufferName));
            synchronized_weights[thread_rank] = state.at(nn::Linear::kParamWeightName).get();
            synchronized_modules[thread_rank] = module;

            auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kFLOAT32, device);
            input->Fill(0.25f);
            auto output = (*ddp)({input})[0];
            forward_outputs[thread_rank] = CopyToCPUData(output);
        });
    }
    for (auto &thread : threads) { thread.join(); }

    ASSERT_FALSE(pre_sync_state[0].empty());
    ASSERT_FALSE(pre_sync_state[1].empty());
    EXPECT_NE(pre_sync_state[0], pre_sync_state[1]);
    EXPECT_NE(pre_sync_frozen_bias[0], pre_sync_frozen_bias[1]);
    for (int rank = 0; rank < kNumRanks; ++rank) {
        EXPECT_EQ(synchronized_state[rank], pre_sync_state[0]);
        EXPECT_EQ(synchronized_buffers[rank], std::vector<float>(5, 1.0f));
        EXPECT_EQ(synchronized_weights[rank], pre_sync_weights[rank]);
        EXPECT_FALSE(synchronized_modules[rank]->StateDict().at(nn::Linear::kParamBiasName)->requires_grad());
    }
    EXPECT_EQ(forward_outputs[0], forward_outputs[1]);
}

TEST(GeneratorDDPTest, DoesNotRebroadcastParametersAfterInitialSynchronization) {
    REQUIRE_MIN_DEVICES(2);
    ManualSeedAll(2027);
    std::vector<std::vector<float>> post_forward_weights(kNumRanks);
    std::vector<std::vector<float>> forward_outputs(kNumRanks);
    std::barrier models_ready(kNumRanks);
    std::barrier ddp_ready(kNumRanks);
    std::barrier parameters_mutated(kNumRanks);
    std::vector<std::thread> threads;

    for (int thread_rank = 0; thread_rank < kNumRanks; ++thread_rank) {
        threads.emplace_back([&, thread_rank]() {
            nn::parallel::Rank rank(0, thread_rank, 1, kNumRanks);
            nn::parallel::global::thread_global_rank = rank.GlobalRank();
            const Device device(Device::DeviceType::kCUDA, thread_rank);

            auto module = std::make_shared<StatefulLinear>(static_cast<float>(thread_rank + 1));
            module->To(device);
            models_ready.arrive_and_wait();
            EnsureDataParallelProcessGroup(rank, device);

            nn::parallel::DistributedDataParallelConfig config;
            auto ddp = std::make_shared<nn::parallel::DistributedDataParallel>(module, rank, config);
            ddp_ready.arrive_and_wait();

            auto weight = module->StateDict().at(nn::Linear::kParamWeightName);
            weight->Fill(static_cast<float>(thread_rank + 3));
            parameters_mutated.arrive_and_wait();

            auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kFLOAT32, device);
            input->Fill(0.25f);
            auto output = (*ddp)({input})[0];
            forward_outputs[thread_rank] = CopyToCPUData(output);
            post_forward_weights[thread_rank] = CopyToCPUData(weight);
        });
    }
    for (auto &thread : threads) { thread.join(); }

    EXPECT_EQ(post_forward_weights[0], std::vector<float>(12, 3.0f));
    EXPECT_EQ(post_forward_weights[1], std::vector<float>(12, 4.0f));
    EXPECT_NE(forward_outputs[0], forward_outputs[1]);
}

} // namespace infini_train::test

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    infini_train::nn::parallel::global::GlobalEnv::Instance().Init(2, 1, false, 1, 1);
    return RUN_ALL_TESTS();
}

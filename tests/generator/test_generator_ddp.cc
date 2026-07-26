#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include <cuda_runtime_api.h>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {

std::vector<float> CopyValuesToCPU(const std::shared_ptr<Tensor> &tensor) {
    Tensor host(tensor->Dims(), tensor->Dtype(), Device());
    host.CopyFrom(*tensor);
    core::GetDeviceGuardImpl(tensor->GetDevice().type())->SynchronizeDevice(tensor->GetDevice());

    const auto *values = static_cast<const float *>(host.DataPtr());
    return {values, values + host.NumElements()};
}

TEST(DistributedDataParallelTest, SynchronizesParametersFromRankZeroAcrossAvailableGPUs) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "requires at least 2 GPUs (found " << device_count << ")";
    }

    const int world_size = nn::parallel::global::GetDataParallelSize();
    ASSERT_EQ(world_size, device_count);

    auto *factory = nn::parallel::ProcessGroupFactory::Instance(Device::DeviceType::kCUDA);
    factory->GetOrCreate(nn::parallel::GetDataParallelProcessGroupName(0), nn::parallel::GetDataParallelGroupRanks(0));

    std::vector<std::vector<float>> synchronized_values(static_cast<size_t>(world_size));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(world_size));
    for (int thread_rank = 0; thread_rank < world_size; ++thread_rank) {
        workers.emplace_back([&, thread_rank] {
            nn::parallel::global::thread_global_rank = thread_rank;
            const Device device(Device::DeviceType::kCUDA, thread_rank);
            core::DeviceGuard guard(device);

            auto module = std::make_shared<nn::Linear>(8, 4);
            module->To(device);
            for (const auto &parameter : module->Parameters()) { parameter->Fill(static_cast<float>(thread_rank + 1)); }

            const nn::parallel::Rank rank(0, thread_rank, 1, world_size);
            auto ddp = std::make_shared<nn::parallel::DistributedDataParallel>(
                std::move(module), rank, nn::parallel::DistributedDataParallelConfig{});

            for (const auto &parameter : ddp->module()->Parameters()) {
                auto values = CopyValuesToCPU(parameter);
                synchronized_values[static_cast<size_t>(thread_rank)].insert(
                    synchronized_values[static_cast<size_t>(thread_rank)].end(), values.begin(), values.end());
            }
        });
    }
    for (auto &worker : workers) { worker.join(); }

    ASSERT_FALSE(synchronized_values.front().empty());
    for (const auto &rank_values : synchronized_values) {
        EXPECT_EQ(rank_values, synchronized_values.front());
        for (float value : rank_values) { EXPECT_FLOAT_EQ(value, 1.0f); }
    }
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
        device_count = 1;
    }
    nn::parallel::global::GlobalEnv::Instance().Init(device_count, 1, false, 1, 1);

    return RUN_ALL_TESTS();
}

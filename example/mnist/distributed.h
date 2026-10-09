#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "example/mnist/training.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"

namespace mnist {
// One process per GPU, launched with infini_run. Reuse the framework's NCCL
// process group, parameter broadcast and autograd-driven DDP gradient reducer.
class DistributedContext {
public:
    DistributedContext(bool ddp, bool cuda) {
        using namespace infini_train;
        using namespace nn::parallel;
        global::InitAllEnv(1, 1, false, 1, 1);
        world_size = global::GetWorldSize();
        rank = global::GetGlobalProcRank();
        global::thread_global_rank = rank;
        CHECK(ddp || world_size == 1) << "Multi-process launch requires --ddp=true";
        CHECK(!ddp || cuda) << "DDP requires --device=cuda";
#if !defined(USE_CUDA) || !defined(USE_NCCL)
        CHECK(!ddp) << "DDP requires a build with USE_CUDA=ON and USE_NCCL=ON";
#endif
        device = cuda ? Device(Device::DeviceType::kCUDA, global::GetLocalProcRank()) : Device();
        if (cuda) {
            auto *runtime = core::GetDeviceGuardImpl(device.type());
            CHECK_LT(device.index(), runtime->DeviceCount()) << "Not enough visible GPUs for LOCAL_RANK";
            runtime->SetDevice(device);
        }
        if (ddp && world_size > 1) {
            CHECK_EQ(global::GetNnodes(), 1) << "This MNIST demo supports single-node DDP";
            group = ProcessGroupFactory::Instance(device.type())
                        ->GetOrCreate(GetDataParallelProcessGroupName(rank), global::GetGroupRanks(global::DP, rank));
        }
    }

    std::shared_ptr<infini_train::nn::Module> Wrap(const std::shared_ptr<MNIST> &net, bool buckets) const {
        using namespace infini_train::nn::parallel;
        if (!group) {
            return net;
        }
        // Do not rely on coincidentally identical per-rank RNG state.
        group->Broadcast(net->Parameters(), 0);
        DistributedDataParallelConfig config;
        config.gradient_bucketing_enabled = buckets;
        config.average_in_collective = true;
        config.zero_stage = 0;
        return std::make_shared<DistributedDataParallel>(net, Rank(rank, 0, global::GetNprocPerNode(), 1), config);
    }

    Metrics Sum(const Metrics &local) const {
        using namespace infini_train;
        if (!group) {
            return local;
        }
        // Sum sufficient statistics, never unweighted per-rank averages.
        auto cpu = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT64);
        const double values[]
            = {local.loss_sum, static_cast<double>(local.correct), static_cast<double>(local.samples)};
        std::memcpy(cpu->DataPtr(), values, sizeof(values));
        auto tensor = std::make_shared<Tensor>(cpu->To(device));
        group->AllReduce(tensor, nn::parallel::function::ReduceOpType::kSum);
        auto result = tensor->To(Device());
        core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
        const auto *v = static_cast<const double *>(result.DataPtr());
        return {v[0], static_cast<int64_t>(v[1]), static_cast<int64_t>(v[2])};
    }

    void Barrier() const { (void)Sum(Metrics{}); }

    void VerifyReplicas(const MNIST &net) const {
        using namespace infini_train;
        if (!group) {
            return;
        }
        for (const auto &[name, parameter] : net.NamedParameters()) {
            auto root = std::make_shared<Tensor>(parameter->Dims(), parameter->Dtype(), device);
            root->CopyFrom(parameter);
            group->Broadcast({root}, 0);
            auto actual = parameter->To(Device()), expected = root->To(Device());
            core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
            const auto *values = static_cast<const float *>(actual.DataPtr());
            for (size_t i = 0; i < actual.NumElements(); ++i) { CHECK(std::isfinite(values[i])) << name; }
            CHECK_EQ(std::memcmp(actual.DataPtr(), expected.DataPtr(), actual.SizeInBytes()), 0)
                << "DDP replica differs from rank 0: " << name;
        }
    }

    int rank = 0;
    int world_size = 1;
    infini_train::Device device;
    const infini_train::nn::parallel::ProcessGroup *group = nullptr;
};
} // namespace mnist

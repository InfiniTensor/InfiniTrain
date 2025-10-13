#pragma once

#include "infini_train/include/nn/parallel/global.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include "infini_train/include/common/cuda/common_cuda.h"
#endif
#include "infini_train/include/nn/parallel/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
namespace infini_train {
class Tensor;
class Device;
namespace nn {
class Module;
}

} // namespace infini_train

namespace infini_train::nn::parallel {
class ProcessGroup;

class ProcessGroup {
public:
    explicit ProcessGroup(int comm_size);

    explicit ProcessGroup(const std::vector<int> &device_indices);

    void AllReduce(const std::shared_ptr<Tensor> &tensor, function::ReduceOpType reduce_op) const;

    ncclComm_t comm(int idx) const;

private:
    std::vector<ncclComm_t> comms_;
};

class ProcessGroupFactory {
public:
    static constexpr char kDefaltProcessGroupName[] = "default";

    static ProcessGroupFactory *Instance();

    void Create(const std::string &name, int comm_size);

    const ProcessGroup *Get(const std::string &name) const;

    const ProcessGroup *GetDefaultProcessGroup() const;

    void InitProcessGroup(int world_size);

private:
    ProcessGroupFactory();
    // TODO(dcj): maybe RWLock later?
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<ProcessGroup>> name_to_group_;
};

inline std::string GetTensorParallelProcessFactoryName(const nn::parallel::DistributedDataParallel::Rank &rank,
                                                       int tp_size) {
    return "TP" + std::to_string(rank.thread_rank() % tp_size);
}
} // namespace infini_train::nn::parallel

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train {
class Tensor;
class Device;
namespace nn {
class Module;
}

} // namespace infini_train

namespace infini_train::nn::parallel {

#ifdef USE_NCCL
class ProcessGroup {
public:
    explicit ProcessGroup(const std::vector<int> &device_indices);

    int GetGroupRank(int thread_rank) const;

    void AllReduce(const std::shared_ptr<Tensor> &tensor, function::ReduceOpType reduce_op) const;

    std::vector<std::shared_ptr<Tensor>> BroadCast(const std::vector<std::shared_ptr<Tensor>> &input_tensors) const;

    std::vector<std::shared_ptr<Tensor>>
    ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads, const Device *destination) const;

    std::vector<std::shared_ptr<Tensor>> Scatter(const std::shared_ptr<Tensor> &tensor,
                                                 std::vector<const Device *> devices, int64_t dim) const;

    std::shared_ptr<Tensor> Gather(const std::vector<std::shared_ptr<Tensor>> &tensors, const Device *destination,
                                   int64_t dim) const;

private:
    std::vector<ncclComm_t> comms_;
    std::vector<const Device *> devices_;

    std::unordered_map<const Device *, ncclComm_t> device_comm_map_;
    std::unordered_map<int, int> thread_group_rank_map_; // thread_rank : group_rank

    int comm_size_ = 0;
};
#endif

class ProcessGroupFactory {
public:
    static constexpr char kDefaltProcessGroupName[] = "default";

    static ProcessGroupFactory *Instance();

    const ProcessGroup *GetOrCreate(const std::string &name, int comm_size);

    const ProcessGroup *GetOrCreate(const std::string &name, const std::vector<int> &device_indices);

    const ProcessGroup *Get(const std::string &name) const;

    const ProcessGroup *GetDefaultProcessGroup() const;

private:
    ProcessGroupFactory();
    // TODO(dcj): maybe RWLock later?
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<ProcessGroup>> name_to_group_;
};
} // namespace infini_train::nn::parallel

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train {
class Tensor;
namespace core {
class CclComm;
class Stream;
} // namespace core
namespace nn {
class Module;
namespace parallel {
class Work;
} // namespace parallel
} // namespace nn

} // namespace infini_train

namespace infini_train::nn::parallel {

class ProcessGroup {
public:
    explicit ProcessGroup(Device::DeviceType backend, const std::string &process_group_name,
                          const std::vector<int> &device_indices);

    virtual ~ProcessGroup();

    virtual int GetGroupRank(int global_rank) const;

    // Asynchronous communication APIs (Compute / Communication stream decoupled)
    virtual std::shared_ptr<Work> AllReduce(const std::shared_ptr<Tensor> &tensor,
                                            function::ReduceOpType reduce_op = function::ReduceOpType::kSum,
                                            bool async_op = false) const;

    virtual std::shared_ptr<Work> AllGather(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                            bool async_op = false) const;

    virtual std::shared_ptr<Work> ReduceScatter(const std::shared_ptr<Tensor> &output,
                                                const std::shared_ptr<Tensor> &input,
                                                function::ReduceOpType reduce_op = function::ReduceOpType::kSum,
                                                bool async_op = false) const;

    virtual std::shared_ptr<Work> Send(std::vector<std::shared_ptr<Tensor>> tensors, int dest_rank,
                                       bool async_op = false) const;

    virtual std::shared_ptr<Work> Recv(std::vector<std::shared_ptr<Tensor>> tensors, int src_rank,
                                       bool async_op = false) const;

    // Legacy communication APIs (Single-stream)
    virtual std::vector<std::shared_ptr<Tensor>>
    BroadCast(const std::vector<std::shared_ptr<Tensor>> &input_tensors) const;

    virtual std::vector<std::shared_ptr<Tensor>>
    ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads, Device destination) const;

    virtual std::vector<std::shared_ptr<Tensor>> Scatter(const std::shared_ptr<Tensor> &tensor,
                                                         std::vector<Device> devices, int64_t dim) const;

    virtual std::shared_ptr<Tensor> Gather(const std::vector<std::shared_ptr<Tensor>> &tensors, Device destination,
                                           int64_t dim) const;

protected:
    ProcessGroup(int world_size, const std::string &name);

    void InitSingleProcess(const std::vector<int> &ranks);

    void InitMultiProcess(const std::vector<int> &ranks);

    void InitStreams();

    std::vector<Device> devices_;

    std::unordered_map<int, int> global_group_rank_map_; // global_rank : group_rank

    int world_size_ = 0;

    const std::string name_ = "";

    bool is_main_process_ = false;
    Device::DeviceType backend_ = Device::DeviceType::kInvalid;

    std::vector<std::unique_ptr<core::CclComm>> comms_;
    std::vector<std::unique_ptr<core::Stream>> comm_streams_;
    std::unordered_map<int, core::CclComm *> device_comm_map_;  // device_index : comm
    std::unordered_map<int, core::Stream *> device_stream_map_; // device_index : comm_stream
};

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

    template <typename Creator, typename = std::enable_if_t<std::is_invocable_v<Creator>>>
    const ProcessGroup *GetOrCreate(const std::string &name, Creator &&creator) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto [it, inserted] = name_to_group_.emplace(name, nullptr);
        if (!inserted) {
            while (it->second == nullptr) { cond_.wait(lock); }
            return it->second.get();
        }

        lock.unlock();
        auto new_group = creator();
        lock.lock();

        it->second = std::move(new_group);
        cond_.notify_all();
        return it->second.get();
    }

private:
    // TODO(dcj): maybe RWLock later?
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::unordered_map<std::string, std::unique_ptr<ProcessGroup>> name_to_group_;
};
} // namespace infini_train::nn::parallel

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
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
namespace parallel {
class Work;
} // namespace parallel
} // namespace nn

} // namespace infini_train

namespace infini_train::nn::parallel {

class ProcessGroup {
public:
    virtual ~ProcessGroup() = default;

    virtual int GetGroupRank(int global_rank) const;

    // Asynchronous communication APIs (Compute / Communication stream decoupled)
    virtual std::shared_ptr<Work> AllReduce(const std::shared_ptr<Tensor> &tensor,
                                            function::ReduceOpType reduce_op = function::ReduceOpType::kSum,
                                            bool async_op = false) const
        = 0;

    virtual std::shared_ptr<Work> AllGather(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                            bool async_op = false) const
        = 0;

    virtual std::shared_ptr<Work>
    ReduceScatter(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                  function::ReduceOpType reduce_op = function::ReduceOpType::kSum, bool async_op = false) const
        = 0;

    virtual std::shared_ptr<Work> Send(std::vector<std::shared_ptr<Tensor>> tensors, int dest_rank,
                                       bool async_op = false) const
        = 0;

    virtual std::shared_ptr<Work> Recv(std::vector<std::shared_ptr<Tensor>> tensors, int src_rank,
                                       bool async_op = false) const
        = 0;

    // Legacy communication APIs (Single-stream)
    virtual std::vector<std::shared_ptr<Tensor>>
    BroadCast(const std::vector<std::shared_ptr<Tensor>> &input_tensors) const = 0;

    virtual std::vector<std::shared_ptr<Tensor>>
    ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads, const Device *destination) const
        = 0;

    virtual std::vector<std::shared_ptr<Tensor>> Scatter(const std::shared_ptr<Tensor> &tensor,
                                                         std::vector<const Device *> devices, int64_t dim) const
        = 0;

    virtual std::shared_ptr<Tensor> Gather(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                           const Device *destination, int64_t dim) const
        = 0;

protected:
    ProcessGroup(int world_size, const std::string &name);

    std::vector<const Device *> devices_;

    std::unordered_map<int, int> global_group_rank_map_; // global_rank : group_rank

    int world_size_ = 0;

    const std::string name_ = "";

    bool is_main_process_ = false;
};

#ifdef USE_NCCL
class ProcessGroupNCCL final : public ProcessGroup {
public:
    explicit ProcessGroupNCCL(const std::string &process_group_name, const std::vector<int> &device_indices);

    ~ProcessGroupNCCL() override;

    // Asynchronous communication APIs (Compute / Communication stream decoupled)
    std::shared_ptr<Work> AllReduce(const std::shared_ptr<Tensor> &tensor, function::ReduceOpType reduce_op,
                                    bool async_op) const override;

    std::shared_ptr<Work> AllGather(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                    bool async_op) const override;

    std::shared_ptr<Work> ReduceScatter(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                        function::ReduceOpType reduce_op, bool async_op) const override;

    std::shared_ptr<Work> Send(std::vector<std::shared_ptr<Tensor>> tensors, int dest_rank,
                               bool async_op) const override;

    std::shared_ptr<Work> Recv(std::vector<std::shared_ptr<Tensor>> tensors, int src_rank,
                               bool async_op) const override;

    // Legacy communication APIs (Single-stream)
    std::vector<std::shared_ptr<Tensor>>
    BroadCast(const std::vector<std::shared_ptr<Tensor>> &input_tensors) const override;

    std::vector<std::shared_ptr<Tensor>>
    ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads,
                       const Device *destination) const override;

    std::vector<std::shared_ptr<Tensor>> Scatter(const std::shared_ptr<Tensor> &tensor,
                                                 std::vector<const Device *> devices, int64_t dim) const override;

    std::shared_ptr<Tensor> Gather(const std::vector<std::shared_ptr<Tensor>> &tensors, const Device *destination,
                                   int64_t dim) const override;

private:
    void InitSingleProcess(const std::vector<int> &ranks);

    void InitMultiProcess(const std::vector<int> &ranks);

    void InitStreams();

private:
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> comm_streams_;

    std::unordered_map<const Device *, ncclComm_t> device_comm_map_;
    std::unordered_map<const Device *, cudaStream_t> device_stream_map_;
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

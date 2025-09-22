#pragma once

#include <memory>
#include <string>
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
class ProcessGroup;

struct ProcessGroupConfig {
    std::string backend;     // "nccl"
    std::string init_method; // "env://", "tcp://host:port"
    int world_size = 0;
    int rank = 0;
    std::vector<int> device_list;
};

std::shared_ptr<ProcessGroup> InitProcessGroup(const ProcessGroupConfig &cfg);
void DestroyProcessGroup();

class ProcessGroup {
public:
    virtual std::vector<std::shared_ptr<Tensor>>
    Gather(const std::vector<std::vector<std::shared_ptr<Tensor>>> &outputs, const Device *target_device, int dim) = 0;
    virtual std::vector<std::vector<std::shared_ptr<Tensor>>>
    Scatter(const std::vector<std::shared_ptr<Tensor>> &input_tensors, const std::vector<const Device *> &device_ids,
            int dim)
        = 0;
    virtual void AllReduce(const std::shared_ptr<Tensor> &tensor, function::ReduceOpType reduce_op) = 0;
    virtual std::vector<std::vector<std::shared_ptr<Tensor>>>
    Broadcast(const std::vector<std::shared_ptr<Tensor>> &tensors, const std::vector<const Device *> &devices) = 0;
    virtual ~ProcessGroup() = default;
};

class ProcessGroupNCCL : public ProcessGroup {
public:
    ProcessGroupNCCL(const ProcessGroupConfig &cfg);
    ~ProcessGroupNCCL() override;
    std::vector<std::shared_ptr<Tensor>> Gather(const std::vector<std::vector<std::shared_ptr<Tensor>>> &outputs,
                                                const Device *target_device, int dim) override;
    std::vector<std::vector<std::shared_ptr<Tensor>>> Scatter(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                                              const std::vector<const Device *> &device_ids,
                                                              int dim) override;
    void AllReduce(const std::shared_ptr<Tensor> &tensor, function::ReduceOpType reduce_op) override;
    std::vector<std::vector<std::shared_ptr<Tensor>>> Broadcast(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                                                const std::vector<const Device *> &devices) override;

private:
#ifdef USE_NCCL
    ncclComm_t comm_ = nullptr;
#endif
    std::vector<int> device_list_;
    int world_size = 1;
    int rank = 0;

    static bool IsHexString(const std::string &s);
    static std::vector<uint8_t> HexToBytes(const std::string &hex);
};
} // namespace infini_train::nn::parallel

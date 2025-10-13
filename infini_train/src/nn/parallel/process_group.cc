#include "infini_train/include/nn/parallel/global.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#ifdef USE_CUDA
#include "infini_train/include/common/cuda/common_cuda.h"
#endif
#include "infini_train/include/datatype.h"
#include "infini_train/include/nn/parallel/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
class Tensor;
class Device;
namespace nn {
class Module;
}

namespace {
const std::unordered_map<DataType, ncclDataType_t> kNcclDtypeMap = {
    {DataType::kUINT8, ncclUint8},       {DataType::kINT8, ncclInt8},     {DataType::kUINT32, ncclUint32},
    {DataType::kINT32, ncclInt32},       {DataType::kUINT64, ncclUint64}, {DataType::kINT64, ncclInt64},
    {DataType::kBFLOAT16, ncclBfloat16}, {DataType::kFLOAT16, ncclHalf},  {DataType::kFLOAT32, ncclFloat32},
    {DataType::kFLOAT64, ncclFloat64},
};

using nn::parallel::function::ReduceOpType;

const std::unordered_map<ReduceOpType, ncclRedOp_t> kNcclReduceOpMap = {
    {ReduceOpType::kSum, ncclSum},
    {ReduceOpType::kProd, ncclProd},
    {ReduceOpType::kMax, ncclMax},
    {ReduceOpType::kAvg, ncclAvg},
};
} // namespace

} // namespace infini_train

namespace infini_train::nn::parallel {

ProcessGroup::ProcessGroup(int comm_size) {
    comms_.resize(comm_size);
    std::vector<int> device_indices;
    for (int i = 0; i < comm_size; i++) { device_indices.push_back(i); }
    NCCL_CHECK(ncclCommInitAll(comms_.data(), comm_size, device_indices.data()));
}

ProcessGroup::ProcessGroup(const std::vector<int> &device_indices) {
    int num_devices = device_indices.size();
    comms_.resize(num_devices);
    NCCL_CHECK(ncclCommInitAll(comms_.data(), num_devices, device_indices.data()));
}

void ProcessGroup::AllReduce(const std::shared_ptr<Tensor> &tensor, function::ReduceOpType reduce_op) const {
    void *buffer = tensor->DataPtr();

    const auto *device = dynamic_cast<const CudaDevice *>(tensor->GetDevice());
    auto rank = device->rank();
    auto comm = comms_.at(rank.thread_rank());

    NCCL_CHECK(ncclAllReduce(buffer, buffer, tensor->NumElements(), kNcclDtypeMap.at(tensor->Dtype()),
                             kNcclReduceOpMap.at(reduce_op), comm, device->Stream()));
}

ncclComm_t ProcessGroup::comm(int idx) const { return comms_.at(idx); }

ProcessGroupFactory *ProcessGroupFactory::Instance() {
    static std::mutex mutex;
    static std::unique_ptr<ProcessGroupFactory> instance = nullptr;
    if (instance == nullptr) {
        std::lock_guard<std::mutex> lock(mutex);
        if (instance == nullptr) {
            instance.reset(new ProcessGroupFactory());
        }
    }
    return instance.get();
}

void ProcessGroupFactory::Create(const std::string &name, int comm_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    name_to_group_[name] = std::make_unique<ProcessGroup>(ProcessGroup(comm_size));
}

const ProcessGroup *ProcessGroupFactory::Get(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return name_to_group_.at(name).get();
}

const ProcessGroup *ProcessGroupFactory::GetDefaultProcessGroup() const {
    return name_to_group_.at(kDefaltProcessGroupName).get();
}

ProcessGroupFactory::ProcessGroupFactory() { Create(kDefaltProcessGroupName, global::GetIntraWorldSize()); }
} // namespace infini_train::nn::parallel

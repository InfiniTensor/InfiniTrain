#include "infini_train/include/nn/parallel/process_group.h"

#include <memory>
#include <numeric>
#include <vector>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#ifdef USE_CUDA
#include "infini_train/include/common/cuda/common_cuda.h"
#endif
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/work.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

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

#ifdef USE_NCCL
ProcessGroup::ProcessGroup(const ncclUniqueId &nccl_id) : world_size_(global::GetWorldSize()) {
    int local_comm_size = global::GetNthreadPerProc();
    comms_.resize(local_comm_size);
    std::vector<int> device_indices(local_comm_size);

    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < local_comm_size; ++i) {
        device_indices[i] = i;

        int global_rank = global::GetGlobalProcRank() * global::GetNthreadPerProc() + i;
        NCCL_CHECK(ncclCommInitRank(&comms_[i], world_size_, nccl_id, global_rank));
    }
    NCCL_CHECK(ncclGroupEnd());

    Init(device_indices);
}

void ProcessGroup::Init(const std::vector<int> &device_indices) {
    comm_streams_.resize(world_size_);
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));

    for (int i = 0; i < world_size_; ++i) {
        auto device = DeviceManager::Instance()->GetDevice(DeviceType::kCUDA, device_indices[i]);
        devices_.push_back(device);
        device_comm_map_[device] = comms_[i];
        thread_group_rank_map_[device->rank().thread_rank()] = i;

        device->SetDevice();
        int low, high;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low, &high));
        CUDA_CHECK(cudaStreamCreateWithPriority(&comm_streams_[i], cudaStreamNonBlocking, high));
        device_stream_map_[device] = comm_streams_[i];
    }

    CUDA_CHECK(cudaSetDevice(current_device));
}

ProcessGroup::~ProcessGroup() {
    for (auto &s : comm_streams_) {
        if (s) {
            cudaStreamDestroy(s);
        }
    }
    for (auto &c : comms_) {
        if (c) {
            ncclCommDestroy(c);
        }
    }
}

int ProcessGroup::GetGroupRank(int thread_rank) const { return thread_group_rank_map_.at(thread_rank); }

void ProcessGroup::AllReduce(const std::shared_ptr<Tensor> &tensor, function::ReduceOpType reduce_op) const {
    void *buffer = tensor->DataPtr();

    const auto *device = dynamic_cast<const CudaDevice *>(tensor->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();
    NCCL_CHECK(ncclAllReduce(buffer, buffer, tensor->NumElements(), kNcclDtypeMap.at(tensor->Dtype()),
                             kNcclReduceOpMap.at(reduce_op), comm, device->Stream()));
}

void ProcessGroup::AllGather(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input) const {
    const auto *device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();
    NCCL_CHECK(ncclAllGather(input->DataPtr(), output->DataPtr(), input->NumElements(),
                             kNcclDtypeMap.at(input->Dtype()), comm, device->Stream()));
}

void ProcessGroup::ReduceScatter(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                 function::ReduceOpType reduce_op) const {
    const auto *device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();
    NCCL_CHECK(ncclReduceScatter(input->DataPtr(), output->DataPtr(), output->NumElements(),
                                 kNcclDtypeMap.at(input->Dtype()), kNcclReduceOpMap.at(reduce_op), comm,
                                 device->Stream()));
}

std::vector<std::shared_ptr<Tensor>>
ProcessGroup::BroadCast(const std::vector<std::shared_ptr<Tensor>> &input_tensors) const {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;
    std::vector<const Device *> devices;

    CHECK_EQ(world_size_, comms_.size());

    for (size_t i = 0; i < world_size_; ++i) {
        auto device = devices_[i];
        for (const auto &input_tensor : input_tensors) {
            outputs.push_back(std::make_shared<Tensor>(input_tensor->Dims(), input_tensor->Dtype(), device));
        }
        devices.push_back(device);
        streams.push_back(dynamic_cast<const CudaDevice *>(device)->Stream());
        comms.push_back(device_comm_map_.at(device));
    }

    int root = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] == input_tensors[0]->GetDevice()) {
            root = i;
            break;
        }
    }
    CHECK_NE(root, -1) << "Root not found in input devices";

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < devices.size(); ++i) {
        devices[i]->SetDevice();
        for (size_t j = 0; j < input_tensors.size(); ++j) {
            const auto &input_tensor = input_tensors[j];
            const auto dtype = input_tensor->Dtype();
            auto nccl_dtype = kNcclDtypeMap.at(dtype);
            auto count = input_tensor->NumElements();
            void *send_buffer = (devices[i] == input_tensor->GetDevice() ? input_tensor->DataPtr() : nullptr);
            NCCL_CHECK(ncclBroadcast(send_buffer, outputs[i * input_tensors.size() + j]->DataPtr(), count, nccl_dtype,
                                     0, comms[i], streams[i]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    return outputs;
}

std::vector<std::shared_ptr<Tensor>>
ProcessGroup::ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads,
                                 const Device *destination) const {
    // grads: [devices, tensors]
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;
    std::vector<const Device *> devices;

    for (size_t i = 0; i < grads[0].size(); ++i) {
        outputs.push_back(std::make_shared<Tensor>(grads[0][i]->Dims(), grads[0][i]->Dtype(), destination));
        outputs[i]->Fill<float>(0.0f);
    }
    for (size_t i = 0; i < grads.size(); ++i) {
        devices.push_back(grads[i][0]->GetDevice());
        streams.push_back(dynamic_cast<const CudaDevice *>(devices[i])->Stream());
        comms.push_back(device_comm_map_.at(devices[i]));
    }

    int root = -1;
    for (size_t i = 0; i < grads.size(); ++i) {
        if (grads[i][0]->GetDevice() == destination) {
            root = i;
            break;
        }
    }
    CHECK_NE(root, -1) << "Destination device not found in grads group";

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < grads.size(); ++i) {
        devices[i]->SetDevice();
        for (size_t j = 0; j < grads[i].size(); ++j) {
            const auto &grad = grads[i][j];
            const auto dtype = grad->Dtype();
            auto nccl_dtype = kNcclDtypeMap.at(dtype);
            auto count = grad->NumElements();
            void *send_buffer = grad->DataPtr();
            NCCL_CHECK(
                ncclReduce(send_buffer, outputs[j]->DataPtr(), count, nccl_dtype, ncclSum, 0, comms[i], streams[i]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    return outputs;
}

std::vector<std::shared_ptr<Tensor>> ProcessGroup::Scatter(const std::shared_ptr<Tensor> &tensor,
                                                           std::vector<const Device *> devices, int64_t dim) const {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<std::shared_ptr<Tensor>> split_tensors = tensor->Split(tensor->Dims()[dim] / devices.size(), dim);
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;
    int src_rank = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (tensor->GetDevice() == devices[i]) {
            src_rank = i;
        }
        outputs.push_back(std::make_shared<Tensor>(split_tensors[i]->Dims(), split_tensors[i]->Dtype(), devices[i]));
        streams.push_back(dynamic_cast<const CudaDevice *>(devices[i])->Stream());
        comms.push_back(device_comm_map_.at(devices[i]));
    }

    CHECK_NE(src_rank, -1) << "Source device not found in input devices";

    NCCL_CHECK(ncclGroupStart());
    const auto dtype = tensor->Dtype();
    auto nccl_dtype = kNcclDtypeMap.at(dtype);

    for (size_t i = 0; i < devices.size(); ++i) {
        devices[i]->SetDevice();
        const auto dtype = tensor->Dtype();
        auto nccl_dtype = kNcclDtypeMap.at(dtype);
        NCCL_CHECK(ncclSend(split_tensors[i]->DataPtr(), split_tensors[i]->NumElements(), nccl_dtype, i,
                            comms[src_rank], streams[src_rank]));
        NCCL_CHECK(
            ncclRecv(outputs[i]->DataPtr(), outputs[i]->NumElements(), nccl_dtype, src_rank, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());
    return outputs;
}

std::shared_ptr<Tensor> ProcessGroup::Gather(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                             const Device *destination, int64_t dim) const {
    std::vector<std::shared_ptr<Tensor>> outouts;
    int64_t num_devices = tensors.size();
    auto dtype = tensors[0]->Dtype();
    auto nccl_dtype = kNcclDtypeMap.at(dtype);

    int64_t total_dim = 0;

    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;
    std::vector<const Device *> devices;

    int dest_rank = -1;
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto device = tensors[i]->GetDevice();
        if (device == destination) {
            dest_rank = i;
        }
        streams.push_back(dynamic_cast<const CudaDevice *>(device)->Stream());
        comms.push_back(device_comm_map_.at(device));
        devices.push_back(device);

        total_dim += tensors[i]->Dims()[dim];
    }

    std::vector<int64_t> out_dims = tensors[0]->Dims();
    out_dims[dim] = total_dim;
    auto output = std::make_shared<Tensor>(out_dims, dtype, destination);

    CHECK_NE(dest_rank, -1) << "Destination device not found in input tensors's devices";

    NCCL_CHECK(ncclGroupStart());
    int64_t offset = 0;

    for (size_t i = 0; i < num_devices; ++i) {
        devices[i]->SetDevice();
        auto &tensor = tensors[i];
        size_t num_elements = tensor->NumElements();
        void *send_ptr = tensor->DataPtr();

        auto recv_ptr = static_cast<int8_t *>(output->DataPtr()) + offset;

        NCCL_CHECK(ncclSend(send_ptr, num_elements, nccl_dtype, dest_rank, comms[i], streams[i]));
        NCCL_CHECK(ncclRecv(recv_ptr, num_elements, nccl_dtype, i, comms[dest_rank], streams[dest_rank]));

        offset += tensor->SizeInBytes();
    }

    NCCL_CHECK(ncclGroupEnd());
    return output;
}

std::vector<std::shared_ptr<Tensor>> ProcessGroup::NcclSend(std::vector<std::shared_ptr<Tensor>> tensors,
                                                            int dest_rank) const {
    for (int i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        CHECK_NOTNULL(tensor);

        auto device_ptr = dynamic_cast<const CudaDevice *>(tensor->GetDevice());
        cudaStream_t stream = device_ptr->Stream();
        ncclComm_t comm = device_comm_map_.at(device_ptr);

        CHECK_NE(dest_rank, -1) << "Destination device not found in input tensors's devices";

        auto dtype = tensor->Dtype();
        auto nccl_dtype = kNcclDtypeMap.at(dtype);
        auto count = tensor->NumElements();
        void *buffer = tensor->DataPtr();
        CHECK_NOTNULL(buffer);

        NCCL_CHECK(ncclSend(buffer, count, nccl_dtype, dest_rank, comm, stream));
    }
    return tensors;
}

std::vector<std::shared_ptr<Tensor>> ProcessGroup::NcclRecv(std::vector<std::shared_ptr<Tensor>> tensors,
                                                            int src_rank) const {
    for (int i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        CHECK_NOTNULL(tensor);

        auto device_ptr = dynamic_cast<const CudaDevice *>(tensor->GetDevice());
        cudaStream_t stream = device_ptr->Stream();
        ncclComm_t comm = device_comm_map_.at(device_ptr);

        CHECK_NE(src_rank, -1) << "Source device not found in input devices";

        auto dtype = tensor->Dtype();
        auto nccl_dtype = kNcclDtypeMap.at(dtype);
        auto count = tensor->NumElements();
        void *buffer = tensor->DataPtr();
        CHECK_NOTNULL(buffer);

        NCCL_CHECK(ncclRecv(buffer, count, nccl_dtype, src_rank, comm, stream));
    }
    return tensors;
}

std::shared_ptr<Work> ProcessGroup::AllReduceAsync(const std::shared_ptr<Tensor> &tensor,
                                                   function::ReduceOpType reduce_op) const {
    void *buffer = tensor->DataPtr();
    const auto *device = dynamic_cast<const CudaDevice *>(tensor->GetDevice());
    device->SetDevice();

    auto comm = device_comm_map_.at(device);

    cudaStream_t compute_stream = device->Stream();
    cudaStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkNccl>(device, comm);

    cudaEvent_t ready_event = reinterpret_cast<cudaEvent_t>(work->ready_event());
    cudaEvent_t done_event = reinterpret_cast<cudaEvent_t>(work->done_event());

    CUDA_CHECK(cudaEventRecord(ready_event, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream, ready_event, 0));

    // Perform NcclAllReduce on comm stream
    NCCL_CHECK(ncclAllReduce(buffer, buffer, tensor->NumElements(), kNcclDtypeMap.at(tensor->Dtype()),
                             kNcclReduceOpMap.at(reduce_op), comm, comm_stream));

    CUDA_CHECK(cudaEventRecord(done_event, comm_stream));

    // Do not let compute stream wait for done event here
    return std::move(work);
}
#endif

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

const ProcessGroup *ProcessGroupFactory::GetOrCreate(const std::string &name, int comm_size) {
    std::vector<int> device_indices(comm_size);
    std::iota(device_indices.begin(), device_indices.end(), 0);
    return GetOrCreate(name, [&]() { return std::make_unique<ProcessGroup>(device_indices); });
}

const ProcessGroup *ProcessGroupFactory::GetOrCreate(const std::string &name, const std::vector<int> &device_indices) {
    return GetOrCreate(name, [&]() { return std::make_unique<ProcessGroup>(device_indices); });
}

#ifdef USE_NCCL
const ProcessGroup *ProcessGroupFactory::GetOrCreate(const std::string &name, const ncclUniqueId &nccl_id) {
    return GetOrCreate(name, [&]() { return std::make_unique<ProcessGroup>(nccl_id); });
}
#endif

const ProcessGroup *ProcessGroupFactory::Get(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return name_to_group_.at(name).get();
}

const ProcessGroup *ProcessGroupFactory::GetDefaultProcessGroup() const {
    return name_to_group_.at(kDefaltProcessGroupName).get();
}

ProcessGroupFactory::ProcessGroupFactory() {
#ifdef USE_NCCL
    GetOrCreate(kDefaltProcessGroupName, global::GetNcclId());
#else
    GetOrCreate(kDefaltProcessGroupName, global::GetWorldSize());
#endif
}
} // namespace infini_train::nn::parallel

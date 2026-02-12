#include "infini_train/include/nn/parallel/process_group.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <ranges>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/ccl/ccl.h"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/core/stream.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/work.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

int ProcessGroup::GetGroupRank(int global_rank) const { return global_group_rank_map_.at(global_rank); }

ProcessGroup::ProcessGroup(int world_size, const std::string &name) : world_size_(world_size), name_(name) {}

ProcessGroup::ProcessGroup(Device::DeviceType backend, const std::string &process_group_name,
                           const std::vector<int> &ranks)
    : backend_(backend), world_size_(ranks.size()), name_(process_group_name) {
    CHECK_GT(world_size_, 0);
    if (global::GetNnodes() == 1 && global::GetNprocPerNode() == 1) {
        InitSingleProcess(ranks);
    } else {
        InitMultiProcess(ranks);
    }
    InitStreams();
}

ProcessGroup::~ProcessGroup() {
    if (is_main_process_) {
        core::GetCclImpl(backend_)->CleanupUniqueIdFile(name_);
    }

    auto *impl = core::GetDeviceGuardImpl(backend_);
    for (auto &s : comm_streams_) {
        if (s) {
            impl->DestroyStream(s.get());
        }
    }

    auto *ccl_impl = core::GetCclImpl(backend_);
    for (auto &c : comms_) {
        if (c) {
            ccl_impl->CommDestroy(c.get());
        }
    }
}

void ProcessGroup::InitSingleProcess(const std::vector<int> &ranks) {
    comms_.resize(world_size_);

    auto *ccl_impl = core::GetCclImpl(backend_);
    std::vector<core::CclComm *> comm_ptrs;
    comm_ptrs.reserve(world_size_);

    for (int i = 0; i < ranks.size(); ++i) {
        auto device = Device(backend_, ranks[i]);
        devices_.push_back(device);

        core::CclComm *comm_raw = nullptr;
        ccl_impl->CreateComm(&comm_raw);
        CHECK_NOTNULL(comm_raw);
        comm_ptrs.push_back(comm_raw);
        device_comm_map_[device.index()] = comm_raw;

        auto comm = std::unique_ptr<core::CclComm>(comm_raw);
        comms_[i] = std::move(comm);
        global_group_rank_map_[device.Rank().GlobalRank()] = i;
    }
    ccl_impl->CommInitAll(comm_ptrs.data(), world_size_, ranks.data());
}

void ProcessGroup::InitMultiProcess(const std::vector<int> &ranks) {
    auto *ccl_impl = core::GetCclImpl(backend_);

    int n_threads = global::GetNthreadPerProc();
    int global_proc_rank = global::GetGlobalProcRank();
    int lower_rank = global_proc_rank * n_threads;
    int upper_rank = (global_proc_rank + 1) * n_threads;

    core::CclUniqueId *unique_id_raw = nullptr;
    ccl_impl->CreateUniqueId(&unique_id_raw);
    CHECK_NOTNULL(unique_id_raw);
    std::unique_ptr<core::CclUniqueId> unique_id(unique_id_raw);
    int min_rank = std::ranges::min(ranks);
    if (min_rank < upper_rank && min_rank >= lower_rank) {
        is_main_process_ = true;
        ccl_impl->GetUniqueId(unique_id.get());
        ccl_impl->WriteUniqueId(*unique_id, name_);
    } else {
        ccl_impl->ReadUniqueId(unique_id.get(), name_);
    }

    core::Ccl ccl(backend_);
    for (int i = 0; i < n_threads; ++i) {
        int global_thread_rank = lower_rank + i;
        auto it = std::ranges::find(ranks, global_thread_rank);
        if (it != ranks.end()) {
            auto device = Device(backend_, static_cast<int8_t>(i));
            core::DeviceGuard guard(device);

            core::CclComm *comm_raw = nullptr;
            ccl_impl->CreateComm(&comm_raw);
            CHECK_NOTNULL(comm_raw);
            auto comm = std::unique_ptr<core::CclComm>(comm_raw);
            int group_rank = std::distance(ranks.begin(), it);
            ccl_impl->CommInitRank(comm.get(), world_size_, *unique_id, group_rank);
            comms_.push_back(std::move(comm));

            global_group_rank_map_[device.Rank().GlobalRank()] = group_rank;
            devices_.push_back(device);
            device_comm_map_[device.index()] = comm_raw;
        }
    }
}

void ProcessGroup::InitStreams() {
    for (const auto &device : devices_) {
        core::DeviceGuard guard(device);
        auto *impl = core::GetDeviceGuardImpl(device.type());
        auto *stream = impl->CreateStreamWithPriority(device, 0);
        comm_streams_.emplace_back(stream);
        device_stream_map_[device.index()] = stream;
    }
}

std::shared_ptr<Work> ProcessGroup::AllReduce(const std::shared_ptr<Tensor> &tensor, function::ReduceOpType reduce_op,
                                              bool async_op) const {
    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);
    auto *runtime_impl = core::GetDeviceGuardImpl(device.type());
    auto *ccl_impl = core::GetCclImpl(device.type());
    auto *compute_stream = runtime_impl->GetStream(device);
    auto *comm_stream = device_stream_map_.at(device.index());
    auto comm = device_comm_map_.at(device.index());

    auto work = std::make_shared<Work>(device, comm);
    runtime_impl->EventRecord(work->ready_event(), compute_stream);
    runtime_impl->StreamWaitEvent(comm_stream, work->ready_event(), 0);
    ccl_impl->AllReduce(tensor->DataPtr(), tensor->DataPtr(), tensor->NumElements(), tensor->Dtype(), reduce_op, comm,
                        comm_stream);
    runtime_impl->EventRecord(work->done_event(), comm_stream);
    if (async_op) {
        return work;
    }
    work->WaitNonBlocking();
    return nullptr;
}

std::shared_ptr<Work> ProcessGroup::AllGather(const std::shared_ptr<Tensor> &output,
                                              const std::shared_ptr<Tensor> &input, bool async_op) const {
    auto device = input->GetDevice();
    core::DeviceGuard guard(device);
    auto *runtime_impl = core::GetDeviceGuardImpl(device.type());
    auto *ccl_impl = core::GetCclImpl(device.type());
    auto *compute_stream = runtime_impl->GetStream(device);
    auto *comm_stream = device_stream_map_.at(device.index());
    auto comm = device_comm_map_.at(device.index());

    auto work = std::make_shared<Work>(device, comm);
    runtime_impl->EventRecord(work->ready_event(), compute_stream);
    runtime_impl->StreamWaitEvent(comm_stream, work->ready_event(), 0);
    ccl_impl->AllGather(input->DataPtr(), output->DataPtr(), input->NumElements(), input->Dtype(), comm, comm_stream);
    runtime_impl->EventRecord(work->done_event(), comm_stream);
    if (async_op) {
        return work;
    }
    work->WaitNonBlocking();
    return nullptr;
}

std::shared_ptr<Work> ProcessGroup::ReduceScatter(const std::shared_ptr<Tensor> &output,
                                                  const std::shared_ptr<Tensor> &input,
                                                  function::ReduceOpType reduce_op, bool async_op) const {
    auto device = input->GetDevice();
    core::DeviceGuard guard(device);
    auto *runtime_impl = core::GetDeviceGuardImpl(device.type());
    auto *ccl_impl = core::GetCclImpl(device.type());
    auto *compute_stream = runtime_impl->GetStream(device);
    auto *comm_stream = device_stream_map_.at(device.index());
    auto comm = device_comm_map_.at(device.index());

    auto work = std::make_shared<Work>(device, comm);
    runtime_impl->EventRecord(work->ready_event(), compute_stream);
    runtime_impl->StreamWaitEvent(comm_stream, work->ready_event(), 0);
    ccl_impl->ReduceScatter(input->DataPtr(), output->DataPtr(), output->NumElements(), input->Dtype(), reduce_op, comm,
                            comm_stream);
    runtime_impl->EventRecord(work->done_event(), comm_stream);
    if (async_op) {
        return work;
    }
    work->WaitNonBlocking();
    return nullptr;
}

std::shared_ptr<Work> ProcessGroup::Send(std::vector<std::shared_ptr<Tensor>> tensors, int dest_rank,
                                         bool async_op) const {
    CHECK_GT(tensors.size(), 0);
    auto device = tensors[0]->GetDevice();
    core::DeviceGuard guard(device);
    auto *runtime_impl = core::GetDeviceGuardImpl(device.type());
    auto *ccl_impl = core::GetCclImpl(device.type());
    auto *compute_stream = runtime_impl->GetStream(device);
    auto *comm_stream = device_stream_map_.at(device.index());
    auto comm = device_comm_map_.at(device.index());

    auto work = std::make_shared<Work>(device, comm);
    runtime_impl->EventRecord(work->ready_event(), compute_stream);
    runtime_impl->StreamWaitEvent(comm_stream, work->ready_event(), 0);
    for (const auto &tensor : tensors) {
        CHECK_NOTNULL(tensor);
        CHECK_EQ(device, tensor->GetDevice());
        ccl_impl->Send(tensor->DataPtr(), tensor->NumElements(), tensor->Dtype(), dest_rank, comm, comm_stream);
    }
    runtime_impl->EventRecord(work->done_event(), comm_stream);
    if (async_op) {
        return work;
    }
    work->WaitNonBlocking();
    return nullptr;
}

std::shared_ptr<Work> ProcessGroup::Recv(std::vector<std::shared_ptr<Tensor>> tensors, int src_rank,
                                         bool async_op) const {
    CHECK_GT(tensors.size(), 0);
    auto device = tensors[0]->GetDevice();
    core::DeviceGuard guard(device);
    auto *runtime_impl = core::GetDeviceGuardImpl(device.type());
    auto *ccl_impl = core::GetCclImpl(device.type());
    auto *compute_stream = runtime_impl->GetStream(device);
    auto *comm_stream = device_stream_map_.at(device.index());
    auto comm = device_comm_map_.at(device.index());

    auto work = std::make_shared<Work>(device, comm);
    runtime_impl->EventRecord(work->ready_event(), compute_stream);
    runtime_impl->StreamWaitEvent(comm_stream, work->ready_event(), 0);
    for (const auto &tensor : tensors) {
        CHECK_NOTNULL(tensor);
        CHECK_EQ(device, tensor->GetDevice());
        ccl_impl->Recv(tensor->DataPtr(), tensor->NumElements(), tensor->Dtype(), src_rank, comm, comm_stream);
    }
    runtime_impl->EventRecord(work->done_event(), comm_stream);
    if (async_op) {
        return work;
    }
    work->WaitNonBlocking();
    return nullptr;
}

std::vector<std::shared_ptr<Tensor>>
ProcessGroup::BroadCast(const std::vector<std::shared_ptr<Tensor>> &input_tensors) const {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<core::Stream *> streams;
    std::vector<core::CclComm *> comms;
    std::vector<Device> devices;

    CHECK_EQ(world_size_, static_cast<int>(comms_.size()));
    for (int i = 0; i < world_size_; ++i) {
        auto device = devices_[static_cast<size_t>(i)];
        for (const auto &input_tensor : input_tensors) {
            outputs.push_back(std::make_shared<Tensor>(input_tensor->Dims(), input_tensor->Dtype(), device));
        }
        devices.push_back(device);
        streams.push_back(core::GetDeviceGuardImpl(device.type())->GetStream(device));
        comms.push_back(device_comm_map_.at(device.index()));
    }

    int root = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] == input_tensors[0]->GetDevice()) {
            root = static_cast<int>(i);
            break;
        }
    }
    CHECK_NE(root, -1) << "Root not found in input devices";

    auto *ccl_impl = core::GetCclImpl(devices[0].type());
    core::Ccl ccl(devices[0].type());
    for (size_t i = 0; i < devices.size(); ++i) {
        core::DeviceGuard guard(devices[i]);
        for (size_t j = 0; j < input_tensors.size(); ++j) {
            const auto &input_tensor = input_tensors[j];
            const void *send_buffer = (static_cast<int>(i) == root ? input_tensor->DataPtr() : nullptr);
            ccl_impl->Broadcast(send_buffer, outputs[i * input_tensors.size() + j]->DataPtr(),
                                input_tensor->NumElements(), input_tensor->Dtype(), root, comms[i], streams[i]);
        }
    }

    return outputs;
}

std::vector<std::shared_ptr<Tensor>>
ProcessGroup::ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads,
                                 Device destination) const {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<core::Stream *> streams;
    std::vector<core::CclComm *> comms;
    std::vector<Device> devices;

    for (size_t i = 0; i < grads[0].size(); ++i) {
        outputs.push_back(std::make_shared<Tensor>(grads[0][i]->Dims(), grads[0][i]->Dtype(), destination));
        outputs[i]->Fill<float>(0.0f);
    }
    for (size_t i = 0; i < grads.size(); ++i) {
        devices.push_back(grads[i][0]->GetDevice());
        streams.push_back(core::GetDeviceGuardImpl(devices[i].type())->GetStream(devices[i]));
        comms.push_back(device_comm_map_.at(devices[i].index()));
    }

    int root = -1;
    for (size_t i = 0; i < grads.size(); ++i) {
        if (grads[i][0]->GetDevice() == destination) {
            root = static_cast<int>(i);
            break;
        }
    }
    CHECK_NE(root, -1) << "Destination device not found in grads group";

    auto *ccl_impl = core::GetCclImpl(devices[0].type());
    core::Ccl ccl(devices[0].type());
    for (size_t i = 0; i < grads.size(); ++i) {
        core::DeviceGuard guard(devices[i]);
        for (size_t j = 0; j < grads[i].size(); ++j) {
            const auto &grad = grads[i][j];
            ccl_impl->Reduce(grad->DataPtr(), outputs[j]->DataPtr(), grad->NumElements(), grad->Dtype(),
                             function::ReduceOpType::kSum, root, comms[i], streams[i]);
        }
    }

    return outputs;
}

std::vector<std::shared_ptr<Tensor>> ProcessGroup::Scatter(const std::shared_ptr<Tensor> &tensor,
                                                           std::vector<Device> devices, int64_t dim) const {
    std::vector<std::shared_ptr<Tensor>> outputs;
    auto split_tensors = tensor->Split(tensor->Dims()[dim] / devices.size(), dim);
    std::vector<core::Stream *> streams;
    std::vector<core::CclComm *> comms;
    int src_rank = -1;

    for (size_t i = 0; i < devices.size(); ++i) {
        if (tensor->GetDevice() == devices[i]) {
            src_rank = static_cast<int>(i);
        }
        outputs.push_back(std::make_shared<Tensor>(split_tensors[i]->Dims(), split_tensors[i]->Dtype(), devices[i]));
        streams.push_back(core::GetDeviceGuardImpl(devices[i].type())->GetStream(devices[i]));
        comms.push_back(device_comm_map_.at(devices[i].index()));
    }
    CHECK_NE(src_rank, -1) << "Source device not found in input devices";

    auto *ccl_impl = core::GetCclImpl(devices[0].type());
    core::Ccl ccl(devices[0].type());
    for (size_t i = 0; i < devices.size(); ++i) {
        core::DeviceGuard guard(devices[i]);
        ccl_impl->Send(split_tensors[i]->DataPtr(), split_tensors[i]->NumElements(), tensor->Dtype(),
                       static_cast<int>(i), comms[static_cast<size_t>(src_rank)],
                       streams[static_cast<size_t>(src_rank)]);
        ccl_impl->Recv(outputs[i]->DataPtr(), outputs[i]->NumElements(), tensor->Dtype(), src_rank, comms[i],
                       streams[i]);
    }
    return outputs;
}

std::shared_ptr<Tensor> ProcessGroup::Gather(const std::vector<std::shared_ptr<Tensor>> &tensors, Device destination,
                                             int64_t dim) const {
    int64_t num_devices = static_cast<int64_t>(tensors.size());
    auto dtype = tensors[0]->Dtype();
    int64_t total_dim = 0;

    std::vector<core::Stream *> streams;
    std::vector<core::CclComm *> comms;
    std::vector<Device> devices;

    int dest_rank = -1;
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto device = tensors[i]->GetDevice();
        if (device == destination) {
            dest_rank = static_cast<int>(i);
        }
        streams.push_back(core::GetDeviceGuardImpl(device.type())->GetStream(device));
        comms.push_back(device_comm_map_.at(device.index()));
        devices.push_back(device);
        total_dim += tensors[i]->Dims()[dim];
    }

    std::vector<int64_t> out_dims = tensors[0]->Dims();
    out_dims[dim] = total_dim;
    auto output = std::make_shared<Tensor>(out_dims, dtype, destination);
    CHECK_NE(dest_rank, -1) << "Destination device not found in input tensors's devices";

    auto *ccl_impl = core::GetCclImpl(devices[0].type());
    core::Ccl ccl(devices[0].type());
    int64_t offset = 0;
    for (int64_t i = 0; i < num_devices; ++i) {
        core::DeviceGuard guard(devices[static_cast<size_t>(i)]);
        auto &tensor = tensors[static_cast<size_t>(i)];
        size_t num_elements = tensor->NumElements();
        void *send_ptr = tensor->DataPtr();
        auto *recv_ptr = static_cast<int8_t *>(output->DataPtr()) + offset;
        ccl_impl->Send(send_ptr, num_elements, dtype, dest_rank, comms[static_cast<size_t>(i)],
                       streams[static_cast<size_t>(i)]);
        ccl_impl->Recv(recv_ptr, num_elements, dtype, static_cast<int>(i), comms[static_cast<size_t>(dest_rank)],
                       streams[static_cast<size_t>(dest_rank)]);
        offset += tensor->SizeInBytes();
    }
    return output;
}

#if 0
// Legacy ProcessGroupNCCL implementation is intentionally kept commented out.
// It was the direct nccl/cuda path (ProcessGroupNCCL::InitSingleProcess / InitMultiProcess / InitStreams /
// AllReduce / AllGather / ReduceScatter / Send / Recv / BroadCast / ReduceAddCoalesced / Scatter / Gather).
// The active implementation above is backend-dispatched through DeviceGuardImpl + CclImpl.
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
    return GetOrCreate(
        name, [&]() { return std::make_unique<ProcessGroup>(Device::DeviceType::kCUDA, name, device_indices); });
}

const ProcessGroup *ProcessGroupFactory::GetOrCreate(const std::string &name, const std::vector<int> &device_indices) {
    return GetOrCreate(
        name, [&]() { return std::make_unique<ProcessGroup>(Device::DeviceType::kCUDA, name, device_indices); });
}

const ProcessGroup *ProcessGroupFactory::Get(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return name_to_group_.at(name).get();
}

const ProcessGroup *ProcessGroupFactory::GetDefaultProcessGroup() const {
    return name_to_group_.at(kDefaltProcessGroupName).get();
}

ProcessGroupFactory::ProcessGroupFactory() { GetOrCreate(kDefaltProcessGroupName, global::GetWorldSize()); }

} // namespace infini_train::nn::parallel

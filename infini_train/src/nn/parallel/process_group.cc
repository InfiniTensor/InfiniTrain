#include "infini_train/include/nn/parallel/process_group.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#ifdef USE_NCCL
#include <nccl.h>
#endif
#ifdef USE_MCCL
#include <mccl.h>
#endif

#ifdef USE_CUDA
#include "infini_train/include/common/cuda/common_cuda.h"
#endif
#ifdef USE_MACA
#include "infini_train/include/common/maca/common_maca.h"
#endif
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/work.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

namespace {
using nn::parallel::function::ReduceOpType;

#ifdef USE_NCCL
const std::unordered_map<DataType, ncclDataType_t> kNcclDtypeMap = {
    {DataType::kUINT8, ncclUint8},       {DataType::kINT8, ncclInt8},     {DataType::kUINT32, ncclUint32},
    {DataType::kINT32, ncclInt32},       {DataType::kUINT64, ncclUint64}, {DataType::kINT64, ncclInt64},
    {DataType::kBFLOAT16, ncclBfloat16}, {DataType::kFLOAT16, ncclHalf},  {DataType::kFLOAT32, ncclFloat32},
    {DataType::kFLOAT64, ncclFloat64},
};

const std::unordered_map<ReduceOpType, ncclRedOp_t> kNcclReduceOpMap = {
    {ReduceOpType::kSum, ncclSum},
    {ReduceOpType::kProd, ncclProd},
    {ReduceOpType::kMax, ncclMax},
    {ReduceOpType::kAvg, ncclAvg},
};

inline std::string NcclFileName(const std::string &name, bool tmp = false) {
    return std::format("ncclUniqueId_{}.{}", name, tmp ? "tmp" : "bin");
}

void WriteNcclUniqueId(const ncclUniqueId &nccl_id, const std::string &pg_name) {
    std::string tmp_path = NcclFileName(pg_name, true);

    std::ofstream ofs(tmp_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(&nccl_id), sizeof(nccl_id));
    ofs.close();

    std::rename(tmp_path.c_str(), NcclFileName(pg_name).c_str());
}

void ReadNcclUniqueId(ncclUniqueId &nccl_id, const std::string &pg_name) {
    std::string file_path = NcclFileName(pg_name);

    while (std::filesystem::exists(file_path) == false) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    std::ifstream ifs(file_path, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(&nccl_id), sizeof(nccl_id));
    ifs.close();
}

void CleanupNcclIdFile(const std::string &pg_name) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    std::string file_path = NcclFileName(pg_name);

    if (std::filesystem::exists(file_path)) {
        std::filesystem::remove(file_path);
    }
}
#endif

#ifdef USE_MCCL
const std::unordered_map<DataType, mcclDataType_t> kMcclDtypeMap = {
    {DataType::kUINT8, mcclUint8},       {DataType::kINT8, mcclInt8},     {DataType::kUINT32, mcclUint32},
    {DataType::kINT32, mcclInt32},       {DataType::kUINT64, mcclUint64}, {DataType::kINT64, mcclInt64},
    {DataType::kBFLOAT16, mcclBfloat16}, {DataType::kFLOAT16, mcclHalf},  {DataType::kFLOAT32, mcclFloat32},
    {DataType::kFLOAT64, mcclFloat64},
};

const std::unordered_map<ReduceOpType, mcclRedOp_t> kMcclReduceOpMap = {
    {ReduceOpType::kSum, mcclSum},
    {ReduceOpType::kProd, mcclProd},
    {ReduceOpType::kMax, mcclMax},
    {ReduceOpType::kAvg, mcclAvg},
};

inline std::string McclFileName(const std::string &name, bool tmp = false) {
    return std::format("mcclUniqueId_{}.{}", name, tmp ? "tmp" : "bin");
}

void WriteMcclUniqueId(const mcclUniqueId &mccl_id, const std::string &pg_name) {
    std::string tmp_path = McclFileName(pg_name, true);

    std::ofstream ofs(tmp_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(&mccl_id), sizeof(mccl_id));
    ofs.close();

    std::rename(tmp_path.c_str(), McclFileName(pg_name).c_str());
}

void ReadMcclUniqueId(mcclUniqueId &mccl_id, const std::string &pg_name) {
    std::string file_path = McclFileName(pg_name);

    while (std::filesystem::exists(file_path) == false) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    std::ifstream ifs(file_path, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(&mccl_id), sizeof(mccl_id));
    ifs.close();
}

void CleanupMcclIdFile(const std::string &pg_name) {
    std::string file_path = McclFileName(pg_name);

    if (std::filesystem::exists(file_path)) {
        std::filesystem::remove(file_path);
    }
}
#endif

} // namespace

} // namespace infini_train

namespace infini_train::nn::parallel {

int ProcessGroup::GetGroupRank(int global_rank) const { return global_group_rank_map_.at(global_rank); }

ProcessGroup::ProcessGroup(int world_size, const std::string &name) : world_size_(world_size), name_(name) {}

#ifdef USE_NCCL
ProcessGroupNCCL::ProcessGroupNCCL(const std::string &process_group_name, const std::vector<int> &ranks)
    : ProcessGroup(ranks.size(), process_group_name) {
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (global::GetNnodes() == 1 && global::GetNprocPerNode() == 1) {
        InitSingleProcess(ranks);
    } else {
        InitMultiProcess(ranks);
    }
    InitStreams();

    CUDA_CHECK(cudaSetDevice(current_device));
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
    if (is_main_process_) {
        CleanupNcclIdFile(name_);
    }

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

void ProcessGroupNCCL::InitSingleProcess(const std::vector<int> &ranks) {
    comms_.resize(world_size_);
    NCCL_CHECK(ncclCommInitAll(comms_.data(), world_size_, ranks.data()));

    for (int i = 0; i < ranks.size(); ++i) {
        auto device = DeviceManager::Instance()->GetDevice(DeviceType::kCUDA, ranks[i]);
        devices_.push_back(device);
        device_comm_map_[device] = comms_[i];
        global_group_rank_map_[device->rank().GlobalRank()] = i;
    }
}

void ProcessGroupNCCL::InitMultiProcess(const std::vector<int> &ranks) {
    int n_threads = global::GetNthreadPerProc();
    int global_proc_rank = global::GetGlobalProcRank();
    int lower_rank = global_proc_rank * n_threads;
    int upper_rank = (global_proc_rank + 1) * n_threads;

    ncclUniqueId nccl_id;

    int min_rank = std::ranges::min(ranks);
    if (min_rank < upper_rank && min_rank >= lower_rank) {
        is_main_process_ = true;

        ncclGetUniqueId(&nccl_id);
        WriteNcclUniqueId(nccl_id, name_);
    } else {
        ReadNcclUniqueId(nccl_id, name_);
    }

    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < n_threads; ++i) {
        int global_thread_rank = lower_rank + i;
        auto it = std::ranges::find(ranks, global_thread_rank);
        if (it != ranks.end()) {
            cudaSetDevice(i);

            ncclComm_t comm;
            int group_rank = std::distance(ranks.begin(), it);
            NCCL_CHECK(ncclCommInitRank(&comm, world_size_, nccl_id, group_rank));
            comms_.push_back(comm);

            auto device = DeviceManager::Instance()->GetDevice(DeviceType::kCUDA, i);
            global_group_rank_map_[device->rank().GlobalRank()] = group_rank;
            devices_.push_back(device);
            device_comm_map_[device] = comm;
        }
    }
    NCCL_CHECK(ncclGroupEnd());
}

void ProcessGroupNCCL::InitStreams() {
    int device_size = devices_.size();
    comm_streams_.resize(device_size);

    for (int i = 0; i < device_size; ++i) {
        devices_[i]->SetDevice();
        int low, high;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low, &high));
        CUDA_CHECK(cudaStreamCreateWithPriority(&comm_streams_[i], cudaStreamNonBlocking, high));
        device_stream_map_[devices_[i]] = comm_streams_[i];
    }
}

void ProcessGroupNCCL::Barrier(const Device *device) const {
    int one = 1;
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(device);
    cuda_device->SetDevice();

    auto comm = device_comm_map_.at(cuda_device);
    cudaStream_t stream = cuda_device->Stream();

    NCCL_CHECK(ncclAllReduce(&one, &one, 1, ncclInt, ncclSum, comm, stream));
}

std::shared_ptr<Work> ProcessGroupNCCL::AllReduce(const std::shared_ptr<Tensor> &tensor,
                                                  function::ReduceOpType reduce_op, bool async_op) const {
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

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }
}

std::shared_ptr<Work> ProcessGroupNCCL::AllGather(const std::shared_ptr<Tensor> &output,
                                                  const std::shared_ptr<Tensor> &input, bool async_op) const {
    const auto *device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();

    cudaStream_t compute_stream = device->Stream();
    cudaStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkNccl>(device, comm);

    cudaEvent_t ready_event = reinterpret_cast<cudaEvent_t>(work->ready_event());
    cudaEvent_t done_event = reinterpret_cast<cudaEvent_t>(work->done_event());

    CUDA_CHECK(cudaEventRecord(ready_event, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream, ready_event, 0));

    NCCL_CHECK(ncclAllGather(input->DataPtr(), output->DataPtr(), input->NumElements(),
                             kNcclDtypeMap.at(input->Dtype()), comm, comm_stream));

    CUDA_CHECK(cudaEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }
}

std::shared_ptr<Work> ProcessGroupNCCL::ReduceScatter(const std::shared_ptr<Tensor> &output,
                                                      const std::shared_ptr<Tensor> &input,
                                                      function::ReduceOpType reduce_op, bool async_op) const {
    const auto *device = dynamic_cast<const CudaDevice *>(input->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();

    cudaStream_t compute_stream = device->Stream();
    cudaStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkNccl>(device, comm);

    cudaEvent_t ready_event = reinterpret_cast<cudaEvent_t>(work->ready_event());
    cudaEvent_t done_event = reinterpret_cast<cudaEvent_t>(work->done_event());

    CUDA_CHECK(cudaEventRecord(ready_event, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream, ready_event, 0));

    NCCL_CHECK(ncclReduceScatter(input->DataPtr(), output->DataPtr(), output->NumElements(),
                                 kNcclDtypeMap.at(input->Dtype()), kNcclReduceOpMap.at(reduce_op), comm, comm_stream));

    CUDA_CHECK(cudaEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }
}

std::shared_ptr<Work> ProcessGroupNCCL::Send(std::vector<std::shared_ptr<Tensor>> tensors, int dest_rank,
                                             bool async_op) const {
    CHECK_GT(tensors.size(), 0);
    const auto *device = dynamic_cast<const CudaDevice *>(tensors[0]->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();

    cudaStream_t compute_stream = device->Stream();
    cudaStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkNccl>(device, comm);

    cudaEvent_t ready_event = reinterpret_cast<cudaEvent_t>(work->ready_event());
    cudaEvent_t done_event = reinterpret_cast<cudaEvent_t>(work->done_event());

    CUDA_CHECK(cudaEventRecord(ready_event, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream, ready_event, 0));

    for (int i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        CHECK_NOTNULL(tensor);

        CHECK_EQ(device, tensor->GetDevice());

        auto dtype = tensor->Dtype();
        auto nccl_dtype = kNcclDtypeMap.at(dtype);
        auto count = tensor->NumElements();
        void *buffer = tensor->DataPtr();
        CHECK_NOTNULL(buffer);

        NCCL_CHECK(ncclSend(buffer, count, nccl_dtype, dest_rank, comm, comm_stream));
    }

    CUDA_CHECK(cudaEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }

    return std::move(work);
}

std::shared_ptr<Work> ProcessGroupNCCL::Recv(std::vector<std::shared_ptr<Tensor>> tensors, int src_rank,
                                             bool async_op) const {
    CHECK_GT(tensors.size(), 0);
    const auto *device = dynamic_cast<const CudaDevice *>(tensors[0]->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();

    cudaStream_t compute_stream = device->Stream();
    cudaStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkNccl>(device, comm);

    cudaEvent_t ready_event = reinterpret_cast<cudaEvent_t>(work->ready_event());
    cudaEvent_t done_event = reinterpret_cast<cudaEvent_t>(work->done_event());

    CUDA_CHECK(cudaEventRecord(ready_event, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream, ready_event, 0));

    for (int i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        CHECK_NOTNULL(tensor);

        CHECK_EQ(device, tensor->GetDevice());

        auto dtype = tensor->Dtype();
        auto nccl_dtype = kNcclDtypeMap.at(dtype);
        auto count = tensor->NumElements();
        void *buffer = tensor->DataPtr();
        CHECK_NOTNULL(buffer);

        NCCL_CHECK(ncclRecv(buffer, count, nccl_dtype, src_rank, comm, compute_stream));
    }

    CUDA_CHECK(cudaEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }

    return std::move(work);
}

std::vector<std::shared_ptr<Tensor>>
ProcessGroupNCCL::BroadCast(const std::vector<std::shared_ptr<Tensor>> &input_tensors) const {
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
ProcessGroupNCCL::ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads,
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

std::vector<std::shared_ptr<Tensor>> ProcessGroupNCCL::Scatter(const std::shared_ptr<Tensor> &tensor,
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

std::shared_ptr<Tensor> ProcessGroupNCCL::Gather(const std::vector<std::shared_ptr<Tensor>> &tensors,
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
#endif

#ifdef USE_MCCL
ProcessGroupMCCL::ProcessGroupMCCL(const std::string &process_group_name, const std::vector<int> &ranks)
    : ProcessGroup(ranks.size(), process_group_name) {
    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));

    if (global::GetNnodes() == 1 && global::GetNprocPerNode() == 1) {
        InitSingleProcess(ranks);
    } else {
        InitMultiProcess(ranks);
    }
    InitStreams();

    MACA_CHECK(mcSetDevice(current_device));
}

ProcessGroupMCCL::~ProcessGroupMCCL() {
    if (is_main_process_) {
        CleanupMcclIdFile(name_);
    }

    for (auto &s : comm_streams_) {
        if (s) {
            mcStreamDestroy(s);
        }
    }
    for (auto &c : comms_) {
        if (c) {
            mcclCommDestroy(c);
        }
    }
}

void ProcessGroupMCCL::InitSingleProcess(const std::vector<int> &ranks) {
    comms_.resize(world_size_);
    MCCL_CHECK(mcclCommInitAll(comms_.data(), world_size_, ranks.data()));

    for (int i = 0; i < ranks.size(); ++i) {
        auto device = DeviceManager::Instance()->GetDevice(DeviceType::kMACA, ranks[i]);
        devices_.push_back(device);
        device_comm_map_[device] = comms_[i];
        global_group_rank_map_[device->rank().GlobalRank()] = i;
    }
}

void ProcessGroupMCCL::InitMultiProcess(const std::vector<int> &ranks) {
    int n_threads = global::GetNthreadPerProc();
    int global_proc_rank = global::GetGlobalProcRank();
    int lower_rank = global_proc_rank * n_threads;
    int upper_rank = (global_proc_rank + 1) * n_threads;

    mcclUniqueId mccl_id;

    int min_rank = std::ranges::min(ranks);
    if (min_rank < upper_rank && min_rank >= lower_rank) {
        is_main_process_ = true;

        mcclGetUniqueId(&mccl_id);
        WriteMcclUniqueId(mccl_id, name_);
    } else {
        ReadMcclUniqueId(mccl_id, name_);
    }

    MCCL_CHECK(mcclGroupStart());
    for (int i = 0; i < n_threads; ++i) {
        int global_thread_rank = lower_rank + i;
        auto it = std::ranges::find(ranks, global_thread_rank);
        if (it != ranks.end()) {
            mcSetDevice(i);

            mcclComm_t comm;
            int group_rank = std::distance(ranks.begin(), it);
            MCCL_CHECK(mcclCommInitRank(&comm, world_size_, mccl_id, group_rank));
            comms_.push_back(comm);

            auto device = DeviceManager::Instance()->GetDevice(DeviceType::kMACA, i);
            global_group_rank_map_[device->rank().GlobalRank()] = group_rank;
            devices_.push_back(device);
            device_comm_map_[device] = comm;
        }
    }
    MCCL_CHECK(mcclGroupEnd());
}

void ProcessGroupMCCL::InitStreams() {
    int device_size = devices_.size();
    comm_streams_.resize(device_size);

    for (int i = 0; i < device_size; ++i) {
        devices_[i]->SetDevice();
        int low, high;
        MACA_CHECK(mcDeviceGetStreamPriorityRange(&low, &high));
        MACA_CHECK(mcStreamCreateWithPriority(&comm_streams_[i], mcStreamNonBlocking, high));
        device_stream_map_[devices_[i]] = comm_streams_[i];
    }
}

void ProcessGroupMCCL::Barrier(const Device *device) const {
    const auto *maca_device = dynamic_cast<const MacaDevice *>(device);
    maca_device->SetDevice();
    auto comm = device_comm_map_.at(maca_device);
    mcStream_t stream = maca_device->Stream();

    int32_t h_one = 1;
    int32_t *d_one = nullptr;
    MACA_CHECK(mcMallocAsync((void **)&d_one, sizeof(int32_t), stream));
    MACA_CHECK(mcMemcpyAsync(d_one, &h_one, sizeof(int32_t), mcMemcpyHostToDevice, stream));

    MCCL_CHECK(mcclAllReduce(d_one, d_one, 1, mcclInt, mcclSum, comm, stream));

    MACA_CHECK(mcFreeAsync(d_one, stream));
}

std::shared_ptr<Work> ProcessGroupMCCL::AllReduce(const std::shared_ptr<Tensor> &tensor,
                                                  function::ReduceOpType reduce_op, bool async_op) const {
    void *buffer = tensor->DataPtr();
    const auto *device = dynamic_cast<const MacaDevice *>(tensor->GetDevice());
    device->SetDevice();

    auto comm = device_comm_map_.at(device);

    mcStream_t compute_stream = device->Stream();
    mcStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkMccl>(device, comm);

    mcEvent_t ready_event = reinterpret_cast<mcEvent_t>(work->ready_event());
    mcEvent_t done_event = reinterpret_cast<mcEvent_t>(work->done_event());

    MACA_CHECK(mcEventRecord(ready_event, compute_stream));
    MACA_CHECK(mcStreamWaitEvent(comm_stream, ready_event, 0));

    MCCL_CHECK(mcclAllReduce(buffer, buffer, tensor->NumElements(), kMcclDtypeMap.at(tensor->Dtype()),
                             kMcclReduceOpMap.at(reduce_op), comm, comm_stream));

    MACA_CHECK(mcEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }
}

std::shared_ptr<Work> ProcessGroupMCCL::AllGather(const std::shared_ptr<Tensor> &output,
                                                  const std::shared_ptr<Tensor> &input, bool async_op) const {
    const auto *device = dynamic_cast<const MacaDevice *>(input->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();

    mcStream_t compute_stream = device->Stream();
    mcStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkMccl>(device, comm);

    mcEvent_t ready_event = reinterpret_cast<mcEvent_t>(work->ready_event());
    mcEvent_t done_event = reinterpret_cast<mcEvent_t>(work->done_event());

    MACA_CHECK(mcEventRecord(ready_event, compute_stream));
    MACA_CHECK(mcStreamWaitEvent(comm_stream, ready_event, 0));

    MCCL_CHECK(mcclAllGather(input->DataPtr(), output->DataPtr(), input->NumElements(),
                             kMcclDtypeMap.at(input->Dtype()), comm, comm_stream));

    MACA_CHECK(mcEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }
}

std::shared_ptr<Work> ProcessGroupMCCL::ReduceScatter(const std::shared_ptr<Tensor> &output,
                                                      const std::shared_ptr<Tensor> &input,
                                                      function::ReduceOpType reduce_op, bool async_op) const {
    const auto *device = dynamic_cast<const MacaDevice *>(input->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();

    mcStream_t compute_stream = device->Stream();
    mcStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkMccl>(device, comm);

    mcEvent_t ready_event = reinterpret_cast<mcEvent_t>(work->ready_event());
    mcEvent_t done_event = reinterpret_cast<mcEvent_t>(work->done_event());

    MACA_CHECK(mcEventRecord(ready_event, compute_stream));
    MACA_CHECK(mcStreamWaitEvent(comm_stream, ready_event, 0));

    MCCL_CHECK(mcclReduceScatter(input->DataPtr(), output->DataPtr(), output->NumElements(),
                                 kMcclDtypeMap.at(input->Dtype()), kMcclReduceOpMap.at(reduce_op), comm, comm_stream));

    MACA_CHECK(mcEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }
}

std::shared_ptr<Work> ProcessGroupMCCL::Send(std::vector<std::shared_ptr<Tensor>> tensors, int dest_rank,
                                             bool async_op) const {
    CHECK_GT(tensors.size(), 0);
    const auto *device = dynamic_cast<const MacaDevice *>(tensors[0]->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();

    mcStream_t compute_stream = device->Stream();
    mcStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkMccl>(device, comm);

    mcEvent_t ready_event = reinterpret_cast<mcEvent_t>(work->ready_event());
    mcEvent_t done_event = reinterpret_cast<mcEvent_t>(work->done_event());

    MACA_CHECK(mcEventRecord(ready_event, compute_stream));
    MACA_CHECK(mcStreamWaitEvent(comm_stream, ready_event, 0));

    for (int i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        CHECK_NOTNULL(tensor);

        CHECK_EQ(device, tensor->GetDevice());

        auto dtype = tensor->Dtype();
        auto mccl_dtype = kMcclDtypeMap.at(dtype);
        auto count = tensor->NumElements();
        void *buffer = tensor->DataPtr();
        CHECK_NOTNULL(buffer);

        MCCL_CHECK(mcclSend(buffer, count, mccl_dtype, dest_rank, comm, comm_stream));
    }

    MACA_CHECK(mcEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }

    return std::move(work);
}

std::shared_ptr<Work> ProcessGroupMCCL::Recv(std::vector<std::shared_ptr<Tensor>> tensors, int src_rank,
                                             bool async_op) const {
    CHECK_GT(tensors.size(), 0);
    const auto *device = dynamic_cast<const MacaDevice *>(tensors[0]->GetDevice());
    auto comm = device_comm_map_.at(device);

    device->SetDevice();

    mcStream_t compute_stream = device->Stream();
    mcStream_t comm_stream = device_stream_map_.at(device);

    auto work = std::make_shared<WorkMccl>(device, comm);

    mcEvent_t ready_event = reinterpret_cast<mcEvent_t>(work->ready_event());
    mcEvent_t done_event = reinterpret_cast<mcEvent_t>(work->done_event());

    MACA_CHECK(mcEventRecord(ready_event, compute_stream));
    MACA_CHECK(mcStreamWaitEvent(comm_stream, ready_event, 0));

    for (int i = 0; i < tensors.size(); ++i) {
        auto tensor = tensors[i];
        CHECK_NOTNULL(tensor);

        CHECK_EQ(device, tensor->GetDevice());

        auto dtype = tensor->Dtype();
        auto mccl_dtype = kMcclDtypeMap.at(dtype);
        auto count = tensor->NumElements();
        void *buffer = tensor->DataPtr();
        CHECK_NOTNULL(buffer);

        MCCL_CHECK(mcclRecv(buffer, count, mccl_dtype, src_rank, comm, compute_stream));
    }

    MACA_CHECK(mcEventRecord(done_event, comm_stream));

    if (async_op) {
        return std::move(work);
    } else {
        work->WaitNonBlocking();
        return nullptr;
    }

    return std::move(work);
}

std::vector<std::shared_ptr<Tensor>>
ProcessGroupMCCL::BroadCast(const std::vector<std::shared_ptr<Tensor>> &input_tensors) const {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<mcStream_t> streams;
    std::vector<mcclComm_t> comms;
    std::vector<const Device *> devices;

    CHECK_EQ(world_size_, comms_.size());

    for (size_t i = 0; i < world_size_; ++i) {
        auto device = devices_[i];
        for (const auto &input_tensor : input_tensors) {
            outputs.push_back(std::make_shared<Tensor>(input_tensor->Dims(), input_tensor->Dtype(), device));
        }
        devices.push_back(device);
        streams.push_back(dynamic_cast<const MacaDevice *>(device)->Stream());
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

    MCCL_CHECK(mcclGroupStart());
    for (size_t i = 0; i < devices.size(); ++i) {
        devices[i]->SetDevice();
        for (size_t j = 0; j < input_tensors.size(); ++j) {
            const auto &input_tensor = input_tensors[j];
            const auto dtype = input_tensor->Dtype();
            auto mccl_dtype = kMcclDtypeMap.at(dtype);
            auto count = input_tensor->NumElements();
            void *send_buffer = (devices[i] == input_tensor->GetDevice() ? input_tensor->DataPtr() : nullptr);
            MCCL_CHECK(mcclBroadcast(send_buffer, outputs[i * input_tensors.size() + j]->DataPtr(), count, mccl_dtype,
                                     0, comms[i], streams[i]));
        }
    }
    MCCL_CHECK(mcclGroupEnd());

    return outputs;
}

std::vector<std::shared_ptr<Tensor>>
ProcessGroupMCCL::ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads,
                                     const Device *destination) const {
    // grads: [devices, tensors]
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<mcStream_t> streams;
    std::vector<mcclComm_t> comms;
    std::vector<const Device *> devices;

    for (size_t i = 0; i < grads[0].size(); ++i) {
        outputs.push_back(std::make_shared<Tensor>(grads[0][i]->Dims(), grads[0][i]->Dtype(), destination));
        outputs[i]->Fill<float>(0.0f);
    }
    for (size_t i = 0; i < grads.size(); ++i) {
        devices.push_back(grads[i][0]->GetDevice());
        streams.push_back(dynamic_cast<const MacaDevice *>(devices[i])->Stream());
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

    MCCL_CHECK(mcclGroupStart());
    for (size_t i = 0; i < grads.size(); ++i) {
        devices[i]->SetDevice();
        for (size_t j = 0; j < grads[i].size(); ++j) {
            const auto &grad = grads[i][j];
            const auto dtype = grad->Dtype();
            auto mccl_dtype = kMcclDtypeMap.at(dtype);
            auto count = grad->NumElements();
            void *send_buffer = grad->DataPtr();
            MCCL_CHECK(
                mcclReduce(send_buffer, outputs[j]->DataPtr(), count, mccl_dtype, mcclSum, 0, comms[i], streams[i]));
        }
    }
    MCCL_CHECK(mcclGroupEnd());

    return outputs;
}

std::vector<std::shared_ptr<Tensor>> ProcessGroupMCCL::Scatter(const std::shared_ptr<Tensor> &tensor,
                                                               std::vector<const Device *> devices, int64_t dim) const {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<std::shared_ptr<Tensor>> split_tensors = tensor->Split(tensor->Dims()[dim] / devices.size(), dim);
    std::vector<mcStream_t> streams;
    std::vector<mcclComm_t> comms;
    int src_rank = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (tensor->GetDevice() == devices[i]) {
            src_rank = i;
        }
        outputs.push_back(std::make_shared<Tensor>(split_tensors[i]->Dims(), split_tensors[i]->Dtype(), devices[i]));
        streams.push_back(dynamic_cast<const MacaDevice *>(devices[i])->Stream());
        comms.push_back(device_comm_map_.at(devices[i]));
    }

    CHECK_NE(src_rank, -1) << "Source device not found in input devices";

    MCCL_CHECK(mcclGroupStart());
    const auto dtype = tensor->Dtype();
    auto mccl_dtype = kMcclDtypeMap.at(dtype);

    for (size_t i = 0; i < devices.size(); ++i) {
        devices[i]->SetDevice();
        const auto dtype = tensor->Dtype();
        auto mccl_dtype = kMcclDtypeMap.at(dtype);
        MCCL_CHECK(mcclSend(split_tensors[i]->DataPtr(), split_tensors[i]->NumElements(), mccl_dtype, i,
                            comms[src_rank], streams[src_rank]));
        MCCL_CHECK(
            mcclRecv(outputs[i]->DataPtr(), outputs[i]->NumElements(), mccl_dtype, src_rank, comms[i], streams[i]));
    }
    MCCL_CHECK(mcclGroupEnd());
    return outputs;
}

std::shared_ptr<Tensor> ProcessGroupMCCL::Gather(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                                 const Device *destination, int64_t dim) const {
    std::vector<std::shared_ptr<Tensor>> outouts;
    int64_t num_devices = tensors.size();
    auto dtype = tensors[0]->Dtype();
    auto mccl_dtype = kMcclDtypeMap.at(dtype);

    int64_t total_dim = 0;

    std::vector<mcStream_t> streams;
    std::vector<mcclComm_t> comms;
    std::vector<const Device *> devices;

    int dest_rank = -1;
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto device = tensors[i]->GetDevice();
        if (device == destination) {
            dest_rank = i;
        }
        streams.push_back(dynamic_cast<const MacaDevice *>(device)->Stream());
        comms.push_back(device_comm_map_.at(device));
        devices.push_back(device);

        total_dim += tensors[i]->Dims()[dim];
    }

    std::vector<int64_t> out_dims = tensors[0]->Dims();
    out_dims[dim] = total_dim;
    auto output = std::make_shared<Tensor>(out_dims, dtype, destination);

    CHECK_NE(dest_rank, -1) << "Destination device not found in input tensors's devices";

    MCCL_CHECK(mcclGroupStart());
    int64_t offset = 0;

    for (size_t i = 0; i < num_devices; ++i) {
        devices[i]->SetDevice();
        auto &tensor = tensors[i];
        size_t num_elements = tensor->NumElements();
        void *send_ptr = tensor->DataPtr();

        auto recv_ptr = static_cast<int8_t *>(output->DataPtr()) + offset;

        MCCL_CHECK(mcclSend(send_ptr, num_elements, mccl_dtype, dest_rank, comms[i], streams[i]));
        MCCL_CHECK(mcclRecv(recv_ptr, num_elements, mccl_dtype, i, comms[dest_rank], streams[dest_rank]));

        offset += tensor->SizeInBytes();
    }

    MCCL_CHECK(mcclGroupEnd());
    return output;
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
    // TODO(dcj): create device-specific ProcessGroup based on the registered device later
#ifdef USE_NCCL
    return GetOrCreate(name, [&]() { return std::make_unique<ProcessGroupNCCL>(name, device_indices); });
#elif defined(USE_MCCL)
    return GetOrCreate(name, [&]() { return std::make_unique<ProcessGroupMCCL>(name, device_indices); });
#else
    return nullptr;
#endif
}

const ProcessGroup *ProcessGroupFactory::GetOrCreate(const std::string &name, const std::vector<int> &device_indices) {
    // TODO(dcj): create device-specific ProcessGroup based on the registered device later
#ifdef USE_NCCL
    return GetOrCreate(name, [&]() { return std::make_unique<ProcessGroupNCCL>(name, device_indices); });
#elif defined(USE_MCCL)
    return GetOrCreate(name, [&]() { return std::make_unique<ProcessGroupMCCL>(name, device_indices); });
#else
    return nullptr;
#endif
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

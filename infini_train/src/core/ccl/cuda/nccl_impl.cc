#include "infini_train/src/core/ccl/cuda/nccl_impl.h"

#ifdef USE_NCCL

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <nccl.h>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/stream.h"
#include "infini_train/include/device.h"
#include "infini_train/src/core/ccl/cuda/nccl_common.h"
#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::core::cuda {
namespace {

inline const std::unordered_map<DataType, ncclDataType_t> kNcclDtypeMap = {
    {DataType::kUINT8, ncclUint8},       {DataType::kINT8, ncclInt8},     {DataType::kUINT32, ncclUint32},
    {DataType::kINT32, ncclInt32},       {DataType::kUINT64, ncclUint64}, {DataType::kINT64, ncclInt64},
    {DataType::kBFLOAT16, ncclBfloat16}, {DataType::kFLOAT16, ncclHalf},  {DataType::kFLOAT32, ncclFloat32},
    {DataType::kFLOAT64, ncclFloat64},
};

inline const std::unordered_map<nn::parallel::function::ReduceOpType, ncclRedOp_t> kNcclReduceOpMap = {
    {nn::parallel::function::ReduceOpType::kSum, ncclSum}, {nn::parallel::function::ReduceOpType::kProd, ncclProd},
    {nn::parallel::function::ReduceOpType::kMin, ncclMin}, {nn::parallel::function::ReduceOpType::kMax, ncclMax},
    {nn::parallel::function::ReduceOpType::kAvg, ncclAvg},
};

inline std::string NcclFileName(const std::string &name, bool tmp = false) {
    return std::format("ncclUniqueId_{}.{}", name, tmp ? "tmp" : "bin");
}

inline ncclComm_t GetNcclComm(const CclComm *comm) {
    auto *nccl_comm = dynamic_cast<const NcclComm *>(comm);
    CHECK_NOTNULL(nccl_comm);
    return nccl_comm->nccl_comm();
}

inline void SetNcclComm(CclComm *comm, ncclComm_t nccl_comm) {
    auto *typed_comm = dynamic_cast<NcclComm *>(comm);
    CHECK_NOTNULL(typed_comm);
    typed_comm->set_nccl_comm(nccl_comm);
}

inline const ncclUniqueId &GetNcclUniqueId(const CclUniqueId &unique_id) {
    auto *nccl_unique_id = dynamic_cast<const NcclUniqueId *>(&unique_id);
    CHECK_NOTNULL(nccl_unique_id);
    return *nccl_unique_id->nccl_unique_id();
}

inline cudaStream_t GetCudaStream(Stream *stream) {
    auto *cuda_stream = dynamic_cast<CudaStream *>(stream);
    CHECK_NOTNULL(cuda_stream);
    return cuda_stream->cuda_stream();
}

} // namespace

Device::DeviceType NcclImpl::Type() const { return Device::DeviceType::kCUDA; }

void NcclImpl::GroupStart() const { NCCL_CHECK(ncclGroupStart()); }

void NcclImpl::GroupEnd() const { NCCL_CHECK(ncclGroupEnd()); }

void NcclImpl::CommGetAsyncError(const CclComm *comm, CclStatus *async_error) const {
    ncclResult_t nccl_async_error = ncclSuccess;
    NCCL_CHECK(ncclCommGetAsyncError(GetNcclComm(comm), &nccl_async_error));
    if (async_error != nullptr) {
        *async_error = (nccl_async_error == ncclSuccess) ? CclStatus::kSuccess : CclStatus::kError;
    }
}

void NcclImpl::CreateComm(CclComm **comm) const {
    CHECK_NOTNULL(comm);
    *comm = new NcclComm();
}

void NcclImpl::CreateUniqueId(CclUniqueId **unique_id) const {
    CHECK_NOTNULL(unique_id);
    *unique_id = new NcclUniqueId();
}

void NcclImpl::GetUniqueId(CclUniqueId *unique_id) const {
    auto *nccl_unique_id = dynamic_cast<NcclUniqueId *>(unique_id);
    CHECK_NOTNULL(nccl_unique_id);
    NCCL_CHECK(ncclGetUniqueId(nccl_unique_id->nccl_unique_id()));
}

void NcclImpl::WriteUniqueId(const CclUniqueId &unique_id, const std::string &pg_name) const {
    const ncclUniqueId &nccl_id = GetNcclUniqueId(unique_id);

    std::string tmp_path = NcclFileName(pg_name, true);
    std::ofstream ofs(tmp_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(&nccl_id), sizeof(nccl_id));
    ofs.close();

    std::rename(tmp_path.c_str(), NcclFileName(pg_name).c_str());
}

void NcclImpl::ReadUniqueId(CclUniqueId *unique_id, const std::string &pg_name) const {
    auto *nccl_unique_id = dynamic_cast<NcclUniqueId *>(unique_id);
    CHECK_NOTNULL(nccl_unique_id);
    std::string file_path = NcclFileName(pg_name);

    while (!std::filesystem::exists(file_path)) { std::this_thread::sleep_for(std::chrono::microseconds(1000)); }

    std::ifstream ifs(file_path, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(nccl_unique_id->nccl_unique_id()), sizeof(ncclUniqueId));
    ifs.close();
}

void NcclImpl::CleanupUniqueIdFile(const std::string &pg_name) const {
    std::string file_path = NcclFileName(pg_name);
    if (std::filesystem::exists(file_path)) {
        std::filesystem::remove(file_path);
    }
}

void NcclImpl::CommInitAll(CclComm **comms, int ndev, const int *devlist) const {
    CHECK_NOTNULL(comms);
    CHECK_GT(ndev, 0);
    CHECK_NOTNULL(devlist);

    std::vector<ncclComm_t> nccl_comms(static_cast<size_t>(ndev), nullptr);
    NCCL_CHECK(ncclCommInitAll(nccl_comms.data(), ndev, devlist));
    for (int i = 0; i < ndev; ++i) { SetNcclComm(comms[i], nccl_comms[static_cast<size_t>(i)]); }
}

void NcclImpl::CommInitRank(CclComm *comm, int nranks, const CclUniqueId &unique_id, int rank) const {

    CHECK_NOTNULL(comm);
    CHECK_GT(nranks, 0);

    ncclComm_t nccl_comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, GetNcclUniqueId(unique_id), rank));
    SetNcclComm(comm, nccl_comm);
}

void NcclImpl::CommDestroy(CclComm *comm) const {
    if (comm == nullptr) {
        return;
    }
    NCCL_CHECK(ncclCommDestroy(GetNcclComm(comm)));
    SetNcclComm(comm, nullptr);
}

void NcclImpl::AllReduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                         nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm, Stream *stream) const {
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, kNcclDtypeMap.at(dtype), kNcclReduceOpMap.at(reduce_op),
                             GetNcclComm(comm), GetCudaStream(stream)));
}

void NcclImpl::Broadcast(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, int root,
                         const CclComm *comm, Stream *stream) const {
    NCCL_CHECK(ncclBroadcast(sendbuff, recvbuff, count, kNcclDtypeMap.at(dtype), root, GetNcclComm(comm),
                             GetCudaStream(stream)));
}

void NcclImpl::Reduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                      nn::parallel::function::ReduceOpType reduce_op, int root, const CclComm *comm,
                      Stream *stream) const {
    NCCL_CHECK(ncclReduce(sendbuff, recvbuff, count, kNcclDtypeMap.at(dtype), kNcclReduceOpMap.at(reduce_op), root,
                          GetNcclComm(comm), GetCudaStream(stream)));
}

void NcclImpl::AllGather(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, const CclComm *comm,
                         Stream *stream) const {
    NCCL_CHECK(
        ncclAllGather(sendbuff, recvbuff, count, kNcclDtypeMap.at(dtype), GetNcclComm(comm), GetCudaStream(stream)));
}

void NcclImpl::ReduceScatter(const void *sendbuff, void *recvbuff, size_t recv_count, DataType dtype,
                             nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm,
                             Stream *stream) const {
    NCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recv_count, kNcclDtypeMap.at(dtype),
                                 kNcclReduceOpMap.at(reduce_op), GetNcclComm(comm), GetCudaStream(stream)));
}

void NcclImpl::Send(const void *buff, size_t count, DataType dtype, int peer, const CclComm *comm,
                    Stream *stream) const {
    NCCL_CHECK(ncclSend(buff, count, kNcclDtypeMap.at(dtype), peer, GetNcclComm(comm), GetCudaStream(stream)));
}

void NcclImpl::Recv(void *buff, size_t count, DataType dtype, int peer, const CclComm *comm, Stream *stream) const {
    NCCL_CHECK(ncclRecv(buff, count, kNcclDtypeMap.at(dtype), peer, GetNcclComm(comm), GetCudaStream(stream)));
}

INFINI_TRAIN_REGISTER_CCL_IMPL(Device::DeviceType::kCUDA, NcclImpl)

} // namespace infini_train::core::cuda

#endif

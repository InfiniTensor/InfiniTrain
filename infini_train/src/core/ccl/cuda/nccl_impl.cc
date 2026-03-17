#include "infini_train/src/core/ccl/cuda/nccl_impl.h"

#include <nccl.h>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/ccl/cuda/nccl_common.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

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

void NcclImpl::GetAsyncError(const CclComm *comm, CclStatus *async_error) const {
    ncclResult_t nccl_async_error = ncclSuccess;
    NCCL_CHECK(ncclCommGetAsyncError(GetNcclComm(comm), &nccl_async_error));
    if (async_error != nullptr) {
        *async_error = (nccl_async_error == ncclSuccess) ? CclStatus::kSuccess : CclStatus::kError;
    }
}

void NcclImpl::GetUniqueId(CclUniqueId **unique_id) const {
    CHECK_NOTNULL(unique_id);
    if (*unique_id == nullptr) {
        *unique_id = new NcclUniqueId();
    }
    auto *nccl_unique_id = dynamic_cast<NcclUniqueId *>(*unique_id);
    CHECK_NOTNULL(nccl_unique_id);
    NCCL_CHECK(ncclGetUniqueId(nccl_unique_id->nccl_unique_id()));
}

void NcclImpl::CommInitAll(CclComm **comms, int ndev, const int *devlist) const {
    CHECK_NOTNULL(comms);
    CHECK_GT(ndev, 0);
    CHECK_NOTNULL(devlist);

    std::vector<ncclComm_t> nccl_comms(static_cast<size_t>(ndev), nullptr);
    NCCL_CHECK(ncclCommInitAll(nccl_comms.data(), ndev, devlist));
    for (int i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            comms[i] = new NcclComm();
        }
        SetNcclComm(comms[i], nccl_comms[static_cast<size_t>(i)]);
    }
}

void NcclImpl::CommInitRank(CclComm **comm, int nranks, const CclUniqueId &unique_id, int rank) const {
    CHECK_NOTNULL(comm);
    CHECK_GT(nranks, 0);

    if (*comm == nullptr) {
        *comm = new NcclComm();
    }

    ncclComm_t nccl_comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, GetNcclUniqueId(unique_id), rank));
    SetNcclComm(*comm, nccl_comm);
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

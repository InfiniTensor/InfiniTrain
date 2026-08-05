#include "infini_train/src/core/ccl/dcu/rccl_impl.h"

#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/ccl/dcu/rccl_common.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::core::dcu {
namespace {

inline const std::unordered_map<DataType, ncclDataType_t> kRcclDtypeMap = {
    {DataType::kUINT8, ncclUint8},       {DataType::kINT8, ncclInt8},     {DataType::kUINT32, ncclUint32},
    {DataType::kINT32, ncclInt32},       {DataType::kUINT64, ncclUint64}, {DataType::kINT64, ncclInt64},
    {DataType::kBFLOAT16, ncclBfloat16}, {DataType::kFLOAT16, ncclHalf},  {DataType::kFLOAT32, ncclFloat32},
    {DataType::kFLOAT64, ncclFloat64},
};

inline const std::unordered_map<nn::parallel::function::ReduceOpType, ncclRedOp_t> kRcclReduceOpMap = {
    {nn::parallel::function::ReduceOpType::kSum, ncclSum}, {nn::parallel::function::ReduceOpType::kProd, ncclProd},
    {nn::parallel::function::ReduceOpType::kMin, ncclMin}, {nn::parallel::function::ReduceOpType::kMax, ncclMax},
    {nn::parallel::function::ReduceOpType::kAvg, ncclAvg},
};

inline ncclComm_t GetRcclComm(const CclComm *comm) {
    auto *nccl_comm = dynamic_cast<const RcclComm *>(comm);
    CHECK_NOTNULL(nccl_comm);
    return nccl_comm->nccl_comm();
}

inline void SetRcclComm(CclComm *comm, ncclComm_t nccl_comm) {
    auto *typed_comm = dynamic_cast<RcclComm *>(comm);
    CHECK_NOTNULL(typed_comm);
    typed_comm->set_nccl_comm(nccl_comm);
}

inline const ncclUniqueId &GetRcclUniqueId(const CclUniqueId &unique_id) {
    auto *nccl_unique_id = dynamic_cast<const RcclUniqueId *>(&unique_id);
    CHECK_NOTNULL(nccl_unique_id);
    return *nccl_unique_id->nccl_unique_id();
}

inline hipStream_t GetDcuStream(Stream *stream) {
    auto *cuda_stream = dynamic_cast<DcuStream *>(stream);
    CHECK_NOTNULL(cuda_stream);
    return cuda_stream->hip_stream();
}

} // namespace

Device::DeviceType RcclImpl::Type() const { return Device::DeviceType::kDCU; }

void RcclImpl::GroupStart() const { NCCL_CHECK(ncclGroupStart()); }

void RcclImpl::GroupEnd() const { NCCL_CHECK(ncclGroupEnd()); }

void RcclImpl::GetAsyncError(const CclComm *comm, CclStatus *async_error) const {
    ncclResult_t nccl_async_error = ncclSuccess;
    NCCL_CHECK(ncclCommGetAsyncError(GetRcclComm(comm), &nccl_async_error));
    if (async_error != nullptr) {
        *async_error = (nccl_async_error == ncclSuccess) ? CclStatus::kSuccess : CclStatus::kError;
    }
}

void RcclImpl::GetUniqueId(CclUniqueId **unique_id) const {
    CHECK_NOTNULL(unique_id);
    if (*unique_id == nullptr) {
        *unique_id = new RcclUniqueId();
    }
    auto *nccl_unique_id = dynamic_cast<RcclUniqueId *>(*unique_id);
    CHECK_NOTNULL(nccl_unique_id);
    NCCL_CHECK(ncclGetUniqueId(nccl_unique_id->nccl_unique_id()));
}

void RcclImpl::CommInitAll(CclComm **comms, int ndev, const int *devlist) const {
    CHECK_NOTNULL(comms);
    CHECK_GT(ndev, 0);
    CHECK_NOTNULL(devlist);

    std::vector<ncclComm_t> nccl_comms(static_cast<size_t>(ndev), nullptr);
    NCCL_CHECK(ncclCommInitAll(nccl_comms.data(), ndev, devlist));
    for (int i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            comms[i] = new RcclComm();
        }
        SetRcclComm(comms[i], nccl_comms[static_cast<size_t>(i)]);
    }
}

void RcclImpl::CommInitRank(CclComm **comm, int nranks, const CclUniqueId &unique_id, int rank) const {
    CHECK_NOTNULL(comm);
    CHECK_GT(nranks, 0);

    if (*comm == nullptr) {
        *comm = new RcclComm();
    }

    ncclComm_t nccl_comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, GetRcclUniqueId(unique_id), rank));
    SetRcclComm(*comm, nccl_comm);
}

void RcclImpl::CommDestroy(CclComm *comm) const {
    if (comm == nullptr) {
        return;
    }
    NCCL_CHECK(ncclCommDestroy(GetRcclComm(comm)));
    SetRcclComm(comm, nullptr);
}

void RcclImpl::AllReduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                         nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm, Stream *stream) const {
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, kRcclDtypeMap.at(dtype), kRcclReduceOpMap.at(reduce_op),
                             GetRcclComm(comm), GetDcuStream(stream)));
}

void RcclImpl::Broadcast(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, int root,
                         const CclComm *comm, Stream *stream) const {
    NCCL_CHECK(ncclBroadcast(sendbuff, recvbuff, count, kRcclDtypeMap.at(dtype), root, GetRcclComm(comm),
                             GetDcuStream(stream)));
}

void RcclImpl::Reduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                      nn::parallel::function::ReduceOpType reduce_op, int root, const CclComm *comm,
                      Stream *stream) const {
    NCCL_CHECK(ncclReduce(sendbuff, recvbuff, count, kRcclDtypeMap.at(dtype), kRcclReduceOpMap.at(reduce_op), root,
                          GetRcclComm(comm), GetDcuStream(stream)));
}

void RcclImpl::AllGather(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, const CclComm *comm,
                         Stream *stream) const {
    NCCL_CHECK(
        ncclAllGather(sendbuff, recvbuff, count, kRcclDtypeMap.at(dtype), GetRcclComm(comm), GetDcuStream(stream)));
}

void RcclImpl::ReduceScatter(const void *sendbuff, void *recvbuff, size_t recv_count, DataType dtype,
                             nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm,
                             Stream *stream) const {
    NCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recv_count, kRcclDtypeMap.at(dtype),
                                 kRcclReduceOpMap.at(reduce_op), GetRcclComm(comm), GetDcuStream(stream)));
}

void RcclImpl::Send(const void *buff, size_t count, DataType dtype, int peer, const CclComm *comm,
                    Stream *stream) const {
    NCCL_CHECK(ncclSend(buff, count, kRcclDtypeMap.at(dtype), peer, GetRcclComm(comm), GetDcuStream(stream)));
}

void RcclImpl::Recv(void *buff, size_t count, DataType dtype, int peer, const CclComm *comm, Stream *stream) const {
    NCCL_CHECK(ncclRecv(buff, count, kRcclDtypeMap.at(dtype), peer, GetRcclComm(comm), GetDcuStream(stream)));
}

INFINI_TRAIN_REGISTER_CCL_IMPL(Device::DeviceType::kDCU, RcclImpl)

} // namespace infini_train::core::dcu

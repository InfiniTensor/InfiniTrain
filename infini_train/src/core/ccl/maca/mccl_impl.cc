#include "infini_train/src/core/ccl/maca/mccl_impl.h"

#include <mccl.h>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/maca/common_maca.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/ccl/maca/mccl_common.h"
#include "infini_train/src/core/runtime/maca/maca_runtime_common.h"

namespace infini_train::core::maca {
namespace {

inline const std::unordered_map<DataType, mcclDataType_t> kMcclDtypeMap = {
    {DataType::kUINT8, mcclUint8},       {DataType::kINT8, mcclInt8},     {DataType::kUINT32, mcclUint32},
    {DataType::kINT32, mcclInt32},       {DataType::kUINT64, mcclUint64}, {DataType::kINT64, mcclInt64},
    {DataType::kBFLOAT16, mcclBfloat16}, {DataType::kFLOAT16, mcclHalf},  {DataType::kFLOAT32, mcclFloat32},
    {DataType::kFLOAT64, mcclFloat64},
};

inline const std::unordered_map<nn::parallel::function::ReduceOpType, mcclRedOp_t> kMcclReduceOpMap = {
    {nn::parallel::function::ReduceOpType::kSum, mcclSum}, {nn::parallel::function::ReduceOpType::kProd, mcclProd},
    {nn::parallel::function::ReduceOpType::kMin, mcclMin}, {nn::parallel::function::ReduceOpType::kMax, mcclMax},
    {nn::parallel::function::ReduceOpType::kAvg, mcclAvg},
};

inline mcclComm_t GetMcclComm(const CclComm *comm) {
    auto *mccl_comm = dynamic_cast<const McclComm *>(comm);
    CHECK_NOTNULL(mccl_comm);
    return mccl_comm->mccl_comm();
}

inline void SetMcclComm(CclComm *comm, mcclComm_t mccl_comm) {
    auto *typed_comm = dynamic_cast<McclComm *>(comm);
    CHECK_NOTNULL(typed_comm);
    typed_comm->set_mccl_comm(mccl_comm);
}

inline const mcclUniqueId &GetMcclUniqueId(const CclUniqueId &unique_id) {
    auto *mccl_unique_id = dynamic_cast<const McclUniqueId *>(&unique_id);
    CHECK_NOTNULL(mccl_unique_id);
    return *mccl_unique_id->mccl_unique_id();
}

inline mcStream_t GetMacaStream(Stream *stream) {
    auto *maca_stream = dynamic_cast<MacaStream *>(stream);
    CHECK_NOTNULL(maca_stream);
    return maca_stream->maca_stream();
}

} // namespace

Device::DeviceType McclImpl::Type() const { return Device::DeviceType::kMACA; }

void McclImpl::GroupStart() const { MCCL_CHECK(mcclGroupStart()); }

void McclImpl::GroupEnd() const { MCCL_CHECK(mcclGroupEnd()); }

void McclImpl::GetAsyncError(const CclComm *comm, CclStatus *async_error) const {
    mcclResult_t mccl_async_error = mcclSuccess;
    MCCL_CHECK(mcclCommGetAsyncError(GetMcclComm(comm), &mccl_async_error));
    if (async_error != nullptr) {
        *async_error = (mccl_async_error == mcclSuccess) ? CclStatus::kSuccess : CclStatus::kError;
    }
}

void McclImpl::GetUniqueId(CclUniqueId **unique_id) const {
    CHECK_NOTNULL(unique_id);
    if (*unique_id == nullptr) {
        *unique_id = new McclUniqueId();
    }
    auto *mccl_unique_id = dynamic_cast<McclUniqueId *>(*unique_id);
    CHECK_NOTNULL(mccl_unique_id);
    MCCL_CHECK(mcclGetUniqueId(mccl_unique_id->mccl_unique_id()));
}

void McclImpl::CommInitAll(CclComm **comms, int ndev, const int *devlist) const {
    CHECK_NOTNULL(comms);
    CHECK_GT(ndev, 0);
    CHECK_NOTNULL(devlist);

    std::vector<mcclComm_t> mccl_comms(static_cast<size_t>(ndev), nullptr);
    MCCL_CHECK(mcclCommInitAll(mccl_comms.data(), ndev, devlist));
    for (int i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            comms[i] = new McclComm();
        }
        SetMcclComm(comms[i], mccl_comms[static_cast<size_t>(i)]);
    }
}

void McclImpl::CommInitRank(CclComm **comm, int nranks, const CclUniqueId &unique_id, int rank) const {
    CHECK_NOTNULL(comm);
    CHECK_GT(nranks, 0);

    if (*comm == nullptr) {
        *comm = new McclComm();
    }

    mcclComm_t mccl_comm = nullptr;
    MCCL_CHECK(mcclCommInitRank(&mccl_comm, nranks, GetMcclUniqueId(unique_id), rank));
    SetMcclComm(*comm, mccl_comm);
}

void McclImpl::CommDestroy(CclComm *comm) const {
    if (comm == nullptr) {
        return;
    }
    MCCL_CHECK(mcclCommDestroy(GetMcclComm(comm)));
    SetMcclComm(comm, nullptr);
}

void McclImpl::AllReduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                         nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm, Stream *stream) const {
    MCCL_CHECK(mcclAllReduce(sendbuff, recvbuff, count, kMcclDtypeMap.at(dtype), kMcclReduceOpMap.at(reduce_op),
                             GetMcclComm(comm), GetMacaStream(stream)));
}

void McclImpl::Broadcast(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, int root,
                         const CclComm *comm, Stream *stream) const {
    MCCL_CHECK(mcclBroadcast(sendbuff, recvbuff, count, kMcclDtypeMap.at(dtype), root, GetMcclComm(comm),
                             GetMacaStream(stream)));
}

void McclImpl::Reduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                      nn::parallel::function::ReduceOpType reduce_op, int root, const CclComm *comm,
                      Stream *stream) const {
    MCCL_CHECK(mcclReduce(sendbuff, recvbuff, count, kMcclDtypeMap.at(dtype), kMcclReduceOpMap.at(reduce_op), root,
                          GetMcclComm(comm), GetMacaStream(stream)));
}

void McclImpl::AllGather(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, const CclComm *comm,
                         Stream *stream) const {
    MCCL_CHECK(
        mcclAllGather(sendbuff, recvbuff, count, kMcclDtypeMap.at(dtype), GetMcclComm(comm), GetMacaStream(stream)));
}

void McclImpl::ReduceScatter(const void *sendbuff, void *recvbuff, size_t recv_count, DataType dtype,
                             nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm,
                             Stream *stream) const {
    MCCL_CHECK(mcclReduceScatter(sendbuff, recvbuff, recv_count, kMcclDtypeMap.at(dtype),
                                 kMcclReduceOpMap.at(reduce_op), GetMcclComm(comm), GetMacaStream(stream)));
}

void McclImpl::Send(const void *buff, size_t count, DataType dtype, int peer, const CclComm *comm,
                    Stream *stream) const {
    MCCL_CHECK(mcclSend(buff, count, kMcclDtypeMap.at(dtype), peer, GetMcclComm(comm), GetMacaStream(stream)));
}

void McclImpl::Recv(void *buff, size_t count, DataType dtype, int peer, const CclComm *comm, Stream *stream) const {
    MCCL_CHECK(mcclRecv(buff, count, kMcclDtypeMap.at(dtype), peer, GetMcclComm(comm), GetMacaStream(stream)));
}

INFINI_TRAIN_REGISTER_CCL_IMPL(Device::DeviceType::kMACA, McclImpl)

} // namespace infini_train::core::maca

#pragma once

#include <string>
#include <unordered_map>

#include "infini_train/include/core/ccl/ccl.h"

namespace infini_train::core::cuda {

#ifdef USE_NCCL

class NcclImpl final : public CclImpl {
public:
    Device::DeviceType Type() const override;

    void GroupStart() const override;

    void GroupEnd() const override;

    void CommGetAsyncError(const CclComm *comm, CclStatus *async_error) const override;

    void CreateComm(CclComm **comm) const override;

    void CreateUniqueId(CclUniqueId **unique_id) const override;

    void GetUniqueId(CclUniqueId *unique_id) const override;

    void WriteUniqueId(const CclUniqueId &unique_id, const std::string &pg_name) const override;

    void ReadUniqueId(CclUniqueId *unique_id, const std::string &pg_name) const override;

    void CleanupUniqueIdFile(const std::string &pg_name) const override;

    void CommInitAll(CclComm **comms, int ndev, const int *devlist) const override;

    void CommInitRank(CclComm *comm, int nranks, const CclUniqueId &unique_id, int rank) const override;

    void CommDestroy(CclComm *comm) const override;

    void AllReduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                   nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm, Stream *stream) const override;

    void Broadcast(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, int root, const CclComm *comm,
                   Stream *stream) const override;

    void Reduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                nn::parallel::function::ReduceOpType reduce_op, int root, const CclComm *comm,
                Stream *stream) const override;

    void AllGather(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, const CclComm *comm,
                   Stream *stream) const override;

    void ReduceScatter(const void *sendbuff, void *recvbuff, size_t recv_count, DataType dtype,
                       nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm,
                       Stream *stream) const override;

    void Send(const void *buff, size_t count, DataType dtype, int peer, const CclComm *comm,
              Stream *stream) const override;

    void Recv(void *buff, size_t count, DataType dtype, int peer, const CclComm *comm, Stream *stream) const override;
};
#endif

} // namespace infini_train::core::cuda

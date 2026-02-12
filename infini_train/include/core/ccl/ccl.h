#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "infini_train/include/core/ccl/ccl_common.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train::core {

class Stream;

class CclImpl {
public:
    CclImpl() {}
    virtual ~CclImpl() = default;

    virtual Device::DeviceType Type() const = 0;

    virtual void GroupStart() const;

    virtual void GroupEnd() const;

    virtual void CommGetAsyncError(const CclComm *comm, CclStatus *async_error) const;

    virtual void CreateComm(CclComm **comm) const;

    virtual void CreateUniqueId(CclUniqueId **unique_id) const;

    virtual void GetUniqueId(CclUniqueId *unique_id) const;

    virtual void WriteUniqueId(const CclUniqueId &unique_id, const std::string &pg_name) const;

    virtual void ReadUniqueId(CclUniqueId *unique_id, const std::string &pg_name) const;

    virtual void CleanupUniqueIdFile(const std::string &pg_name) const;

    virtual void CommInitAll(CclComm **comms, int ndev, const int *devlist) const;

    virtual void CommInitRank(CclComm *comm, int nranks, const CclUniqueId &unique_id, int rank) const;

    virtual void CommDestroy(CclComm *comm) const;

    virtual void AllReduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                           nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm, Stream *stream) const;

    virtual void Broadcast(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, int root,
                           const CclComm *comm, Stream *stream) const;

    virtual void Reduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                        nn::parallel::function::ReduceOpType reduce_op, int root, const CclComm *comm,
                        Stream *stream) const;

    virtual void AllGather(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, const CclComm *comm,
                           Stream *stream) const;

    virtual void ReduceScatter(const void *sendbuff, void *recvbuff, size_t recv_count, DataType dtype,
                               nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm,
                               Stream *stream) const;

    virtual void Send(const void *buff, size_t count, DataType dtype, int peer, const CclComm *comm,
                      Stream *stream) const;

    virtual void Recv(void *buff, size_t count, DataType dtype, int peer, const CclComm *comm, Stream *stream) const;
};

class Ccl {
public:
    explicit Ccl(Device::DeviceType type);
    ~Ccl();

    Ccl(const Ccl &) = delete;
    Ccl &operator=(const Ccl &) = delete;
    Ccl(Ccl &&) = delete;
    Ccl &operator=(Ccl &&) = delete;

private:
    CclImpl *impl_ = nullptr;
};

class CclImplRegistry {
public:
    static CclImplRegistry &Instance();

    void Register(Device::DeviceType type, std::unique_ptr<CclImpl> impl);

    CclImpl *Get(Device::DeviceType type) const;

private:
    CclImplRegistry() = default;
    CclImplRegistry(const CclImplRegistry &) = delete;
    CclImplRegistry &operator=(const CclImplRegistry &) = delete;

    std::unordered_map<Device::DeviceType, std::unique_ptr<CclImpl>> impls_;
};

CclImpl *GetCclImpl(Device::DeviceType type);

} // namespace infini_train::core

#define INFINI_TRAIN_REGISTER_CCL_IMPL(device_type, class_impl)                                                        \
    static const bool __infini_train_ccl_registered##__COUNTER__ = []() {                                              \
        infini_train::core::CclImplRegistry::Instance().Register(device_type, std::make_unique<class_impl>());         \
        return true;                                                                                                   \
    }();

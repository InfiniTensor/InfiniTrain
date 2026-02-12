#include "infini_train/include/core/ccl/ccl.h"

#include <format>
#include <memory>
#include <utility>

#include "glog/logging.h"

namespace infini_train::core {

void CclImpl::GroupStart() const { LOG(FATAL) << "CclImpl::GroupStart is not implemented."; }

void CclImpl::GroupEnd() const { LOG(FATAL) << "CclImpl::GroupEnd is not implemented."; }

void CclImpl::CommGetAsyncError(const CclComm *comm, CclStatus *async_error) const {
    LOG(FATAL) << "CclImpl::CommGetAsyncError is not implemented.";
}

void CclImpl::CreateComm(CclComm **comm) const { LOG(FATAL) << "CclImpl::CreateComm is not implemented."; }

void CclImpl::CreateUniqueId(CclUniqueId **unique_id) const {
    LOG(FATAL) << "CclImpl::CreateUniqueId is not implemented.";
}

void CclImpl::GetUniqueId(CclUniqueId *unique_id) const { LOG(FATAL) << "CclImpl::GetUniqueId is not implemented."; }

void CclImpl::WriteUniqueId(const CclUniqueId &unique_id, const std::string &pg_name) const {
    LOG(FATAL) << "CclImpl::WriteUniqueId is not implemented.";
}

void CclImpl::ReadUniqueId(CclUniqueId *unique_id, const std::string &pg_name) const {
    LOG(FATAL) << "CclImpl::ReadUniqueId is not implemented.";
}

void CclImpl::CleanupUniqueIdFile(const std::string &pg_name) const {
    LOG(FATAL) << "CclImpl::CleanupUniqueIdFile is not implemented.";
}

void CclImpl::CommInitAll(CclComm **comms, int ndev, const int *devlist) const {
    LOG(FATAL) << "CclImpl::CommInitAll is not implemented.";
}

void CclImpl::CommInitRank(CclComm *comm, int nranks, const CclUniqueId &unique_id, int rank) const {
    LOG(FATAL) << "CclImpl::CommInitRank is not implemented.";
}

void CclImpl::CommDestroy(CclComm *comm) const { LOG(FATAL) << "CclImpl::CommDestroy is not implemented."; }

void CclImpl::AllReduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                        nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm, Stream *stream) const {
    LOG(FATAL) << "CclImpl::AllReduce is not implemented.";
}

void CclImpl::Broadcast(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, int root,
                        const CclComm *comm, Stream *stream) const {
    LOG(FATAL) << "CclImpl::Broadcast is not implemented.";
}

void CclImpl::Reduce(const void *sendbuff, void *recvbuff, size_t count, DataType dtype,
                     nn::parallel::function::ReduceOpType reduce_op, int root, const CclComm *comm,
                     Stream *stream) const {
    LOG(FATAL) << "CclImpl::Reduce is not implemented.";
}

void CclImpl::AllGather(const void *sendbuff, void *recvbuff, size_t count, DataType dtype, const CclComm *comm,
                        Stream *stream) const {
    LOG(FATAL) << "CclImpl::AllGather is not implemented.";
}

void CclImpl::ReduceScatter(const void *sendbuff, void *recvbuff, size_t recv_count, DataType dtype,
                            nn::parallel::function::ReduceOpType reduce_op, const CclComm *comm, Stream *stream) const {
    LOG(FATAL) << "CclImpl::ReduceScatter is not implemented.";
}

void CclImpl::Send(const void *buff, size_t count, DataType dtype, int peer, const CclComm *comm,
                   Stream *stream) const {
    LOG(FATAL) << "CclImpl::Send is not implemented.";
}

void CclImpl::Recv(void *buff, size_t count, DataType dtype, int peer, const CclComm *comm, Stream *stream) const {
    LOG(FATAL) << "CclImpl::Recv is not implemented.";
}

Ccl::Ccl(Device::DeviceType type) : impl_(GetCclImpl(type)) { impl_->GroupStart(); }

Ccl::~Ccl() { impl_->GroupEnd(); }

CclImplRegistry &CclImplRegistry::Instance() {
    static CclImplRegistry instance;
    return instance;
}

void CclImplRegistry::Register(Device::DeviceType type, std::unique_ptr<CclImpl> impl) {
    if (type != impl->Type()) {
        LOG(FATAL) << std::format("Register CclImpl with type {}, but as type {}", static_cast<int>(impl->Type()),
                                  static_cast<int>(type));
    }

    if (impls_.contains(type)) {
        LOG(FATAL) << std::format("CclImpl for type {} already registered", static_cast<int>(type));
    }

    impls_[type] = std::move(impl);
}

CclImpl *CclImplRegistry::Get(Device::DeviceType type) const {
    auto it = impls_.find(type);
    if (it == impls_.end()) {
        LOG(FATAL) << "No CclImpl registered for type " << static_cast<int>(type);
    }
    return it->second.get();
}

CclImpl *GetCclImpl(Device::DeviceType type) { return CclImplRegistry::Instance().Get(type); }

} // namespace infini_train::core

#include "infini_train/src/core/ccl/dcu/rccl_common.h"

#include <cstring>

#include "glog/logging.h"

namespace infini_train::core {

RcclComm::RcclComm() = default;

RcclComm::RcclComm(ncclComm_t comm) : nccl_comm_(comm) {}

ncclComm_t RcclComm::nccl_comm() const { return nccl_comm_; }

void RcclComm::set_nccl_comm(ncclComm_t comm) { nccl_comm_ = comm; }

RcclUniqueId::RcclUniqueId() = default;

RcclUniqueId::RcclUniqueId(const ncclUniqueId &id) : id_(id) {}

size_t RcclUniqueId::Size() const { return sizeof(id_); }

const void *RcclUniqueId::Data() const { return &id_; }

void RcclUniqueId::Load(const void *src, size_t size) {
    CHECK_NOTNULL(src);
    CHECK_EQ(size, sizeof(id_));
    std::memcpy(&id_, src, sizeof(id_));
}

ncclUniqueId *RcclUniqueId::nccl_unique_id() { return &id_; }

const ncclUniqueId *RcclUniqueId::nccl_unique_id() const { return &id_; }

} // namespace infini_train::core

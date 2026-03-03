#include "infini_train/src/core/ccl/cuda/nccl_common.h"

#include <cstring>

#include "glog/logging.h"

namespace infini_train::core {

NcclComm::NcclComm() = default;

NcclComm::NcclComm(ncclComm_t comm) : nccl_comm_(comm) {}

ncclComm_t NcclComm::nccl_comm() const { return nccl_comm_; }

void NcclComm::set_nccl_comm(ncclComm_t comm) { nccl_comm_ = comm; }

NcclUniqueId::NcclUniqueId() = default;

NcclUniqueId::NcclUniqueId(const ncclUniqueId &id) : id_(id) {}

size_t NcclUniqueId::Size() const { return sizeof(id_); }

const void *NcclUniqueId::Data() const { return &id_; }

void NcclUniqueId::Load(const void *src, size_t size) {
    CHECK_NOTNULL(src);
    CHECK_EQ(size, sizeof(id_));
    std::memcpy(&id_, src, sizeof(id_));
}

ncclUniqueId *NcclUniqueId::nccl_unique_id() { return &id_; }

const ncclUniqueId *NcclUniqueId::nccl_unique_id() const { return &id_; }

} // namespace infini_train::core

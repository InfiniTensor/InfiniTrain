#include "infini_train/src/core/ccl/maca/mccl_common.h"

#include <cstring>

#include "glog/logging.h"

namespace infini_train::core {

McclComm::McclComm() = default;

McclComm::McclComm(mcclComm_t comm) : mccl_comm_(comm) {}

mcclComm_t McclComm::mccl_comm() const { return mccl_comm_; }

void McclComm::set_mccl_comm(mcclComm_t comm) { mccl_comm_ = comm; }

McclUniqueId::McclUniqueId() = default;

McclUniqueId::McclUniqueId(const mcclUniqueId &id) : id_(id) {}

size_t McclUniqueId::Size() const { return sizeof(id_); }

const void *McclUniqueId::Data() const { return &id_; }

void McclUniqueId::Load(const void *src, size_t size) {
    CHECK_NOTNULL(src);
    CHECK_EQ(size, sizeof(id_));
    std::memcpy(&id_, src, sizeof(id_));
}

mcclUniqueId *McclUniqueId::mccl_unique_id() { return &id_; }

const mcclUniqueId *McclUniqueId::mccl_unique_id() const { return &id_; }

} // namespace infini_train::core

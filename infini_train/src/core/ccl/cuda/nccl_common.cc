#include "infini_train/src/core/ccl/cuda/nccl_common.h"

namespace infini_train::core {

#ifdef USE_NCCL
NcclComm::NcclComm() = default;

NcclComm::NcclComm(ncclComm_t comm) : comm_(comm) {}

ncclComm_t NcclComm::nccl_comm() const { return comm_; }

void NcclComm::set_nccl_comm(ncclComm_t comm) { comm_ = comm; }

NcclUniqueId::NcclUniqueId() = default;

NcclUniqueId::NcclUniqueId(const ncclUniqueId &id) : id_(id) {}

ncclUniqueId *NcclUniqueId::nccl_unique_id() { return &id_; }

const ncclUniqueId *NcclUniqueId::nccl_unique_id() const { return &id_; }
#endif

} // namespace infini_train::core

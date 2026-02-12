#pragma once

#ifdef USE_NCCL
#include <nccl.h>

#include "infini_train/include/core/ccl/ccl_common.h"

namespace infini_train::core {

class NcclComm final : public CclComm {
public:
    NcclComm();
    explicit NcclComm(ncclComm_t comm);

    ncclComm_t nccl_comm() const;
    void set_nccl_comm(ncclComm_t comm);

private:
    ncclComm_t comm_ = nullptr;
};

class NcclUniqueId final : public CclUniqueId {
public:
    NcclUniqueId();
    explicit NcclUniqueId(const ncclUniqueId &id);

    ncclUniqueId *nccl_unique_id();
    const ncclUniqueId *nccl_unique_id() const;

private:
    ncclUniqueId id_{};
};
#endif

} // namespace infini_train::core

#pragma once

#include "infini_train/include/common/dcu/rccl_compat.h"

#include "infini_train/include/core/ccl/ccl_common.h"

namespace infini_train::core {

class RcclComm final : public CclComm {
public:
    RcclComm();
    explicit RcclComm(ncclComm_t comm);

    ncclComm_t nccl_comm() const;
    void set_nccl_comm(ncclComm_t comm);

private:
    ncclComm_t nccl_comm_ = nullptr;
};

class RcclUniqueId final : public CclUniqueId {
public:
    RcclUniqueId();
    explicit RcclUniqueId(const ncclUniqueId &id);

    size_t Size() const override;
    const void *Data() const override;
    void Load(const void *src, size_t size) override;

    ncclUniqueId *nccl_unique_id();
    const ncclUniqueId *nccl_unique_id() const;

private:
    ncclUniqueId id_;
};

} // namespace infini_train::core

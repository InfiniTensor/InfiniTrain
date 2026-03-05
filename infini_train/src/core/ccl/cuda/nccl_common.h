#pragma once

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
    ncclComm_t nccl_comm_ = nullptr;
};

class NcclUniqueId final : public CclUniqueId {
public:
    NcclUniqueId();
    explicit NcclUniqueId(const ncclUniqueId &id);

    size_t Size() const override;
    const void *Data() const override;
    void Load(const void *src, size_t size) override;

    ncclUniqueId *nccl_unique_id();
    const ncclUniqueId *nccl_unique_id() const;

private:
    ncclUniqueId id_;
};

} // namespace infini_train::core

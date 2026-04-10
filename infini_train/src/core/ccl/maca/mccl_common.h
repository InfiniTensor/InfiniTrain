#pragma once

#include <mccl.h>

#include "infini_train/include/core/ccl/ccl_common.h"

namespace infini_train::core {

class McclComm final : public CclComm {
public:
    McclComm();
    explicit McclComm(mcclComm_t comm);

    mcclComm_t mccl_comm() const;
    void set_mccl_comm(mcclComm_t comm);

private:
    mcclComm_t mccl_comm_ = nullptr;
};

class McclUniqueId final : public CclUniqueId {
public:
    McclUniqueId();
    explicit McclUniqueId(const mcclUniqueId &id);

    size_t Size() const override;
    const void *Data() const override;
    void Load(const void *src, size_t size) override;

    mcclUniqueId *mccl_unique_id();
    const mcclUniqueId *mccl_unique_id() const;

private:
    mcclUniqueId id_;
};

} // namespace infini_train::core

#pragma once

#include "infini_train/include/core/device_guard.h"

namespace infini_train::core::cpu {

class CpuGuardImpl : public DeviceGuardImpl {
public:
    CpuGuardImpl();

    Device GetDevice() const;

    Device::DeviceType Type() const;

    void Malloc(void **dev_ptr, size_t size);

    void Free(void *dev_ptr);

    void Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind);
};

} // namespace infini_train::core::cpu

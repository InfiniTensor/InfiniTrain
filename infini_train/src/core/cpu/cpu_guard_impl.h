#pragma once

#include <utility>

#include "infini_train/include/core/device_guard.h"

namespace infini_train::core::cpu {

class CpuGuardImpl final : public DeviceGuardImpl {
public:
    CpuGuardImpl();

    // Device management
    Device GetDevice() const override;

    void SetDevice(Device device) const override; // CPU: no-op

    int DeviceCount() const override; // CPU: 1

    Device::DeviceType Type() const override;

    // Stream management (explicitly unsupported for now)
    Stream *GetStream(Device device) const override;

    // Synchronization
    void SynchronizeDevice(Device device) const override; // CPU: no-op

    void SynchronizeStream(Stream *stream) const override; // CPU: no-op

    // BLAS handle (explicitly unsupported for now)
    BlasHandle *GetBlasHandle(Device device) const override;

    // Memory ops (async ops falls back to blocking explicitly in CPU impl)
    void Malloc(void **dev_ptr, size_t size) override;

    void Free(void *dev_ptr) override;

    void MallocAsync(void **dev_ptr, size_t size, Stream *stream) override;

    void FreeAsync(void *dev_ptr, Stream *stream) override;

    void Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) override;

    void MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) override;

    void ResetMemPoolHighWatermarks(Device device) const override; // CPU: no-op

    std::pair<size_t, size_t> GetMemPoolPeakMB(Device device) const override; // CPU: {0, 0}
};

} // namespace infini_train::core::cpu

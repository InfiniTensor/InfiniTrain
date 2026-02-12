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

    Stream *CreateStream(Device device) const override;

    Stream *CreateStreamWithPriority(Device device, int priority) const override;

    void DestroyStream(Stream *stream) const override;

    // Event management (explicitly unsupported for now)
    void EventCreate(Event **event) const override;

    void EventCreateWithFlags(Event **event, uint32_t flags) const override;

    void EventDestroy(Event *event) const override;

    void EventRecord(Event *event, Stream *stream) const override;

    void StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const override;

    RuntimeStatus EventSynchronize(Event *event) const override;

    RuntimeStatus EventQuery(Event *event) const override;

    float EventElapsedTime(Event *start_event, Event *stop_event) const override;

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

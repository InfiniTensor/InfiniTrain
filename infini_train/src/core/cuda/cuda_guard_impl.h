#pragma once

#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/device.h"

namespace infini_train::core {
class Stream;
class BlasHandle;
} // namespace infini_train::core

namespace infini_train::core::cuda {

class CudaGuardImpl final : public DeviceGuardImpl {
public:
    static void InitSingleStream(Device device);

    static void InitSingleHandle(Device device);

    CudaGuardImpl();

    // device
    Device GetDevice() const override;

    void SetDevice(Device device) const override;

    int DeviceCount() const override;

    Device::DeviceType Type() const override;

    // stream
    Stream *GetStream(Device device) const override;

    Stream *CreateStream(Device device) const override;

    Stream *CreateStreamWithPriority(Device device, int priority) const override;

    void DestroyStream(Stream *stream) const override;

    // event
    void EventCreate(Event **event) const override;

    void EventCreateWithFlags(Event **event, uint32_t flags) const override;

    void EventDestroy(Event *event) const override;

    void EventRecord(Event *event, Stream *stream) const override;

    void StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const override;

    RuntimeStatus EventSynchronize(Event *event) const override;

    RuntimeStatus EventQuery(Event *event) const override;

    float EventElapsedTime(Event *start_event, Event *stop_event) const override;

    // sync
    void SynchronizeDevice(Device device) const override;

    // blas
    BlasHandle *GetBlasHandle(Device device) const override;

    // memory
    void Malloc(void **dev_ptr, size_t size) override;

    void MallocAsync(void **dev_ptr, size_t size, Stream *stream) override;

    void Free(void *dev_ptr) override;

    void FreeAsync(void *dev_ptr, Stream *stream) override;

    void Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) override;

    void MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) override;

    void ResetMemPoolHighWatermarks(Device device) const override;

    std::pair<size_t, size_t> GetMemPoolPeakMB(Device device) const override;
};

} // namespace infini_train::core::cuda

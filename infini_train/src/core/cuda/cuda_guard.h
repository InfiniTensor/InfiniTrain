#pragma once

#include <cstdint>

#include "infini_train/include/core/blas_handle.h"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/core/stream.h"
#include "infini_train/include/device.h"

namespace infini_train::core::cuda {

class CudaGuardImpl : public DeviceGuardImpl {
public:
    static void InitSingleStream(Device device);

    static void InitSingleHandle(Device device);

    CudaGuardImpl();

    // device
    Device GetDevice() const override;

    void SetDevice(Device device) const override;

    int8_t DeviceCount() const override;

    Device::DeviceType Type() const override;

    // stream
    Stream *GetStream(Device device) const override;

    // event

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
};

} // namespace infini_train::core::cuda

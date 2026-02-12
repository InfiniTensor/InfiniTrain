#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "infini_train/include/core/event.h"
#include "infini_train/include/core/runtime_status.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

class Stream;
class BlasHandle;

// Note(dcj): In the CPU backend, kD2D corresponds to a regular memcpy.
enum class MemcpyKind : int8_t {
    kH2D = 0,
    kD2H = 1,
    kD2D = 2,
    kInvalid = -1,
};

inline const char *MemcpyKindToString(MemcpyKind k) {
    switch (k) {
    case MemcpyKind::kH2D:
        return "kH2D";
    case MemcpyKind::kD2H:
        return "kD2H";
    case MemcpyKind::kD2D:
        return "kD2D";
    case MemcpyKind::kInvalid:
        return "kInvalid";
    default:
        return "Unknown";
    }
}

//
// ----------------------------------------------------------------------------
// DeviceGuardImpl: Backend-specific device/stream/memory/BLAS implementation
// ----------------------------------------------------------------------------
// This is the low-level virtual interface that each backend must implement.
// Examples:
//   - CUDA:   CudaDeviceGuardImpl
//   - CPU:    CpuDeviceGuardImpl
//   - Custom: MyChipDeviceGuardImpl
//
// DeviceGuardImpl encapsulates **all device-runtime behaviors**, including:
//
//   • Querying / setting the current device
//   • Stream creation/lookup
//   • Synchronization primitives
//   • Memory allocation & copy
//   • Access to BLAS handles
//
// DeviceGuard (the public RAII wrapper) forwards calls to the DeviceGuardImpl
// instance registered for the device type.
//
class DeviceGuardImpl {
public:
    DeviceGuardImpl() {}

    virtual ~DeviceGuardImpl() = default;

    // ----------------------------------------------------------------------
    // Device management
    // ----------------------------------------------------------------------

    virtual Device GetDevice() const = 0;

    virtual void SetDevice(Device device) const;

    virtual int DeviceCount() const;

    virtual Device::DeviceType Type() const = 0;

    // ----------------------------------------------------------------------
    // Stream management
    // ----------------------------------------------------------------------

    virtual Stream *GetStream(Device) const;

    virtual Stream *CreateStream(Device) const;

    virtual Stream *CreateStreamWithPriority(Device, int priority) const;

    virtual void DestroyStream(Stream *) const;

    // ----------------------------------------------------------------------
    // Event management
    // ----------------------------------------------------------------------

    virtual void EventCreate(Event **event) const;

    virtual void EventCreateWithFlags(Event **event, uint32_t flags) const;

    virtual void EventDestroy(Event *event) const;

    virtual void EventRecord(Event *event, Stream *stream) const;

    virtual void StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const;

    virtual RuntimeStatus EventSynchronize(Event *event) const;

    virtual RuntimeStatus EventQuery(Event *event) const;

    virtual float EventElapsedTime(Event *start_event, Event *stop_event) const;

    // ----------------------------------------------------------------------
    // Synchronization
    // ----------------------------------------------------------------------

    virtual void SynchronizeDevice(Device) const;

    virtual void SynchronizeStream(Stream *) const;

    // ----------------------------------------------------------------------
    // BLAS handle
    // ----------------------------------------------------------------------

    virtual BlasHandle *GetBlasHandle(Device) const;

    // ----------------------------------------------------------------------
    // Memory operations
    // ----------------------------------------------------------------------

    virtual void Malloc(void **dev_ptr, size_t size) = 0;

    virtual void MallocAsync(void **dev_ptr, size_t size, Stream *stream);

    virtual void Free(void *dev_ptr) = 0;

    virtual void FreeAsync(void *dev_ptr, Stream *stream);

    virtual void Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) = 0;

    virtual void MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream);

    virtual void ResetMemPoolHighWatermarks(Device device) const;

    virtual std::pair<size_t, size_t> GetMemPoolPeakMB(Device device) const;
};

//
// ----------------------------------------------------------------------------
// DeviceGuard: RAII front-end wrapper for DeviceGuardImpl
// ----------------------------------------------------------------------------
// This class is the **public-facing device interface** for the framework.
// It automatically:
//
//   • Saves the current device on construction
//   • Switches to the target device
//   • Restores the previous device on destruction
//
// All runtime operations are forwarded to the backend-specific DeviceGuardImpl
// instance registered for that device type.
//
class DeviceGuard {
public:
    explicit DeviceGuard(Device device);

    ~DeviceGuard();

    // Copy is disallowed
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;

    // Move is disallowed, as DeviceGuard does not have an uninitialized state,
    // which is required for moves on types with nontrival destructors.
    DeviceGuard(DeviceGuard &&other) = delete;
    DeviceGuard &operator=(DeviceGuard &&other) = delete;

    void SetDevice(Device device);

    Device current_device() const;

    Device original_device() const;

private:
    DeviceGuardImpl *impl_ = nullptr;
    Device original_device_;
    Device current_device_;
};

//
// ----------------------------------------------------------------------------
// DeviceGuardImplRegistry: Global registry of backend implementations
// ----------------------------------------------------------------------------
// This registry stores at most one DeviceGuardImpl per DeviceType.
// Backends register themselves at static initialization time via the macro
// INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL().
//
// Example:
//   INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kCUDA, CudaGuardImpl)
//
class DeviceGuardImplRegistry {
public:
    static DeviceGuardImplRegistry &Instance();

    void Register(Device::DeviceType type, std::unique_ptr<DeviceGuardImpl> impl);

    DeviceGuardImpl *Get(Device::DeviceType type) const;

private:
    DeviceGuardImplRegistry() = default;
    DeviceGuardImplRegistry(const DeviceGuardImplRegistry &) = delete;
    DeviceGuardImplRegistry &operator=(const DeviceGuardImplRegistry &) = delete;

    std::unordered_map<Device::DeviceType, std::unique_ptr<DeviceGuardImpl>> impls_;
};

DeviceGuardImpl *GetDeviceGuardImpl(Device::DeviceType type);

} // namespace infini_train::core

//
// ----------------------------------------------------------------------------
// Registration macro
// ----------------------------------------------------------------------------
// Registers a DeviceGuardImpl implementation into the global registry
// at static initialization time.
//
// Example usage:
//   INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kCUDA, CudaGuardImpl)
//
#define INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(device_type, class_impl)                                               \
    static const bool __infini_train_device_guard_registered##__COUNTER__ = []() {                                     \
        infini_train::core::DeviceGuardImplRegistry::Instance().Register(device_type, std::make_unique<class_impl>()); \
        return true;                                                                                                   \
    }();

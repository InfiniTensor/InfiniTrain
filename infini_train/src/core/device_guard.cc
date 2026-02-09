#include "infini_train/include/core/device_guard.h"

#include <format>
#include <memory>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/core/blas_handle.h"
#include "infini_train/include/core/stream.h"

namespace infini_train::core {

// DeviceGuardImpl
void DeviceGuardImpl::SetDevice(Device device) const {
    LOG(WARNING) << std::format("SetDevice is not supported for device type {} (index {}). "
                                "The call is ignored.",
                                static_cast<int>(device.type()), device.index());
}

int8_t DeviceGuardImpl::DeviceCount() const { return -1; }

Stream *DeviceGuardImpl::GetStream(Device) const { return nullptr; }

void DeviceGuardImpl::SynchronizeDevice(Device device) const {
    LOG(WARNING) << std::format("SynchronizeDevice is not supported for this device. "
                                "The call is ignored.",
                                static_cast<int>(device.type()), device.index());
}

void DeviceGuardImpl::SynchronizeStream(Stream *) const {
    LOG(WARNING) << "SynchronizeStream is not supported for this device. "
                    "The call is ignored.";
}

BlasHandle *DeviceGuardImpl::GetBlasHandle(Device device) const {
    LOG(FATAL) << std::format("GetBlasHandle is not supported for device type {} (index {}). ",
                              static_cast<int>(device.type()), device.index());
}

void DeviceGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    LOG(WARNING) << "MallocAsync is not supported on this device. Falling back to blocking Malloc()";
    Malloc(dev_ptr, size);
}

void DeviceGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    LOG(WARNING) << "FreeAsync is not supported on this device. Falling back to blocking Free()";
    Free(dev_ptr);
}

void DeviceGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    LOG(WARNING) << "MemcpyAsync is not supported on this device. Falling back to blocking Memcpy()";
    Memcpy(dst, src, count, kind);
}

void DeviceGuardImpl::ResetMemPoolHighWatermarks(Device device) const {
    LOG(WARNING) << std::format("ResetMemPoolHighWatermarks is not supported for device type {} (index {}). "
                                "The call is ignored.",
                                static_cast<int>(device.type()), device.index());
}

std::pair<size_t, size_t> DeviceGuardImpl::GetMemPoolPeakMB(Device device) const {
    LOG(WARNING) << std::format("GetMemPoolPeakMB is not supported for device type {} (index {}). "
                                "Returning {{0, 0}}.",
                                static_cast<int>(device.type()), device.index());
    return {0, 0};
}

// DeviceGuard
DeviceGuard::DeviceGuard(Device device) : impl_(GetDeviceGuardImpl(device.type())) {
    original_device_ = impl_->GetDevice();
    impl_->SetDevice(device);
    current_device_ = device;
}

void DeviceGuard::SetDevice(Device device) {
    if (current_device_ == device) {
        return;
    }
    impl_->SetDevice(device);
    current_device_ = device;
}

Device DeviceGuard::current_device() const { return current_device_; }

Device DeviceGuard::original_device() const { return original_device_; }

DeviceGuard::~DeviceGuard() { impl_->SetDevice(original_device_); }

// DeviceGuardImplRegistry
DeviceGuardImplRegistry &DeviceGuardImplRegistry::Instance() {
    static DeviceGuardImplRegistry instance;
    return instance;
}

void DeviceGuardImplRegistry::Register(Device::DeviceType type, std::unique_ptr<DeviceGuardImpl> impl) {
    if (type != impl->Type()) {
        LOG(FATAL) << std::format("Register device guard impl with type {}, but as type {}",
                                  static_cast<int>(impl->Type()), static_cast<int>(type));
    }

    if (impls_.contains(type)) {
        LOG(FATAL) << std::format("DeviceGuardImpl for type {} already registrered", static_cast<int>(type));
    }

    if (!impls_.empty()) {
        for (auto &kv : impls_) {
            if (kv.first != Device::DeviceType::kCPU) {
                LOG(FATAL) << std::format("Only CPU and one GPU backend allowed. Already have GPU={}, new={} rejected.",
                                          static_cast<int>(kv.first), static_cast<int>(type));
            }
        }
    }

    impls_[type] = std::move(impl);
}

DeviceGuardImpl *DeviceGuardImplRegistry::Get(Device::DeviceType type) const {
    auto it = impls_.find(type);
    if (it == impls_.end()) {
        LOG(FATAL) << "No DeviceGuardImpl registered for type " << static_cast<int>(type);
    }
    return it->second.get();
}

DeviceGuardImpl *GetDeviceGuardImpl(Device::DeviceType type) { return DeviceGuardImplRegistry::Instance().Get(type); }

} // namespace infini_train::core

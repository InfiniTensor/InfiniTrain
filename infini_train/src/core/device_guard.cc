#include "infini_train/include/core/device_guard.h"

#include <format>
#include <memory>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/core/blas_handle.h"
#include "infini_train/include/core/stream.h"

namespace infini_train::core {

// DeviceGuardImpl (base fallback: FATAL only)
void DeviceGuardImpl::SetDevice(Device) const { LOG(FATAL) << "DeviceGuardImpl::SetDevice is not implemented."; }

int DeviceGuardImpl::DeviceCount() const {
    LOG(FATAL) << "DeviceGuardImpl::DeviceCount is not implemented.";
    return -1; // unreachable
}

Stream *DeviceGuardImpl::GetStream(Device) const {
    LOG(FATAL) << "DeviceGuardImpl::GetStream is not implemented.";
    return nullptr; // unreachable
}

void DeviceGuardImpl::SynchronizeDevice(Device) const {
    LOG(FATAL) << "DeviceGuardImpl::SynchronizeDevice is not implemented.";
}

void DeviceGuardImpl::SynchronizeStream(Stream *) const {
    LOG(FATAL) << "DeviceGuardImpl::SynchronizeStream is not implemented.";
}

BlasHandle *DeviceGuardImpl::GetBlasHandle(Device device) const {
    LOG(FATAL) << "DeviceGuardImpl::GetBlasHandle is not implemented.";
    return nullptr; // unreachable
}

void DeviceGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    LOG(FATAL) << "DeviceGuardImpl::MallocAsync is not implemented.";
}

void DeviceGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    LOG(FATAL) << "DeviceGuardImpl::FreeAsync is not implemented";
}

void DeviceGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    LOG(FATAL) << "DeviceGuardImpl::MemcpyAsync is not implemented";
}

void DeviceGuardImpl::ResetMemPoolHighWatermarks(Device device) const {
    LOG(FATAL) << "DeviceGuardImpl::ResetMemPoolHighWatermarks is not implemented.";
}

std::pair<size_t, size_t> DeviceGuardImpl::GetMemPoolPeakMB(Device device) const {
    LOG(FATAL) << "DeviceGuardImpl::GetMemPoolPeakMB is not implemented for device type {} (index {}).";
    return {0, 0}; // unreachable
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

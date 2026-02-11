#include "infini_train/src/core/cpu/cpu_guard_impl.h"

#include <cstdlib>
#include <cstring>
#include <format>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/core/device_guard.h"

namespace infini_train::core::cpu {

CpuGuardImpl::CpuGuardImpl() {}

Device CpuGuardImpl::GetDevice() const { return Device(Device::DeviceType::kCPU, 0); }

Device::DeviceType CpuGuardImpl::Type() const { return Device::DeviceType::kCPU; }

void CpuGuardImpl::SetDevice(Device device) const {
    // No-op for CPU
    CHECK(device.type() == Device::DeviceType::kCPU);
    LOG(WARNING) << "CpuGuardImpl::SetDevice is not supported. "
                    "The call is ignored.";
}

int CpuGuardImpl::DeviceCount() const { return 1; }

Stream *CpuGuardImpl::GetStream(Device device) const {
    CHECK(device.type() == Device::DeviceType::kCPU);
    LOG(WARNING) << "CpuGuardImpl::GetStream is not supported. "
                    "Return nullptr.";
    return nullptr;
}

void CpuGuardImpl::SynchronizeDevice(Device device) const {
    // No-op for CPU
    CHECK(device.type() == Device::DeviceType::kCPU);
    LOG(WARNING) << "CpuGuardImpl::SynchronizeDevice is not supported. "
                    "The call is ignored.";
}

void CpuGuardImpl::SynchronizeStream(Stream *) const {
    // No-op for CPU
    LOG(WARNING) << "CpuGuardImpl::SynchronizeStream is not supported. "
                    "The call is ignored.";
}

BlasHandle *CpuGuardImpl::GetBlasHandle(Device device) const {
    CHECK(device.type() == Device::DeviceType::kCPU);
    LOG(WARNING) << "CpuGuardImpl::GetBlasHandle is not supported. "
                    "Return nullptr.";
    return nullptr;
}

void CpuGuardImpl::Malloc(void **dev_ptr, size_t size) { *dev_ptr = std::malloc(size); }

void CpuGuardImpl::Free(void *dev_ptr) { std::free(dev_ptr); }

void CpuGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    LOG(WARNING) << "CpuGuardImpl::MallocAsync is not supported. Falling back to blocking Malloc()";
    Malloc(dev_ptr, size);
}

void CpuGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    LOG(WARNING) << "CpuGuardImpl::FreeAsync is not supported. Falling back to blocking Free()";
    Free(dev_ptr);
}

void CpuGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    CHECK(kind == MemcpyKind::kD2D) << std::format("CpuGuardImpl::Memcpy only supports kD2D (host-to-host) memcpy, "
                                                   "but got MemcpyKind={}",
                                                   MemcpyKindToString(kind));
    std::memcpy(dst, src, count);
}

void CpuGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    LOG(WARNING) << "CpuGuardImpl::MemcpyAsync is not supported. Falling back to blocking Memcpy()";
    Memcpy(dst, src, count, kind);
}

void CpuGuardImpl::ResetMemPoolHighWatermarks(Device device) const {
    // No-op for CPU
    CHECK(device.type() == Device::DeviceType::kCPU);
    LOG(WARNING) << "CpuGuardImpl::ResetMemPoolHighWatermarks is not supported. "
                    "The call is ignored.";
}

std::pair<size_t, size_t> CpuGuardImpl::GetMemPoolPeakMB(Device device) const {
    CHECK(device.type() == Device::DeviceType::kCPU);
    LOG(WARNING) << "CpuGuardImpl::GetMemPoolPeakMB is not supported. "
                    "Return {0, 0}.";
    return {0, 0};
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kCPU, CpuGuardImpl)

} // namespace infini_train::core::cpu

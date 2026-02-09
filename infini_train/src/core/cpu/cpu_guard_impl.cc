#include "infini_train/src/core/cpu/cpu_guard_impl.h"

#include <cstdlib>
#include <cstring>

#include "glog/logging.h"

#include "infini_train/include/core/device_guard.h"

namespace infini_train::core::cpu {

CpuGuardImpl::CpuGuardImpl() {}

Device CpuGuardImpl::GetDevice() const { return Device(Device::DeviceType::kCPU, 0); }

Device::DeviceType CpuGuardImpl::Type() const { return Device::DeviceType::kCPU; }

void CpuGuardImpl::Malloc(void **dev_ptr, size_t size) { *dev_ptr = std::malloc(size); }

void CpuGuardImpl::Free(void *dev_ptr) { std::free(dev_ptr); }

void CpuGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    CHECK(kind == MemcpyKind::kD2D) << "CpuGuardImpl::Memcpy only supports kD2D (host-to-host) memcpy, "
                                    << "but got MemcpyKind=" << static_cast<int>(kind);

    std::memcpy(dst, src, count);
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kCPU, CpuGuardImpl)

} // namespace infini_train::core::cpu

#include "infini_train/src/core/cpu/cpu_guard.h"

#include <cstdlib>
#include <cstring>

namespace infini_train::core::cpu {

Device CpuGuardImpl::GetDevice() const { return Device(Device::DeviceType::kCPU, 0); }

Device::DeviceType CpuGuardImpl::Type() const { return Device::DeviceType::kCPU; }

void CpuGuardImpl::Malloc(void **dev_ptr, size_t size) { *dev_ptr = std::malloc(size); }

void CpuGuardImpl::Free(void *dev_ptr) { std::free(dev_ptr); }

void CpuGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) { std::memcpy(dst, src, count); }

} // namespace infini_train::core::cpu

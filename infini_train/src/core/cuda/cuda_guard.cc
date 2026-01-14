#include "infini_train/src/core/cuda/cuda_guard.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/blas_handle.h"
#include "infini_train/include/device.h"

#include "infini_train/src/core/cuda/cuda_blas_handle.h"
#include "infini_train/src/core/cuda/cuda_stream.h"

namespace infini_train::core::cuda {
namespace {
constexpr int kMaxGpus = 8;

static std::array<std::unique_ptr<CudaStream>, kMaxGpus> cuda_streams;
static std::array<std::unique_ptr<CudaBlasHandle>, kMaxGpus> cuda_blas_handles;

static std::array<std::once_flag, kMaxGpus> device_stream_flags;
static std::array<std::once_flag, kMaxGpus> device_handle_flags;
} // namespace

void CudaGuardImpl::InitSingleStream(Device device) {
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaSetDevice(device.index()));

    cuda_streams[device.index()] = std::make_unique<CudaStream>();

    CUDA_CHECK(cudaSetDevice(current_device));
}

void CudaGuardImpl::InitSingleHandle(Device device) {
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaSetDevice(device.index()));

    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device.index());

    cuda_blas_handles[device.index()] = std::make_unique<CudaBlasHandle>(cuda_streams[device.index()].get());

    CUDA_CHECK(cudaSetDevice(current_device));
}

CudaGuardImpl::CudaGuardImpl() {}

// device
Device CudaGuardImpl::GetDevice() const {
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    return Device(Device::DeviceType::kCUDA, current_device);
}

void CudaGuardImpl::SetDevice(Device device) const { CUDA_CHECK(cudaSetDevice(device.index())); }

int8_t CudaGuardImpl::DeviceCount() const {
    int device_count = 0;
    CUDA_DRIVER_CHECK(cuDeviceGetCount(&device_count));
    return device_count;
}

Device::DeviceType CudaGuardImpl::Type() const { return Device::DeviceType::kCUDA; }

// stream
Stream *CudaGuardImpl::GetStream(Device device) const {
    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);
    return cuda_streams.at(device.index()).get();
}

// event

// sync
void CudaGuardImpl::SynchronizeDevice(Device device) const {
    auto original_device = GetDevice();
    SetDevice(device);

    CUDA_CHECK(cudaDeviceSynchronize());

    SetDevice(original_device);
}

// blas
BlasHandle *CudaGuardImpl::GetBlasHandle(Device device) const {
    std::call_once(device_handle_flags.at(device.index()), InitSingleStream, device);
    return cuda_blas_handles.at(device.index()).get();
}

// memory
void CudaGuardImpl::Malloc(void **dev_ptr, size_t size) { CUDA_CHECK(cudaMalloc(dev_ptr, size)); }

void CudaGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    CUDA_CHECK(cudaMallocAsync(dev_ptr, size, dynamic_cast<CudaStream *>(stream)->cuda_stream()));
}

void CudaGuardImpl::Free(void *dev_ptr) { CUDA_CHECK(cudaFree(dev_ptr)); }

void CudaGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    CUDA_CHECK(cudaFreeAsync(dev_ptr, dynamic_cast<CudaStream *>(stream)->cuda_stream()));
}

void CudaGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    if (kind == MemcpyKind::kH2D) {
        CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    } else if (kind == MemcpyKind::kD2H) {
        CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    } else if (kind == MemcpyKind::kD2D) {
        CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
    } else {
        LOG(FATAL) << "Invalid MemcpyKind";
    }
}

void CudaGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    cudaStream_t cuda_stream = dynamic_cast<CudaStream *>(stream)->cuda_stream();
    if (kind == MemcpyKind::kH2D) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, cuda_stream));
    } else if (kind == MemcpyKind::kD2H) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, cuda_stream));
    } else if (kind == MemcpyKind::kD2D) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, cuda_stream));
    } else {
        LOG(FATAL) << "Invalid MemcpyKind";
    }
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kCUDA, CudaGuardImpl)

} // namespace infini_train::core::cuda

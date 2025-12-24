#include "infini_train/include/device.h"

#include <cstdint>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/global.h"
#ifdef USE_CUDA
#include "infini_train/include/common/cuda/common_cuda.h"
#endif

namespace infini_train {
namespace {
constexpr size_t kBytesPerMB = 1024ULL * 1024ULL;
} // namespace

Device::Device(DeviceType type, int8_t index) : type_(type), index_(index) {
    if (type_ == DeviceType::kCPU && index_ != 0) {
        LOG(FATAL) << "CPU device index should be 0";
    }
}

DeviceType Device::Type() const { return type_; }
int8_t Device::Index() const { return index_; }

bool Device::IsCPU() const { return type_ == DeviceType::kCPU; }
bool Device::IsCUDA() const { return type_ == DeviceType::kCUDA; }

std::string Device::ToString() const {
    std::ostringstream oss;
    oss << "Device(" << (type_ == DeviceType::kCPU ? "CPU" : "CUDA") << ", " << static_cast<int>(index_) << ")";
    return oss.str();
}

nn::parallel::Rank Device::rank() const {
    LOG(FATAL) << "Unimplemented";
    // prevent the compiler warning about control reaching the end of non-void function
    std::abort();
}

std::ostream &operator<<(std::ostream &os, const Device &device) {
    os << device.ToString();
    return os;
}

CpuDevice::CpuDevice() : Device(DeviceType::kCPU, 0) {}

#ifdef USE_CUDA
CudaDevice::~CudaDevice() {
    if (stream_ != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    if (cublas_handle_ != nullptr) {
        CUBLAS_CHECK(cublasDestroy(cublas_handle_));
    }
}

void CudaDevice::SetDevice() const { CUDA_CHECK(cudaSetDevice(index_)); }
void CudaDevice::Synchronize() const { CUDA_CHECK(cudaDeviceSynchronize()); }

cudaStream_t CudaDevice::Stream() const { return stream_; }

cublasHandle_t CudaDevice::CublasHandle() const { return cublas_handle_; }

nn::parallel::Rank CudaDevice::rank() const { return rank_; }

CudaDevice::CudaDevice(int8_t index)
    : Device(DeviceType::kCUDA, index),
      rank_({nn::parallel::global::GetGlobalProcRank(), index, nn::parallel::global::GetNprocPerNode(),
             nn::parallel::global::GetNthreadPerProc()}) {
    // TODO(dcj): make CudaDevice initialization lazy to avoid allocating memory on all GPUs in single-GPU mode
    SetDevice();
    CUDA_CHECK(cudaStreamCreate(&stream_));

    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
}

void CudaDevice::ResetMemPoolHighWatermarks() const {
    SetDevice();
    cudaMemPool_t pool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, index_));

    cuuint64_t zero = 0;
    // High watermark can only be reset to zero; non-zero is illegal.
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &zero));
}

std::pair<size_t, size_t> CudaDevice::GetMemPoolPeakMB() const {
    SetDevice();
    cudaMemPool_t pool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, index_));

    cuuint64_t used = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used));

    cuuint64_t reserved = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reserved));

    return std::make_pair<size_t, size_t>(static_cast<size_t>(used / kBytesPerMB),
                                          static_cast<size_t>(reserved / kBytesPerMB));
}
#endif // USE_CUDA

const DeviceManager *DeviceManager::Instance() {
    static auto instance = std::unique_ptr<DeviceManager>(new DeviceManager());
    return instance.get();
}

const Device *DeviceManager::GetDevice(DeviceType type, int8_t index) const {
    return devices_map_.at(type).at(index).get();
}

const Device *DeviceManager::GetDefaultDevice() const { return devices_map_.at(DeviceType::kCPU).at(0).get(); }

std::vector<const Device *> DeviceManager::GetAllAvailableDevices(DeviceType device_type) const {
    std::vector<const Device *> devices;
    for (const auto &device : devices_map_.at(device_type)) { devices.push_back(device.get()); }
    return devices;
}

DeviceManager::DeviceManager() {
    devices_map_[DeviceType::kCPU].push_back(std::unique_ptr<CpuDevice>(new CpuDevice()));
#ifdef USE_CUDA
    CUDA_DRIVER_CHECK(cuInit(0));
    int device_count = 0;
    CUDA_DRIVER_CHECK(cuDeviceGetCount(&device_count));
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    for (int idx = 0; idx < device_count; ++idx) {
        devices_map_[DeviceType::kCUDA].push_back(std::unique_ptr<CudaDevice>(new CudaDevice(idx)));
    }
    CUDA_CHECK(cudaSetDevice(current_device));
#endif
}

} // namespace infini_train

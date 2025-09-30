#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#ifdef USE_CUDA
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#endif
#ifdef USE_NCCL
#include "nccl.h"
#endif

namespace infini_train {

enum class DeviceType : int8_t {
    kCPU = 0,
    kCUDA = 1,
};

class DeviceManager;

class Device {
public:
    DeviceType Type() const;
    int8_t Index() const;

    bool IsCPU() const;
    bool IsCUDA() const;

    virtual void SetDevice() const {}
    virtual void Synchronize() const {}

    std::string ToString() const;

    friend std::ostream &operator<<(std::ostream &os, const Device &device);

protected:
    Device(DeviceType type, int8_t index);

    DeviceType type_;
    int8_t index_;
};

class CpuDevice : public Device {
private:
    CpuDevice();

    friend class DeviceManager;
};

#ifdef USE_CUDA
class CudaDevice : public Device {
public:
    ~CudaDevice();

    void SetDevice() const override;
    void Synchronize() const override;

    cudaStream_t Stream() const;

    cublasHandle_t CublasHandle() const;
#ifdef USE_NCCL
    ncclComm_t NcclComm() const;
#endif

private:
    CudaDevice(int8_t index);

    cudaStream_t stream_ = nullptr;

    cublasHandle_t cublas_handle_ = nullptr;
#ifdef USE_NCCL
    ncclComm_t nccl_comm_ = nullptr;
#endif

    friend class DeviceManager;
};
#endif

class DeviceManager {
public:
    static const DeviceManager *Instance();

    const Device *GetDevice(DeviceType type, int8_t index = 0) const;

    const Device *GetDefaultDevice() const;

    std::vector<const Device *> GetAllAvailableDevices(DeviceType device_type) const;

private:
    DeviceManager();

#ifdef USE_NCCL
    void InitNcclCommunicators();
#endif

    std::unordered_map<DeviceType, std::vector<std::unique_ptr<Device>>> devices_map_;
};
} // namespace infini_train

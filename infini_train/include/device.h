#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include <cublas_v2.h>
#endif

#ifdef USE_MACA
#include <mcr/mc_runtime.h>
#include <mcblas/mcblas.h>
#endif

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/rank.h"

namespace infini_train {

enum class DeviceType : int8_t {
    kCPU = 0,
    kCUDA = 1,
    kMACA = 2,
    kCount = 3,
};

class DeviceManager;

class Device {
public:
    virtual ~Device() = default;

    DeviceType Type() const;
    int8_t Index() const;

    bool IsCPU() const;
    bool IsCUDA() const;
    bool IsMACA() const;

    virtual void SetDevice() const {}
    virtual void Synchronize() const {}

    std::string ToString() const;

    virtual nn::parallel::Rank rank() const;

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
    ~CudaDevice() override;

    void SetDevice() const override;
    void Synchronize() const override;

    cudaStream_t Stream() const;

    cublasHandle_t CublasHandle() const;

    nn::parallel::Rank rank() const override;

private:
    CudaDevice(int8_t index);

    cudaStream_t stream_ = nullptr;

    cublasHandle_t cublas_handle_ = nullptr;

    nn::parallel::Rank rank_;

    friend class DeviceManager;
};
#endif

#ifdef USE_MACA
class MacaDevice : public Device {
public:
    ~MacaDevice() override;

    void SetDevice() const override;
    void Synchronize() const override;

    mcStream_t Stream() const;

    mcblasHandle_t McblasHandle() const;

    nn::parallel::Rank rank() const override;

private:
    explicit MacaDevice(int8_t index);

    mcStream_t stream_ = nullptr;

    mcblasHandle_t mcblas_handle_ = nullptr;

    nn::parallel::Rank rank_;

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

    std::unordered_map<DeviceType, std::vector<std::unique_ptr<Device>>> devices_map_;
};
} // namespace infini_train

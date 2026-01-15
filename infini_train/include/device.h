#pragma once

#include <cstdint>
#include <ostream>
#include <string>

#include "infini_train/include/nn/parallel/rank.h"

namespace infini_train {

class Device {
public:
    enum class DeviceType : int8_t {
        kCPU = 0,
        kCUDA = 1,
        kCount = 2,
        kInvalid = -1,
    };

    Device();

    Device(DeviceType type, int8_t index);

    Device &operator=(const Device &) = default;

    ~Device() = default;

    DeviceType type() const;
    int8_t index() const;

    bool IsCPU() const;
    bool IsCUDA() const;

    std::string ToString() const;

    virtual nn::parallel::Rank Rank() const;

    friend std::ostream &operator<<(std::ostream &os, const Device &device);

    friend bool operator==(const Device &a, const Device &b);

    friend bool operator!=(const Device &a, const Device &b);

private:
    DeviceType type_ = DeviceType::kInvalid;
    int8_t index_ = -1;
};

} // namespace infini_train

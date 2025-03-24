#pragma once

#include <cstdint>

#include "glog/logging.h"

namespace infini_train {
enum class DeviceType : int8_t {
    kCPU = 0,
    kCUDA = 1,
};

class Device {
public:
    Device();

    Device(DeviceType type, int8_t index);

    bool operator==(const Device &other) const;

    bool operator!=(const Device &other) const;

    DeviceType Type() const;
    int8_t Index() const;

    bool IsCPU() const;
    bool IsCUDA() const;

    std::string ToString() const;

    friend std::ostream &operator<<(std::ostream &os, const Device &device);

private:
    DeviceType type_;
    int8_t index_;
};

} // namespace infini_train

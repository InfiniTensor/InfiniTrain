#include "infini_train/include/device.h"

#include <cstdint>

#include "glog/logging.h"

namespace infini_train {
Device::Device() : type_(DeviceType::kCPU), index_(0) {}

Device::Device(DeviceType type, int8_t index) : type_(type), index_(index) {
    if (type_ == DeviceType::kCPU && index_ != 0) {
        LOG(FATAL) << "CPU device index should be 0";
    }
    // TODO(dcj): Support CUDA device with index != 0 later.
    if (type_ == DeviceType::kCUDA && index_ != 0) {
        // TODO(dcj): Check if indexed CUDA device is valid later.
        LOG(FATAL) << "CUDA device index should be 0";
    }
}

bool Device::operator==(const Device &other) const { return type_ == other.type_ && index_ == other.index_; }

bool Device::operator!=(const Device &other) const { return !(*this == other); }

DeviceType Device::Type() const { return type_; }
int8_t Device::Index() const { return index_; }

bool Device::IsCPU() const { return type_ == DeviceType::kCPU; }
bool Device::IsCUDA() const { return type_ == DeviceType::kCUDA; }

std::string Device::ToString() const {
    std::ostringstream oss;
    oss << "Device(" << (type_ == DeviceType::kCPU ? "CPU" : "CUDA") << ", " << static_cast<int>(index_) << ")";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const Device &device) {
    os << device.ToString();
    return os;
}

} // namespace infini_train

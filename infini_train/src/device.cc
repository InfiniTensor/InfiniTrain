#include "infini_train/include/device.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/global.h"

namespace infini_train {
Device::Device() : type_(DeviceType::kCPU), index_(0) {}

Device::Device(DeviceType type, int8_t index) : type_(type), index_(index) {
    if (type_ == DeviceType::kCPU && index_ != 0) {
        LOG(FATAL) << "CPU device index should be 0";
    }
}

Device::DeviceType Device::type() const { return type_; }

int8_t Device::index() const { return index_; }

bool Device::IsCPU() const { return type_ == DeviceType::kCPU; }

bool Device::IsCUDA() const { return type_ == DeviceType::kCUDA; }

<<<<<<< ours
std::string Device::ToString() const {
    std::ostringstream oss;
    oss << std::format("Device({}, {})", type_ == DeviceType::kCPU ? "CPU" : "CUDA", index_);
=======
bool Device::IsMACA() const { return type_ == DeviceType::kMACA; }

bool Device::IsDCU() const { return type_ == DeviceType::kDCU; }

std::string Device::ToString() const {
    const char *type_str = "Unknown";
    switch (type_) {
    case DeviceType::kCPU:
        type_str = "CPU";
        break;
    case DeviceType::kCUDA:
        type_str = "CUDA";
        break;
    case DeviceType::kMACA:
        type_str = "MACA";
        break;
    case DeviceType::kDCU:
        type_str = "DCU";
        break;
    default:
        break;
    }
    std::ostringstream oss;
    oss << "Device(" << type_str << ", " << static_cast<int>(index_) << ")";
>>>>>>> theirs
    return oss.str();
}

nn::parallel::Rank Device::Rank() const {
    return {nn::parallel::global::GetGlobalProcRank(), index_, nn::parallel::global::GetNprocPerNode(),
            nn::parallel::global::GetNthreadPerProc()};
}

std::ostream &operator<<(std::ostream &os, const Device &device) {
    os << device.ToString();
    return os;
}

bool operator==(const Device &a, const Device &b) { return a.type_ == b.type_ && a.index_ == b.index_; }

bool operator!=(const Device &a, const Device &b) { return !(a == b); }

} // namespace infini_train

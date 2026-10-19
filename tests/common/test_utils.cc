#include "tests/common/test_utils.h"

#include "infini_train/include/core/runtime/device_guard.h"

namespace infini_train::test {

Device InfiniTrainTest::GetDevice() const {
#if defined(INFINI_TRAIN_TEST_DEVICE_INDEX)
    return Device(GetParam(), INFINI_TRAIN_TEST_DEVICE_INDEX);
#else
    return Device(GetParam(), 0);
#endif
}

int InfiniTrainTest::DeviceCount() const { return core::GetDeviceGuardImpl(GetDevice().type())->DeviceCount(); }

bool InfiniTrainTest::IsAccelerator() const { return GetDevice().type() != Device::DeviceType::kCPU; }

} // namespace infini_train::test

#include "infini_train/include/core/runtime/stream_guard.h"

namespace infini_train::core {

StreamGuard::StreamGuard(Device device, Stream *stream)
    : device_guard_(device), impl_(GetDeviceGuardImpl(device.type())),
      previous_stream_(impl_->ExchangeStream(device, stream)) {}

StreamGuard::~StreamGuard() {
    impl_->ExchangeStream(device_guard_.current_device(), previous_stream_);
    // device_guard_ restores the original device after the stream has been restored.
}

} // namespace infini_train::core

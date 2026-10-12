#pragma once

#include "infini_train/include/core/runtime/device_guard.h"

namespace infini_train::core {

// Select device and its current stream for this scope. On destruction, restore
// the target device's previous stream, then restore the original device within
// the same backend. The stream is borrowed, must belong to device, and must
// outlive this guard. No synchronization is performed; callers remain
// responsible for cross-stream dependencies.
class StreamGuard {
 public:
  explicit StreamGuard(Device device, Stream *stream);
  ~StreamGuard();

  StreamGuard(const StreamGuard &) = delete;
  StreamGuard &operator=(const StreamGuard &) = delete;
  StreamGuard(StreamGuard &&) = delete;
  StreamGuard &operator=(StreamGuard &&) = delete;

 private:
  // Declared first so device restoration also runs if stream exchange throws
  // during construction.
  DeviceGuard device_guard_;
  DeviceGuardImpl *impl_;
  Stream *previous_stream_;
};

}  // namespace infini_train::core

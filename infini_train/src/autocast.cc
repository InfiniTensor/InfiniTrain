#include "infini_train/include/autocast.h"

namespace infini_train {

AutocastGuard::AutocastGuard(const AutocastContext &context) {
    saved_context_ = tls_autocast_context;
    tls_autocast_context = context;
}

AutocastGuard::AutocastGuard(Device::DeviceType device_type, DataType autocast_dtype) {
    saved_context_ = tls_autocast_context;
    tls_autocast_context.enabled = true;
    tls_autocast_context.device_type = device_type;
    tls_autocast_context.autocast_dtype = autocast_dtype;
}

AutocastGuard::~AutocastGuard() { tls_autocast_context = saved_context_; }

void AutocastGuard::Enable() const { tls_autocast_context.enabled = true; }

void AutocastGuard::Disable() const { tls_autocast_context.enabled = false; }

bool AutocastGuard::IsEnabled() const { return tls_autocast_context.enabled; }

} // namespace infini_train

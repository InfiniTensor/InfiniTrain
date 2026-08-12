#include "infini_train/include/core/privateuse1_backend.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <string_view>

#include "glog/logging.h"

#include "infini_train/include/core/ccl/ccl.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"

namespace infini_train::core {
namespace {

struct PrivateUse1BackendState {
    enum class Status {
        kUnregistered,
        kRegistering,
        kRegistered,
    };

    Status status = Status::kUnregistered;
    std::string name;
};

PrivateUse1BackendState g_backend_state;
std::mutex g_backend_mutex;

constexpr std::array<std::string_view, 4> kRequiredKernels = {
    "Cast",
    "Fill",
    "NoOpForward",
    "NoOpBackward",
};

bool IsValidBackendName(std::string_view name) {
    return !name.empty() && name != "cpu" && name != "cuda"
        && std::all_of(name.begin(), name.end(),
                       [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'; });
}

} // namespace

void RegisterPrivateUse1Backend(const PrivateUse1BackendRegistration &registration) {
    CHECK(IsValidBackendName(registration.name))
        << "PrivateUse1 backend name must be non-reserved and contain only lowercase ASCII letters, digits, and "
           "underscores";
    CHECK(registration.register_runtime != nullptr) << "PrivateUse1 backend must register a runtime";
    CHECK(registration.register_kernels != nullptr) << "PrivateUse1 backend must register kernels";

    {
        std::lock_guard<std::mutex> lock(g_backend_mutex);
        CHECK(g_backend_state.status == PrivateUse1BackendState::Status::kUnregistered)
            << "PrivateUse1 backend is already registered or registration is in progress as " << g_backend_state.name;
        g_backend_state.status = PrivateUse1BackendState::Status::kRegistering;
        g_backend_state.name = registration.name;
    }

    // Backend callbacks may query the provider metadata, so they must run
    // outside g_backend_mutex.
    registration.register_runtime();
    CHECK(DeviceGuardImplRegistry::Instance().Has(Device::DeviceType::kPrivateUse1))
        << "PrivateUse1 runtime callback did not register DeviceGuardImpl";

    if (registration.register_ccl != nullptr) {
        registration.register_ccl();
        CHECK(CclImplRegistry::Instance().Has(Device::DeviceType::kPrivateUse1))
            << "PrivateUse1 CCL callback did not register CclImpl";
    }

    registration.register_kernels();
    for (const auto kernel : kRequiredKernels) {
        CHECK(Dispatcher::Instance().HasKernel({Device::DeviceType::kPrivateUse1, std::string(kernel)}))
            << "PrivateUse1 backend is missing required kernel " << kernel;
    }

    {
        std::lock_guard<std::mutex> lock(g_backend_mutex);
        CHECK(g_backend_state.status == PrivateUse1BackendState::Status::kRegistering);
        g_backend_state.status = PrivateUse1BackendState::Status::kRegistered;
    }
}

bool HasPrivateUse1Backend() {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    return g_backend_state.status == PrivateUse1BackendState::Status::kRegistered;
}

std::string GetPrivateUse1BackendName() {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    return g_backend_state.status == PrivateUse1BackendState::Status::kUnregistered ? "privateuse1"
                                                                                    : g_backend_state.name;
}

} // namespace infini_train::core

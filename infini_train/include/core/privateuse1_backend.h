#pragma once

#include <optional>
#include <string>

#include "infini_train/include/datatype.h"

namespace infini_train::core {

// A PrivateUse1 provider must expose an explicitly referenced, idempotent registration entry point
// (conventionally RegisterBackend) and call it before parsing or using the provider device. Registration must not
// rely solely on file-scope static initialization because an unreferenced static archive member may be discarded.
using PrivateUse1RegistrationCallback = void (*)();

struct PrivateUse1BackendRegistration {
    std::string name;
    // Required at registration; optional representation allows omission to be diagnosed.
    std::optional<DataType> default_autocast_dtype;
    PrivateUse1RegistrationCallback register_runtime = nullptr;
    PrivateUse1RegistrationCallback register_kernels = nullptr;
    PrivateUse1RegistrationCallback register_ccl = nullptr;
};

void RegisterPrivateUse1Backend(const PrivateUse1BackendRegistration &registration);

bool HasPrivateUse1Backend();

std::string GetPrivateUse1BackendName();

DataType GetPrivateUse1BackendDefaultAutocastDtype();

} // namespace infini_train::core

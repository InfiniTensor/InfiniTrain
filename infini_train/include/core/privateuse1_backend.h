#pragma once

#include <string>

namespace infini_train::core {

using PrivateUse1RegistrationCallback = void (*)();

struct PrivateUse1BackendRegistration {
    std::string name;
    PrivateUse1RegistrationCallback register_runtime = nullptr;
    PrivateUse1RegistrationCallback register_kernels = nullptr;
    PrivateUse1RegistrationCallback register_ccl = nullptr;
};

void RegisterPrivateUse1Backend(const PrivateUse1BackendRegistration &registration);

bool HasPrivateUse1Backend();

std::string GetPrivateUse1BackendName();

} // namespace infini_train::core

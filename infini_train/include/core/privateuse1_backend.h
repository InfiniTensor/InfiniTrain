#pragma once

#include <optional>
#include <string>

#include "infini_train/include/datatype.h"

namespace infini_train::core {

// A provider exposes an explicitly referenced metadata registrar. Runtime, kernel,
// and optional CCL implementations register through static objects; the final
// executable must retain those registration-only archive members at link time.
struct PrivateUse1BackendRegistration {
    std::string name;
    // Required at registration; optional representation allows omission to be diagnosed.
    std::optional<DataType> default_autocast_dtype;
};

void RegisterPrivateUse1Backend(const PrivateUse1BackendRegistration &registration);

bool HasPrivateUse1Backend();

std::string GetPrivateUse1BackendName();

DataType GetPrivateUse1BackendDefaultAutocastDtype();

} // namespace infini_train::core

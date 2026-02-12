#pragma once

#include <cstdint>

namespace infini_train::core {

// Generic runtime status for backend-agnostic control flow.
#define INFINI_TRAIN_RUNTIME_STATUS_LIST(X)                                                                            \
    X(kSuccess, 0)                                                                                                     \
    X(kNotReady, 1)                                                                                                    \
    X(kTimeout, 2)                                                                                                     \
    X(kError, -1)                                                                                                      \
    X(kInvalidArgument, -2)                                                                                            \
    X(kOutOfMemory, -3)                                                                                                \
    X(kUnavailable, -4)                                                                                                \
    X(kNotSupported, -5)                                                                                               \
    X(kAlreadyExists, -6)                                                                                              \
    X(kPermissionDenied, -7)                                                                                           \
    X(kInternal, -8)                                                                                                   \
    X(kUnknown, -127)

enum class RuntimeStatus : int32_t {
#define INFINI_TRAIN_RUNTIME_STATUS_ENUM_ITEM(name, value) name = value,
    INFINI_TRAIN_RUNTIME_STATUS_LIST(INFINI_TRAIN_RUNTIME_STATUS_ENUM_ITEM)
#undef INFINI_TRAIN_RUNTIME_STATUS_ENUM_ITEM
};

inline const char *RuntimeStatusToString(RuntimeStatus s) {
    switch (s) {
#define INFINI_TRAIN_RUNTIME_STATUS_CASE(name, value)                                                                  \
    case RuntimeStatus::name:                                                                                          \
        return #name;
        INFINI_TRAIN_RUNTIME_STATUS_LIST(INFINI_TRAIN_RUNTIME_STATUS_CASE)
#undef INFINI_TRAIN_RUNTIME_STATUS_CASE
    default:
        return "Unknown";
    }
}

#undef INFINI_TRAIN_RUNTIME_STATUS_LIST

} // namespace infini_train::core

#pragma once

#include <cstdint>

namespace infini_train::core {

#define INFINI_TRAIN_CCL_STATUS_LIST(X)                                                                                \
    X(kSuccess, 0)                                                                                                     \
    X(kInProgress, 1)                                                                                                  \
    X(kTimeout, 2)                                                                                                     \
    X(kError, -1)                                                                                                      \
    X(kInvalidArgument, -2)                                                                                            \
    X(kUnavailable, -3)                                                                                                \
    X(kNotSupported, -4)                                                                                               \
    X(kInternal, -5)                                                                                                   \
    X(kUnknown, -127)

enum class CclStatus : int32_t {
#define INFINI_TRAIN_CCL_STATUS_ENUM_ITEM(name, value) name = value,
    INFINI_TRAIN_CCL_STATUS_LIST(INFINI_TRAIN_CCL_STATUS_ENUM_ITEM)
#undef INFINI_TRAIN_CCL_STATUS_ENUM_ITEM
};

inline const char *CclStatusToString(CclStatus status) {
    switch (status) {
#define INFINI_TRAIN_CCL_STATUS_CASE(name, value)                                                                      \
    case CclStatus::name:                                                                                              \
        return #name;
        INFINI_TRAIN_CCL_STATUS_LIST(INFINI_TRAIN_CCL_STATUS_CASE)
#undef INFINI_TRAIN_CCL_STATUS_CASE
    default:
        return "Unknown";
    }
}

#undef INFINI_TRAIN_CCL_STATUS_LIST

class CclComm {
public:
    CclComm() = default;
    virtual ~CclComm() = default;
};

class CclUniqueId {
public:
    CclUniqueId() = default;
    virtual ~CclUniqueId() = default;
};

} // namespace infini_train::core

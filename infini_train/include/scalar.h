#pragma once

#include <cstdint>
#include <type_traits>

#include "glog/logging.h"

#include "infini_train/include/common/cpu/common_cpu.h"

namespace infini_train {

struct Scalar {
    enum class Kind : uint8_t { kBool, kDouble, kInt64, kUInt64 };

    Scalar() : kind(Kind::kInt64), i(0) {}
    Scalar(bool v) : kind(Kind::kBool), u(v ? 1 : 0) {}

    template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    Scalar(T v) : kind(Kind::kDouble), d(static_cast<double>(v)) {}

    template <typename T,
              typename std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T> && !std::is_same_v<T, bool>, int>
              = 0>
    Scalar(T v) : kind(Kind::kInt64), i(static_cast<int64_t>(v)) {}

    template <typename T,
              typename std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T> && !std::is_same_v<T, bool>, int>
              = 0>
    Scalar(T v) : kind(Kind::kUInt64), u(static_cast<uint64_t>(v)) {}

    Scalar(FP16 v) : kind(Kind::kDouble), d(static_cast<float>(v)) {}
    Scalar(BF16 v) : kind(Kind::kDouble), d(static_cast<float>(v)) {}

    template <typename T> T to() const {
        switch (kind) {
        case Kind::kBool:
            return common::cpu::Cast<T>(u != 0);
        case Kind::kDouble:
            return common::cpu::Cast<T>(d);
        case Kind::kInt64:
            return common::cpu::Cast<T>(i);
        case Kind::kUInt64:
            return common::cpu::Cast<T>(u);
        default:
            LOG(FATAL) << "Unknown scalar kind";
        }

        std::abort();
    }

    Kind kind;
    union {
        double d;
        int64_t i;
        uint64_t u;
    };
};

} // namespace infini_train

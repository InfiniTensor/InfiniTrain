#pragma once

#include <cmath>
#include <limits>

#include "glog/logging.h"

namespace infini_train::detail {

inline void CheckUniformBounds(float from, float to) {
    CHECK_LE(from, to);
    CHECK(std::isfinite(from)) << "Uniform lower bound must be finite";
    CHECK(std::isfinite(to)) << "Uniform upper bound must be finite";
    const double range = static_cast<double>(to) - static_cast<double>(from);
    CHECK_LE(range, static_cast<double>(std::numeric_limits<float>::max()))
        << "Uniform bounds range exceeds float maximum";
}

} // namespace infini_train::detail

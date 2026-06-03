#pragma once

#include <vector>

#include "glog/logging.h"

namespace infini_train::utils {
std::string DimsToString(const std::vector<int64_t> &dims);
}

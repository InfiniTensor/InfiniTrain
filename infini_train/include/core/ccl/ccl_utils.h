#pragma once

#include <string>

#include "infini_train/include/core/ccl/ccl_common.h"

namespace infini_train::core {

void WriteUniqueIdFile(const CclUniqueId &unique_id, const std::string &pg_name);

void ReadUniqueIdFile(CclUniqueId *unique_id, const std::string &pg_name);

void CleanupUniqueIdFile(const std::string &pg_name);

} // namespace infini_train::core

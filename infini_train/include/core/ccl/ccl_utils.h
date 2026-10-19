#pragma once

#include <string>

namespace infini_train::core {
class CclUniqueId;

void WriteUniqueIdFile(const CclUniqueId &unique_id,
                       const std::string &pg_name);

void ReadUniqueIdFile(CclUniqueId *unique_id, const std::string &pg_name);

void CleanupUniqueIdFile(const std::string &pg_name);

}  // namespace infini_train::core

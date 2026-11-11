#pragma once

#include <string>

namespace infini_train::core {
class CclUniqueId;

void WriteUniqueId(const CclUniqueId &unique_id, const std::string &pg_name);

void ReadUniqueId(CclUniqueId *unique_id, const std::string &pg_name);

}  // namespace infini_train::core

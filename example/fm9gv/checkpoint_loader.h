#pragma once

#include <memory>
#include <string>

#include "example/fm9gv/model.h"

namespace fm9gv {
std::shared_ptr<Model> LoadFromFM9GBin(const std::string &filepath);
} // namespace fm9gv

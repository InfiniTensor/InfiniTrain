#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include "infini_train/include/checkpoint/load_planner.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::checkpoint {

using LoadedStateDict = std::unordered_map<std::string, std::shared_ptr<Tensor>>;

class LoadStrategy {
public:
    virtual ~LoadStrategy() = default;
    virtual LoadedStateDict Execute(const std::filesystem::path &checkpoint_dir, const LoadPlan &plan) = 0;
};

/// Reads source regions directly from metadata offsets while caching one open stream per file.
class IndexedRegionLoadStrategy final : public LoadStrategy {
public:
    LoadedStateDict Execute(const std::filesystem::path &checkpoint_dir, const LoadPlan &plan) override;
};

} // namespace infini_train::checkpoint

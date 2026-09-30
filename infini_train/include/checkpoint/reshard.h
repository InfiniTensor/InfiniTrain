#pragma once

#include <filesystem>

#include "infini_train/include/checkpoint/checkpoint.h"

namespace infini_train {
class LRScheduler;
class Optimizer;
namespace nn {
class Module;
}
} // namespace infini_train

namespace infini_train::checkpoint {

// Restore this rank's target model shards from a distributed checkpoint.
void LoadDistributedCheckpoint(const std::filesystem::path &checkpoint_dir, nn::Module &model, Optimizer *optimizer,
                               TrainerState &state, LRScheduler *lr_scheduler,
                               const Checkpoint::CheckpointMetadata &metadata);

} // namespace infini_train::checkpoint

#pragma once

namespace infini_train::checkpoint {

inline constexpr char kModelCheckpointFilename[] = "model.ckpt";
inline constexpr char kOptimizerCheckpointFilename[] = "optimizer.ckpt";
inline constexpr char kMetadataFilename[] = "metadata.json";
inline constexpr char kTemporaryMetadataFilename[] = "metadata.json.tmp";
inline constexpr char kTrainerStateFilename[] = "trainer_state.json";
inline constexpr char kLRSchedulerFilename[] = "lr_scheduler.ckpt";
inline constexpr char kLatestIterationFilename[] =
    "latest_checkpointed_iteration.txt";
inline constexpr char kTemporaryLatestIterationFilename[] =
    "latest_checkpointed_iteration.txt.tmp";
inline constexpr int kDistributedCheckpointVersion = 3;

}  // namespace infini_train::checkpoint

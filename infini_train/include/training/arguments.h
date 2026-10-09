#pragma once

#include <filesystem>

#include "infini_train/include/training/training.h"

namespace infini_train::training {

// Optional gflags adapter, provided by InfiniTrain::training_cli. The training
// runtime itself accepts typed options and does not register command-line flags.
TrainingOptions TrainingOptionsFromFlags();

// Parse gflags with strict flag-file validation. A relative default file is
// resolved beside the executable. Missing defaults may be replaced by an
// explicit --flagfile. Invalid input exits with status 1, like gflags itself.
// Call once during startup, before worker threads are created.
void ParseTrainingFlags(int *argc, char ***argv, const std::filesystem::path &default_flagfile = {});

} // namespace infini_train::training

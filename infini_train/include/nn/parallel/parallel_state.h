#pragma once

#include "infini_train/include/device.h"

namespace infini_train::nn::parallel {

struct ModelParallelInitOptions {
    Device::DeviceType backend;
    bool initialize_expert_groups;
};

// Creates all non-trivial model-parallel process groups in a deterministic
// global order. GlobalEnv must be initialized before calling this function.
// All distributed processes must call it with identical options before worker
// execution starts.
void InitializeModelParallel(const ModelParallelInitOptions &options);

} // namespace infini_train::nn::parallel

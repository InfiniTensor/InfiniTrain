#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::utils::checkpoint {

using CheckpointForwardFn
    = std::function<std::vector<std::shared_ptr<Tensor>>(const std::vector<std::shared_ptr<Tensor>> &)>;

std::vector<std::shared_ptr<Tensor>> Checkpoint(const CheckpointForwardFn &forward_fn,
                                                const std::vector<std::shared_ptr<Tensor>> &inputs,
                                                bool use_reentrant = false, bool preserve_rng_state = false,
                                                bool determinism_check = true, bool early_stop = true);

} // namespace infini_train::utils::checkpoint

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::utils::checkpoint {

class CheckpointFunction : public autograd::Function {
public:
    using ForwardFn = std::function<std::vector<std::shared_ptr<Tensor>>(const std::vector<std::shared_ptr<Tensor>> &)>;

    explicit CheckpointFunction(ForwardFn forward_fn);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    ForwardFn forward_fn_;

    std::vector<std::shared_ptr<Tensor>> saved_inputs_;
    std::vector<bool> saved_inputs_requires_grad_;
    AutocastState saved_autocast_;
};

// Reentrant activation checkpointing (torch.utils.checkpoint.checkpoint style).
std::vector<std::shared_ptr<Tensor>> Checkpoint(const CheckpointFunction::ForwardFn &forward_fn,
                                                const std::vector<std::shared_ptr<Tensor>> &inputs,
                                                bool use_reentrant = false, bool preserve_rng_state = true,
                                                bool determinism_check = true, bool early_stop = true);

} // namespace infini_train::utils::checkpoint

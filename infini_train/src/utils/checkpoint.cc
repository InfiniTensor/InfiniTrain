#include "infini_train/include/utils/checkpoint.h"

#include <algorithm>
#include <atomic>
#include <unordered_map>

#include "glog/logging.h"

#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/tensor.h"

namespace infini_train::utils::checkpoint {
namespace {
constexpr char kCheckpointType[] = "Checkpoint";
std::atomic<int64_t> g_checkpoint_gid{0};

int64_t NextCheckpointGid() { return g_checkpoint_gid.fetch_add(1) + 1; }

struct SavedTensorMeta {
    std::vector<int64_t> dims;
    DataType dtype = DataType::kFLOAT32;
    Device::DeviceType device_type = Device::DeviceType::kCPU;
};

bool MetaEquals(const SavedTensorMeta &a, const SavedTensorMeta &b) {
    return a.dtype == b.dtype && a.device_type == b.device_type && a.dims == b.dims;
}

struct CheckpointFrame;

struct SavedTensorHolder {
    std::shared_ptr<CheckpointFrame> frame;
    size_t index = 0;
    std::unordered_map<int64_t, bool> handles;
    std::shared_ptr<Tensor> tensor;
};

struct CheckpointFrame {
    CheckpointFunction::ForwardFn forward_fn;
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<bool> inputs_requires_grad;
    AutocastState autocast_state;
    std::vector<SavedTensorMeta> forward_metas;
    bool early_stop = true;
    bool determinism_check = true;
    std::unordered_map<int64_t, bool> recomputed;
    std::vector<std::weak_ptr<SavedTensorHolder>> weak_holders;
    int64_t current_gid = -1;

    struct StopRecomputeError : public std::exception {};

    bool HasActiveHandles(int64_t gid) const {
        for (const auto &weak_holder : weak_holders) {
            auto holder = weak_holder.lock();
            if (!holder) {
                continue;
            }
            auto it = holder->handles.find(gid);
            if (it != holder->handles.end() && it->second) {
                return true;
            }
        }
        return false;
    }

    size_t CountAliveHolders() const {
        size_t alive = 0;
        for (const auto &weak_holder : weak_holders) {
            if (weak_holder.lock()) {
                ++alive;
            }
        }
        return alive;
    }

    int64_t GetOrCreateGid() {
        if (current_gid < 0) {
            current_gid = NextCheckpointGid();
            return current_gid;
        }
        if (recomputed[current_gid] && !HasActiveHandles(current_gid)) {
            current_gid = NextCheckpointGid();
        }
        return current_gid;
    }

    void Recompute(int64_t gid) {
        if (recomputed[gid]) {
            return;
        }
        const size_t alive_needed = CountAliveHolders();
        size_t filled = 0;
        size_t recompute_index = 0;

        std::vector<std::shared_ptr<Tensor>> detached_inputs;
        detached_inputs.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i]) {
                detached_inputs.push_back(nullptr);
                continue;
            }
            auto detached = inputs[i]->Detach();
            detached->set_requires_grad(inputs_requires_grad[i]);
            detached->set_is_leaf(true);
            detached->set_grad_fn(nullptr);
            detached_inputs.push_back(detached);
        }

        auto prev_autocast = GetAutocastState();
        SetAutocastState(autocast_state);
        // Unlike PyTorch's engine, this autograd implementation mutates
        // dependency counters eagerly while building a graph. The recompute
        // graph is not traversed by non-reentrant checkpoint here, so building
        // it would pollute parameter/input dependency counts and break
        // accumulation. Recompute under no-grad, while Function::Apply still
        // propagates requires_grad metadata and populates needs_input_grad_ so
        // SetupContext saves the same tensors as the original forward.
        autograd::NoGradGuard no_grad;
        autograd::PropagateRequiresGradGuard propagate_requires_grad;

        autograd::FunctionCtx::SavedTensorHooks hooks;
        hooks.pack = [this, gid, alive_needed, &filled,
                      &recompute_index](const std::shared_ptr<Tensor> &tensor) -> std::shared_ptr<void> {
            size_t idx = recompute_index++;
            if (idx >= weak_holders.size()) {
                LOG(FATAL) << "Checkpoint: recomputed more tensors than saved during forward.";
            }
            auto holder = weak_holders[idx].lock();
            if (tensor) {
                if (determinism_check) {
                    SavedTensorMeta meta;
                    meta.dims = tensor->Dims();
                    meta.dtype = tensor->Dtype();
                    meta.device_type = tensor->GetDevice().type();
                    if (!MetaEquals(meta, forward_metas[idx])) {
                        LOG(FATAL) << "Checkpoint: recomputed tensor metadata mismatch at index " << idx << ".";
                    }
                }
                if (holder) {
                    holder->handles[gid] = true;
                    holder->tensor = tensor;
                    ++filled;
                }
            } else {
                if (holder) {
                    holder->handles[gid] = true;
                    holder->tensor.reset();
                    ++filled;
                }
            }
            if (early_stop && filled >= alive_needed) {
                throw StopRecomputeError();
            }
            return tensor;
        };
        hooks.unpack = [](const std::shared_ptr<void> &state) -> std::shared_ptr<Tensor> {
            return std::static_pointer_cast<Tensor>(state);
        };
        autograd::FunctionCtx::SavedTensorHooksGuard guard(std::move(hooks));

        try {
            forward_fn(detached_inputs);
        } catch (const StopRecomputeError &) {
            // Early-stop: expected when all needed tensors are recomputed.
        }
        if (filled < alive_needed) {
            LOG(FATAL) << "Checkpoint: recomputed fewer tensors (" << filled << ") than required (" << alive_needed
                       << ").";
        }

        SetAutocastState(prev_autocast);
        recomputed[gid] = true;

        // Break potential reference cycles once recomputation is done.
        inputs.clear();
        inputs_requires_grad.clear();
        forward_fn = nullptr;
    }
};
} // namespace

CheckpointFunction::CheckpointFunction(ForwardFn forward_fn)
    : autograd::Function(kCheckpointType), forward_fn_(std::move(forward_fn)) {}

std::vector<std::shared_ptr<Tensor>>
CheckpointFunction::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    saved_autocast_ = GetAutocastState();
    saved_inputs_.clear();
    saved_inputs_requires_grad_.clear();
    saved_inputs_.reserve(input_tensors.size());
    saved_inputs_requires_grad_.reserve(input_tensors.size());
    for (const auto &input : input_tensors) {
        if (!input) {
            saved_inputs_requires_grad_.push_back(false);
            saved_inputs_.push_back(nullptr);
            continue;
        }
        saved_inputs_requires_grad_.push_back(input->requires_grad());
        saved_inputs_.push_back(input->Detach());
    }

    // TODO(zbl): RNG state is not captured yet. Dropout or random ops are not supported.
    return forward_fn_(input_tensors);
}

void CheckpointFunction::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                                      const std::vector<std::shared_ptr<Tensor>> &) {
    // Intentionally empty: checkpoint avoids saving intermediate tensors.
}

std::vector<std::shared_ptr<Tensor>>
CheckpointFunction::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // TODO(zbl): RNG state is not captured yet. Dropout or random ops are not supported.

    CHECK(!saved_inputs_.empty());
    CHECK_EQ(grad_outputs.size(), 1) << "Checkpoint currently supports single-output forward only.";

    auto prev_autocast = GetAutocastState();
    SetAutocastState(saved_autocast_);
    autograd::EnableGradGuard enable_grad;

    std::vector<std::shared_ptr<Tensor>> detached_inputs;
    detached_inputs.reserve(saved_inputs_.size());
    for (size_t i = 0; i < saved_inputs_.size(); ++i) {
        if (!saved_inputs_[i]) {
            detached_inputs.push_back(nullptr);
            continue;
        }
        auto detached = saved_inputs_[i]->Detach();
        detached->set_requires_grad(saved_inputs_requires_grad_[i]);
        detached->set_is_leaf(true);
        detached->set_grad_fn(nullptr);
        detached_inputs.push_back(detached);
    }

    auto outputs = forward_fn_(detached_inputs);
    // TODO(zbl): Support multiple-output forward.
    CHECK_EQ(outputs.size(), 1) << "Checkpoint currently supports single-output forward only.";

    if (grad_outputs[0]) {
        outputs[0]->Backward(grad_outputs[0]);
    }

    SetAutocastState(prev_autocast);

    std::vector<std::shared_ptr<Tensor>> grad_inputs;
    grad_inputs.reserve(detached_inputs.size());
    for (const auto &detached : detached_inputs) {
        if (detached && detached->requires_grad()) {
            grad_inputs.push_back(detached->grad());
        } else {
            grad_inputs.push_back(nullptr);
        }
    }

    saved_inputs_.clear();
    saved_inputs_requires_grad_.clear();
    return grad_inputs;
}

std::vector<std::shared_ptr<Tensor>> Checkpoint(const CheckpointFunction::ForwardFn &forward_fn,
                                                const std::vector<std::shared_ptr<Tensor>> &inputs, bool use_reentrant,
                                                bool preserve_rng_state, bool determinism_check, bool early_stop) {
    if (preserve_rng_state) {
        // TODO(zbl): Preserve and restore RNG state for CPU/CUDA.
    }
    if (!autograd::GradMode::IsEnabled()) {
        return forward_fn(inputs);
    }

    if (use_reentrant) {
        const bool any_requires_grad = std::any_of(
            inputs.begin(), inputs.end(), [](const std::shared_ptr<Tensor> &t) { return t && t->requires_grad(); });
        if (!any_requires_grad) {
            return forward_fn(inputs);
        }
        auto func = std::make_shared<CheckpointFunction>(forward_fn);
        return func->Apply(inputs);
    }

    auto frame = std::make_shared<CheckpointFrame>();
    frame->forward_fn = forward_fn;
    frame->early_stop = early_stop;
    frame->determinism_check = determinism_check;
    frame->inputs.reserve(inputs.size());
    frame->inputs_requires_grad.reserve(inputs.size());
    for (const auto &input : inputs) {
        if (input) {
            frame->inputs.push_back(input->Detach());
            frame->inputs_requires_grad.push_back(input->requires_grad());
        } else {
            frame->inputs.push_back(nullptr);
            frame->inputs_requires_grad.push_back(false);
        }
    }
    frame->autocast_state = GetAutocastState();

    autograd::FunctionCtx::SavedTensorHooks hooks;
    hooks.pack = [frame](const std::shared_ptr<Tensor> &tensor) -> std::shared_ptr<void> {
        auto holder = std::make_shared<SavedTensorHolder>();
        holder->frame = frame;
        holder->index = frame->forward_metas.size();
        frame->weak_holders.push_back(holder);
        if (tensor) {
            SavedTensorMeta meta;
            meta.dims = tensor->Dims();
            meta.dtype = tensor->Dtype();
            meta.device_type = tensor->GetDevice().type();
            frame->forward_metas.push_back(std::move(meta));
        } else {
            frame->forward_metas.push_back({});
        }
        return holder;
    };
    hooks.unpack = [](const std::shared_ptr<void> &state) -> std::shared_ptr<Tensor> {
        auto holder = std::static_pointer_cast<SavedTensorHolder>(state);
        auto frame = holder->frame;
        const int64_t gid = frame->GetOrCreateGid();
        if (!frame->recomputed[gid]) {
            frame->Recompute(gid);
        }
        auto it = holder->handles.find(gid);
        if (it == holder->handles.end() || !it->second) {
            LOG(FATAL) << "Checkpoint: unpack called more than once for index " << holder->index << ".";
        }
        auto recomputed = holder->tensor;
        const auto &meta = frame->forward_metas[holder->index];
        if (recomputed && meta.dims.empty() && meta.dtype == DataType::kFLOAT32
            && meta.device_type == Device::DeviceType::kCPU) {
            LOG(FATAL) << "Checkpoint: recomputed non-null tensor for saved null entry.";
        }
        if (!recomputed
            && !(meta.dims.empty() && meta.dtype == DataType::kFLOAT32
                 && meta.device_type == Device::DeviceType::kCPU)) {
            LOG(FATAL) << "Checkpoint: recomputed null tensor for saved non-null entry.";
        }
        // TODO(zbl): Determinism check (shape/dtype/device) vs forward_metas.
        // Release recomputed tensor as soon as it's unpacked to reduce peak memory.
        holder->tensor.reset();
        it->second = false;
        return recomputed;
    };

    autograd::FunctionCtx::SavedTensorHooksGuard guard(std::move(hooks));
    return forward_fn(inputs);
}

} // namespace infini_train::utils::checkpoint

#include "infini_train/include/autograd/function.h"

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/accumulate.h"
#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/common/hook.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/precision_check_config.h"
#include "infini_train/include/utils/precision_checker.h"

namespace infini_train::autograd {
namespace {
thread_local std::vector<Function::SavedTensorHooks> tls_saved_tensor_hooks;
} // namespace

std::vector<std::shared_ptr<Tensor>> Function::Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 1);
    auto device = input_tensors[0]->GetDevice();
    core::DeviceGuard guard(device);

    // Register precision check hooks if enabled (before forward)
    if (!precision_check_registered_) {
        auto precision_level = utils::PrecisionCheckEnv::Instance().GetConfig().level;
        if (precision_level == utils::PrecisionCheckLevel::FUNCTION) {
            utils::PrecisionChecker::RegisterForFunction(this, type_);
            precision_check_registered_ = true;
        }
    }

    // Call forward pre-hooks
    for (const auto &hook : forward_pre_hooks_) {
        if (hook) {
            hook(this, input_tensors);
        }
    }

    // Populate needs_input_grad_ before Forward/SetupContext so that
    // SetupContext can use it for saved-tensor pruning. This must not depend
    // on GradMode: non-reentrant checkpoint recomputes under no-grad to avoid
    // wiring an unused recompute graph into the engine, but SetupContext still
    // needs the original per-input grad requirements.
    needs_input_grad_.resize(input_tensors.size());
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
        needs_input_grad_[idx] = input_tensors[idx] && input_tensors[idx]->requires_grad();
    }

    // Apply autocast once at the autograd boundary so Forward / SetupContext receive
    // tensors already in the compute dtype. The shared_ptr copies are local; we keep
    // the caller's `input_tensors` untouched so next_functions_ wires up to the
    // original autograd graph (leaf -> AccumulateGrad / non-leaf -> grad_fn).
    auto compute_inputs = input_tensors;
    for (auto &t : compute_inputs) { tls_autocast_context.Autocast(type_, t); }

    std::vector<std::shared_ptr<Tensor>> output_tensors;
    {
        autograd::NoGradGuard no_grad;
        // no_grad in autograd.Function.Forward()
        output_tensors = Forward(compute_inputs);
        SetupContext(compute_inputs, output_tensors);
    }

    // Call forward post-hooks
    for (const auto &hook : forward_post_hooks_) {
        if (hook) {
            hook(this, input_tensors, output_tensors);
        }
    }

    if (!autograd::GradMode::IsEnabled()) {
        // with no_grad: block graph building operations. Non-reentrant
        // checkpoint recomputation still needs requires_grad to flow through
        // outputs so later SetupContext calls save the same tensors as the
        // original forward.
        if (autograd::GradMode::PropagateRequiresGrad()) {
            bool output_requires_grad = false;
            for (const auto &input_tensor : input_tensors) {
                output_requires_grad |= input_tensor && input_tensor->requires_grad();
            }
            for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
                auto &output_tensor = output_tensors[output_idx];
                if (!output_tensor) {
                    continue;
                }
                output_tensor->set_requires_grad(output_requires_grad);
                output_tensor->set_grad_fn(nullptr);
                output_tensor->set_is_leaf(true);
                output_tensor->set_output_idx(output_idx);
            }
        }
        return output_tensors;
    }

    bool output_requires_grad = false;
    for (int idx = 0; idx < input_tensors.size(); ++idx) {
        const auto &input_tensor = input_tensors[idx];
        if (!input_tensor) {
            next_functions_.emplace_back(nullptr, 0);
            continue;
        }
        if (input_tensor->requires_grad() && input_tensor->is_leaf()) {
            next_functions_.emplace_back(input_tensor->grad_accumulator(), input_tensor->output_idx());
            input_tensor->grad_accumulator()->IncreaseDependenciesNumber();
        } else {
            next_functions_.emplace_back(input_tensor->grad_fn(), input_tensor->output_idx());
            if (input_tensor->grad_fn()) {
                input_tensor->grad_fn()->IncreaseDependenciesNumber();
            }
        }
        output_requires_grad |= input_tensor->requires_grad();
    }

    grad_outputs_reached_ = 0;
    grad_outputs_.resize(output_tensors.size(), nullptr);
    for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
        auto &output_tensor = output_tensors[output_idx];
        // TODO(dcj): Mark if an output tensor need differentiable or not.
        output_tensor->set_requires_grad(output_requires_grad);
        output_tensor->set_grad_fn(output_requires_grad ? shared_from_this() : nullptr);
        output_tensor->set_is_leaf(!output_requires_grad
                                   || ((output_tensor->grad_fn() == nullptr) && output_requires_grad));
        output_tensor->set_output_idx(output_idx);
    }

    return output_tensors;
}

void Function::BackwardPartial(std::shared_ptr<Tensor> grad_output, int grad_output_idx) {
    auto device = grad_output->GetDevice();
    core::DeviceGuard guard(device);

    // NOTE(dcj): The accumulate autograd function has no grad_outputs.
    // Temporarily resize the vector to hold one nullptr as a buffer.
    if (grad_outputs_.empty()) {
        grad_outputs_.resize(1, nullptr);
    }
    if (!grad_outputs_.at(grad_output_idx)) {
        grad_outputs_[grad_output_idx] = std::move(grad_output);
        ++grad_outputs_reached_;
    } else {
        auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AccumulateGrad"});
        kernel.Call<void>(grad_output, 1.0f, grad_outputs_.at(grad_output_idx));
    }
    ++dependencies_reached_;
    if (grad_outputs_reached_ == grad_outputs_.size()
        && (dependencies_reached_ == dependencies_number_ || dependencies_number_ == 0)) {

        // Call backward pre-hooks
        for (const auto &hook : backward_pre_hooks_) {
            if (hook) {
                hook(this, grad_outputs_);
            }
        }

        std::vector<std::shared_ptr<Tensor>> grad_inputs;
        {
            autograd::NoGradGuard no_grad;
            // no_grad in autograd.Function.Backward()
            grad_inputs = Backward(grad_outputs_);
        }

        // Call backward post-hooks
        for (const auto &hook : backward_post_hooks_) {
            if (hook) {
                hook(this, grad_inputs, grad_outputs_);
            }
        }

        saved_tensors_.clear();
        grad_outputs_.clear();
        needs_input_grad_.clear();
        grad_outputs_reached_ = 0;
        dependencies_reached_ = 0;

        CHECK_EQ(grad_inputs.size(), next_functions_.size());
        auto propagate_grad_input = [&](size_t idx) {
            auto grad_input = std::move(grad_inputs[idx]);
            auto &[next_function, output_idx] = next_functions_[idx];
            if (grad_input && next_function) {
                next_function->BackwardPartial(std::move(grad_input), output_idx);
            }
            grad_inputs[idx].reset();
        };

        // Send leaf gradients out first. This recursive engine keeps the
        // current function's full grad_inputs vector alive while traversing
        // earlier inputs; for ops like Linear(input, weight, bias), visiting
        // input first would retain weight/bias gradients across all preceding
        // layers. PyTorch's non-recursive engine does not have that stack
        // retention pattern, so flush AccumulateGrad edges before recursing
        // into non-leaf activation edges.
        for (size_t idx = 0; idx < grad_inputs.size(); ++idx) {
            const auto &next_function = next_functions_[idx].first;
            if (next_function && std::dynamic_pointer_cast<AccumulateGrad>(next_function)) {
                propagate_grad_input(idx);
            }
        }
        for (size_t idx = 0; idx < grad_inputs.size(); ++idx) {
            const auto &next_function = next_functions_[idx].first;
            if (next_function && !std::dynamic_pointer_cast<AccumulateGrad>(next_function)) {
                propagate_grad_input(idx);
            }
        }
        next_functions_.clear();
    }
}

void Function::SaveForBackward(const std::vector<std::shared_ptr<Tensor>> &tensors) {
    saved_tensors_.clear();
    saved_tensors_.reserve(tensors.size());
    for (const auto &tensor : tensors) {
        SavedTensorEntry entry;
        if (!tensor || tls_saved_tensor_hooks.empty()) {
            // If no hooks are registered, save the tensor itself
            entry.tensor = tensor;
        } else {
            // Otherwise, use the pack_hook to obtain the related states and save unpack hook
            const auto &hooks = tls_saved_tensor_hooks.back();
            if (!hooks.pack && !hooks.unpack) {
                entry.tensor = tensor;
            } else {
                entry.hook_state = hooks.pack ? hooks.pack(tensor) : nullptr;
                entry.unpack = hooks.unpack;
            }
        }
        saved_tensors_.push_back(std::move(entry));
    }
}

std::shared_ptr<Tensor> Function::GetSavedTensor(size_t index) const {
    CHECK_LT(index, SavedTensorsSize());
    const auto &entry = saved_tensors_[index];
    if (entry.tensor) {
        // If the tensor itself is saved, then no recomputation is needed
        return entry.tensor;
    }
    if (entry.hook_state && entry.unpack) {
        // If unpack hook is saved, then do the recomputation
        return entry.unpack(entry.hook_state);
    }
    return nullptr;
}

std::vector<std::shared_ptr<Tensor>> Function::GetSavedTensors() const {
    std::vector<std::shared_ptr<Tensor>> out;
    out.reserve(SavedTensorsSize());
    for (size_t i = 0; i < SavedTensorsSize(); ++i) { out.push_back(GetSavedTensor(i)); }
    return out;
}

Function::SavedTensorHooksGuard::SavedTensorHooksGuard(SavedTensorHooks hooks) {
    tls_saved_tensor_hooks.push_back(std::move(hooks));
    depth_ = tls_saved_tensor_hooks.size();
}

Function::SavedTensorHooksGuard::~SavedTensorHooksGuard() {
    if (tls_saved_tensor_hooks.size() == depth_) {
        // Generally depth_ should be equal to the number of hooks
        tls_saved_tensor_hooks.pop_back();
    } else if (!tls_saved_tensor_hooks.empty()) {
        LOG(WARNING) << "SavedTensorHooksGuard: redundant hooks are detected.";
        tls_saved_tensor_hooks.pop_back();
    }
}

void Function::IncreaseDependenciesNumber() { ++dependencies_number_; }

std::shared_ptr<infini_train::HookHandle> Function::RegisterForwardPreHook(FunctionPreHook hook) {
    forward_pre_hooks_.push_back(std::move(hook));
    return std::make_shared<FunctionHookHandleImpl<FunctionPreHook>>(&forward_pre_hooks_,
                                                                     forward_pre_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Function::RegisterForwardPostHook(FunctionPostHook hook) {
    forward_post_hooks_.push_back(std::move(hook));
    return std::make_shared<FunctionHookHandleImpl<FunctionPostHook>>(&forward_post_hooks_,
                                                                      forward_post_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Function::RegisterBackwardPreHook(FunctionPreHook hook) {
    backward_pre_hooks_.push_back(std::move(hook));
    return std::make_shared<FunctionHookHandleImpl<FunctionPreHook>>(&backward_pre_hooks_,
                                                                     backward_pre_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Function::RegisterBackwardPostHook(FunctionPostHook hook) {
    backward_post_hooks_.push_back(std::move(hook));
    return std::make_shared<FunctionHookHandleImpl<FunctionPostHook>>(&backward_post_hooks_,
                                                                      backward_post_hooks_.size() - 1);
}
} // namespace infini_train::autograd

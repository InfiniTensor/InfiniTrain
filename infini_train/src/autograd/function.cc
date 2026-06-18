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
thread_local std::vector<FunctionCtx::SavedTensorHooks> tls_saved_tensor_hooks;

std::shared_ptr<Tensor> ShallowCopyWithoutAutogradMeta(const std::shared_ptr<Tensor> &tensor) {
    if (!tensor) {
        return nullptr;
    }
    return std::make_shared<Tensor>(*tensor, 0, tensor->Dims());
}
} // namespace

FunctionCtx::SavedTensorHooksGuard::SavedTensorHooksGuard(SavedTensorHooks hooks) {
    CHECK(hooks.pack) << "Saved tensor pack hook must be set";
    CHECK(hooks.unpack) << "Saved tensor unpack hook must be set";
    tls_saved_tensor_hooks.push_back(std::move(hooks));
    depth_ = tls_saved_tensor_hooks.size();
}

FunctionCtx::SavedTensorHooksGuard::~SavedTensorHooksGuard() {
    if (tls_saved_tensor_hooks.empty()) {
        return;
    }
    if (tls_saved_tensor_hooks.size() != depth_) {
        LOG(WARNING) << "SavedTensorHooksGuard destroyed out of order";
    }
    tls_saved_tensor_hooks.pop_back();
}

FunctionCtx::FunctionCtx() = default;

std::vector<std::shared_ptr<Tensor>> FunctionCtx::GetSavedTensors() const {
    std::vector<std::shared_ptr<Tensor>> saved_tensors;
    saved_tensors.reserve(saved_tensor_entries_.size());
    for (const auto &entry : saved_tensor_entries_) {
        if (entry.tensor) {
            saved_tensors.push_back(entry.tensor);
        } else if (entry.unpack) {
            saved_tensors.push_back(entry.unpack(entry.hook_state));
        } else {
            saved_tensors.push_back(nullptr);
        }
    }
    return saved_tensors;
}

const std::vector<bool> &FunctionCtx::needs_input_grad() const { return needs_input_grad_; }

void FunctionCtx::SaveForBackward(const std::vector<std::shared_ptr<Tensor>> &tensors) { to_save_ = tensors; }

void FunctionCtx::MarkNonDifferentiable(const std::vector<std::shared_ptr<Tensor>> &outputs) {
    non_differentiable_.clear();
    non_differentiable_.reserve(outputs.size());
    for (const auto &output : outputs) {
        if (output) {
            non_differentiable_.push_back(output.get());
        }
    }
}

void FunctionCtx::set_needs_input_grad(std::vector<bool> needs_input_grad) {
    needs_input_grad_ = std::move(needs_input_grad);
}

void FunctionCtx::SaveVariables(const std::vector<std::shared_ptr<Tensor>> &outputs) {
    saved_tensor_entries_.clear();
    saved_tensor_entries_.reserve(to_save_.size());
    for (const auto &tensor : to_save_) {
        SavedTensorEntry entry;
        if (!tensor) {
            saved_tensor_entries_.push_back(std::move(entry));
            continue;
        }

        bool is_output = false;
        for (const auto &output : outputs) {
            if (tensor.get() == output.get()) {
                is_output = true;
                break;
            }
        }
        auto tensor_to_save = is_output ? ShallowCopyWithoutAutogradMeta(tensor) : tensor;
        if (tls_saved_tensor_hooks.empty()) {
            entry.tensor = std::move(tensor_to_save);
        } else {
            const auto &hooks = tls_saved_tensor_hooks.back();
            entry.hook_state = hooks.pack(tensor_to_save);
            entry.unpack = hooks.unpack;
        }
        saved_tensor_entries_.push_back(std::move(entry));
    }
    to_save_.clear();
}

void FunctionCtx::ReleaseVariables() {
    to_save_.clear();
    saved_tensor_entries_.clear();
    needs_input_grad_.clear();
    non_differentiable_.clear();
}

bool FunctionCtx::IsNonDifferentiable(const std::shared_ptr<Tensor> &output) const {
    if (!output) {
        return false;
    }
    for (const auto *non_differentiable : non_differentiable_) {
        if (output.get() == non_differentiable) {
            return true;
        }
    }
    return false;
}

Function::Function() : ctx_(), type_(kUndefinedType) {}

Function::Function(const std::string &type) : ctx_(), type_(type) {}

Function::~Function() = default;

void Function::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                            const std::vector<std::shared_ptr<Tensor>> &) {}

const std::string &Function::type() const { return type_; }

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

    // Populate needs_input_grad before Forward/SetupContext so that
    // SetupContext can use it for saved-tensor pruning.
    // Must be done before NoGradGuard since it checks GradMode.
    ctx_.ReleaseVariables();
    if (autograd::GradMode::IsEnabled()) {
        std::vector<bool> needs_input_grad(input_tensors.size());
        for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
            needs_input_grad[idx] = input_tensors[idx]->requires_grad();
        }
        ctx_.set_needs_input_grad(std::move(needs_input_grad));
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

    ctx_.SaveVariables(output_tensors);

    // Call forward post-hooks
    for (const auto &hook : forward_post_hooks_) {
        if (hook) {
            hook(this, input_tensors, output_tensors);
        }
    }

    if (!autograd::GradMode::IsEnabled()) {
        // with no_grad: block graph building operations
        return output_tensors;
    }

    bool output_requires_grad = false;
    for (const auto &input_tensor : input_tensors) { output_requires_grad |= input_tensor->requires_grad(); }

    grad_outputs_reached_ = 0;
    grad_outputs_.resize(output_tensors.size(), nullptr);
    std::vector<bool> differentiable_outputs(output_tensors.size(), false);
    bool has_differentiable_output = false;
    for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
        differentiable_outputs[output_idx]
            = output_requires_grad && !ctx_.IsNonDifferentiable(output_tensors[output_idx]);
        has_differentiable_output |= differentiable_outputs[output_idx];
        if (!differentiable_outputs[output_idx]) {
            ++grad_outputs_reached_;
        }
    }

    if (!has_differentiable_output) {
        next_functions_.clear();
        return output_tensors;
    }

    for (int idx = 0; idx < input_tensors.size(); ++idx) {
        const auto &input_tensor = input_tensors[idx];
        if (input_tensor->requires_grad() && input_tensor->is_leaf()) {
            next_functions_.emplace_back(input_tensor->grad_accumulator(), input_tensor->output_idx());
            input_tensor->grad_accumulator()->IncreaseDependenciesNumber();
        } else {
            next_functions_.emplace_back(input_tensor->grad_fn(), input_tensor->output_idx());
            if (input_tensor->grad_fn()) {
                input_tensor->grad_fn()->IncreaseDependenciesNumber();
            }
        }
    }

    for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
        auto &output_tensor = output_tensors[output_idx];
        const bool differentiable_output = differentiable_outputs[output_idx];
        output_tensor->set_requires_grad(differentiable_output);
        output_tensor->set_grad_fn(differentiable_output ? shared_from_this() : nullptr);
        output_tensor->set_is_leaf(!differentiable_output);
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

        ctx_.ReleaseVariables();
        grad_outputs_.clear();
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

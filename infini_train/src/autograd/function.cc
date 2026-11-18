#include "infini_train/include/autograd/function.h"

#include <algorithm>
#include <ostream>
#include <unordered_map>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/accumulate.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/common/hook.h"  // IWYU pragma: keep
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/precision_check_config.h"
#include "infini_train/include/utils/precision_checker.h"

namespace infini_train::autograd {

namespace {
thread_local std::vector<FunctionCtx::SavedTensorHooks> tls_saved_tensor_hooks;

}  // namespace

FunctionCtx::SavedTensorHooksGuard::SavedTensorHooksGuard(
    SavedTensorHooks hooks) {
  CHECK(hooks.pack) << "Saved tensor pack hook must be set";
  CHECK(hooks.unpack) << "Saved tensor unpack hook must be set";
  tls_saved_tensor_hooks.push_back(std::move(hooks));
  depth_ = tls_saved_tensor_hooks.size();
}

FunctionCtx::SavedTensorHooksGuard::~SavedTensorHooksGuard() {
  if (tls_saved_tensor_hooks.empty()) {
    return;
  }
  CHECK_EQ(tls_saved_tensor_hooks.size(), depth_)
      << "SavedTensorHooksGuard destroyed out of order";
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

const std::vector<bool> &FunctionCtx::needs_input_grad() const {
  return needs_input_grad_;
}

void FunctionCtx::SaveForBackward(
    const std::vector<std::shared_ptr<Tensor>> &tensors) {
  to_save_ = tensors;
}

const AutocastContext &FunctionCtx::forward_autocast_context() const {
  CHECK(forward_autocast_context_.has_value())
      << "Forward autocast context has not been saved";
  return *forward_autocast_context_;
}

void FunctionCtx::MarkNonDifferentiable(
    const std::vector<std::shared_ptr<Tensor>> &outputs) {
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

void FunctionCtx::set_forward_autocast_context(const AutocastContext &context) {
  forward_autocast_context_ = context;
}

void FunctionCtx::SaveVariables(
    const std::vector<std::shared_ptr<Tensor>> &outputs) {
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
    auto tensor_to_save = is_output ? tensor->Detach() : tensor;
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
  forward_autocast_context_.reset();
}

bool FunctionCtx::IsNonDifferentiable(
    const std::shared_ptr<Tensor> &output) const {
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

std::vector<std::shared_ptr<Tensor>> Function::Apply(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK_GE(input_tensors.size(), 1);
  auto device = input_tensors[0]->GetDevice();
  core::DeviceGuard guard(device);

  // Register precision check hooks if enabled (before forward)
  if (!precision_check_registered_) {
    auto precision_level =
        utils::PrecisionCheckEnv::Instance().GetConfig().level;
    if (precision_level == utils::PrecisionCheckLevel::kFunction) {
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
  // Non-reentrant checkpoint recomputes under no-grad; PropagateRequiresGrad
  // keeps the same context metadata without wiring the recompute graph.
  ctx_.ReleaseVariables();
  const bool grad_enabled = autograd::GradMode::IsEnabled();
  const bool record_context =
      grad_enabled || autograd::GradMode::PropagateRequiresGrad();
  if (record_context) {
    std::vector<bool> needs_input_grad(input_tensors.size());
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      needs_input_grad[idx] =
          input_tensors[idx] && input_tensors[idx]->requires_grad();
    }
    ctx_.set_needs_input_grad(std::move(needs_input_grad));
  }

  // Apply autocast once at the autograd boundary so Forward / SetupContext
  // receive tensors already in the compute dtype. The shared_ptr copies are
  // local; we keep the caller's `input_tensors` untouched so next_functions_
  // wires up to the original autograd graph (leaf -> AccumulateGrad / non-leaf
  // -> grad_fn). Also, save the autocast context in FunctionCtx so that custom
  // backward can explicitly restore the forward autocast context from
  // FunctionCtx if needed.
  ctx_.set_forward_autocast_context(GetCurrentAutocastContext());
  auto compute_inputs = input_tensors;
  for (auto &t : compute_inputs) {
    tls_autocast_context.Autocast(type_, t);
  }

  std::vector<std::shared_ptr<Tensor>> output_tensors;
  {
    autograd::NoGradGuard no_grad;
    // Nested Functions inside Forward/SetupContext must not record context,
    // including during checkpoint recomputation. Restore the caller mode
    // before saving this Function's tensors and propagating output metadata.
    autograd::PropagateRequiresGradGuard no_propagate(false);
    // no_grad in autograd.Function.Forward()
    output_tensors = Forward(compute_inputs);
    SetupContext(compute_inputs, output_tensors);
  }

  if (record_context) {
    ctx_.SaveVariables(output_tensors);
  } else {
    ctx_.ReleaseVariables();
  }

  // Call forward post-hooks
  for (const auto &hook : forward_post_hooks_) {
    if (hook) {
      hook(this, input_tensors, output_tensors);
    }
  }

  if (!record_context) {
    return output_tensors;
  }

  bool output_requires_grad = false;
  for (const auto &input_tensor : input_tensors) {
    output_requires_grad |= input_tensor && input_tensor->requires_grad();
  }

  std::vector<bool> differentiable_outputs(output_tensors.size(), false);
  bool has_differentiable_output = false;
  for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
    differentiable_outputs[output_idx] =
        output_tensors[output_idx] && output_requires_grad &&
        !ctx_.IsNonDifferentiable(output_tensors[output_idx]);
    has_differentiable_output |= differentiable_outputs[output_idx];
  }

  // Metadata-only recomputation must not change the recorded graph.
  if (grad_enabled) {
    num_outputs_ = output_tensors.size();
    output_metadata_.clear();
    if (output_tensors.size() > 1) {
      for (size_t idx = 0; idx < output_tensors.size(); ++idx) {
        if (differentiable_outputs[idx]) {
          const auto &output = output_tensors[idx];
          output_metadata_.push_back(OutputMetadata{
              output->Dims(), output->Dtype(), output->GetDevice()});
        } else {
          output_metadata_.push_back(std::nullopt);
        }
      }
    }

    if (!has_differentiable_output) {
      next_functions_.clear();
    } else {
      for (int idx = 0; idx < input_tensors.size(); ++idx) {
        const auto &input_tensor = input_tensors[idx];
        if (!input_tensor) {
          next_functions_.emplace_back(nullptr, 0);
          continue;
        }
        if (input_tensor->requires_grad() && input_tensor->is_leaf()) {
          next_functions_.emplace_back(input_tensor->grad_accumulator(),
                                       input_tensor->output_idx());
        } else {
          next_functions_.emplace_back(input_tensor->grad_fn(),
                                       input_tensor->output_idx());
        }
      }
    }
  }

  // Both modes take MarkNonDifferentiable into consideration.
  // Set output metadata only after wiring input edges, since a custom Function
  // may return an input tensor.
  for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
    auto &output_tensor = output_tensors[output_idx];
    if (!output_tensor) {
      continue;
    }
    const bool differentiable_output = differentiable_outputs[output_idx];
    const bool attach_grad_fn = grad_enabled && differentiable_output;
    output_tensor->set_requires_grad(differentiable_output);
    output_tensor->set_grad_fn(attach_grad_fn ? shared_from_this() : nullptr);
    output_tensor->set_is_leaf(!attach_grad_fn);
    output_tensor->set_output_idx(output_idx);
  }

  return output_tensors;
}

struct Function::BackwardContext {
  struct State {
    int dependencies = 0;

    int reached = 0;

    std::vector<std::shared_ptr<Tensor>> grad_outputs;
  };

  std::unordered_map<Function *, State> states;
};

void Function::BackwardPartial(std::shared_ptr<Tensor> grad_output,
                               int grad_output_idx) {
  // Each backward call owns its state, including checkpoint recomputation.
  BackwardContext context;
  context.states[this].grad_outputs.resize(num_outputs_);
  std::vector<Function *> functions{this};
  for (size_t idx = 0; idx < functions.size(); ++idx) {
    for (const auto &[next, output_idx] : functions[idx]->next_functions_) {
      if (!next) {
        continue;
      }
      auto [entry, inserted] = context.states.try_emplace(next.get());
      ++entry->second.dependencies;
      if (inserted) {
        entry->second.grad_outputs.resize(next->num_outputs_);
        functions.push_back(next.get());
      }
    }
  }
  BackwardPartial(std::move(grad_output), grad_output_idx, context);
}

void Function::BackwardPartial(std::shared_ptr<Tensor> grad_output,
                               int grad_output_idx, BackwardContext &context) {
  auto &state = context.states.at(this);
  auto &grad_outputs = state.grad_outputs;
  if (grad_output) {
    core::DeviceGuard guard(grad_output->GetDevice());
    if (!grad_outputs.at(grad_output_idx)) {
      grad_outputs.at(grad_output_idx) = std::move(grad_output);
    } else {
      auto kernel = Dispatcher::Instance().GetKernel(
          {grad_output->GetDevice().type(), "AccumulateGrad"});
      kernel.Call<void>(grad_output, 1.0f, grad_outputs.at(grad_output_idx));
    }
  }
  ++state.reached;
  if (state.reached == state.dependencies || state.dependencies == 0) {
    std::vector<std::shared_ptr<Tensor>> grad_inputs(next_functions_.size());
    auto first_gradient =
        std::find_if(grad_outputs.begin(), grad_outputs.end(),
                     [](const auto &gradient) { return gradient != nullptr; });
    if (first_gradient != grad_outputs.end()) {
      core::DeviceGuard guard((*first_gradient)->GetDevice());
      for (size_t idx = 0; idx < output_metadata_.size(); ++idx) {
        if (!grad_outputs[idx] && output_metadata_[idx]) {
          const auto &metadata = *output_metadata_[idx];
          grad_outputs[idx] = std::make_shared<Tensor>(
              metadata.dims, metadata.dtype, metadata.device);
          grad_outputs[idx]->Fill(0.0f);
        }
      }
      // Call backward pre-hooks
      for (const auto &hook : backward_pre_hooks_) {
        if (hook) {
          hook(this, grad_outputs);
        }
      }

      {
        autograd::NoGradGuard no_grad;
        // no_grad in autograd.Function.Backward()
        grad_inputs = Backward(grad_outputs);
      }

      // Call backward post-hooks
      for (const auto &hook : backward_post_hooks_) {
        if (hook) {
          hook(this, grad_inputs, grad_outputs);
        }
      }
    } else if (dynamic_cast<AccumulateGrad *>(this)) {
      grad_inputs = Backward(grad_outputs);
    }

    ctx_.ReleaseVariables();
    grad_outputs.clear();

    CHECK_EQ(grad_inputs.size(), next_functions_.size());
    auto propagate_grad_input = [&](size_t idx) {
      auto grad_input = std::move(grad_inputs[idx]);
      auto &[next_function, output_idx] = next_functions_[idx];
      if (next_function) {
        next_function->BackwardPartial(std::move(grad_input), output_idx,
                                       context);
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
      if (next_function &&
          std::dynamic_pointer_cast<AccumulateGrad>(next_function)) {
        propagate_grad_input(idx);
      }
    }
    for (size_t idx = 0; idx < grad_inputs.size(); ++idx) {
      const auto &next_function = next_functions_[idx].first;
      if (next_function &&
          !std::dynamic_pointer_cast<AccumulateGrad>(next_function)) {
        propagate_grad_input(idx);
      }
    }
    next_functions_.clear();
  }
}

std::shared_ptr<infini_train::HookHandle> Function::RegisterForwardPreHook(
    FunctionPreHook hook) {
  forward_pre_hooks_.push_back(std::move(hook));
  return std::make_shared<FunctionHookHandleImpl<FunctionPreHook>>(
      &forward_pre_hooks_, forward_pre_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Function::RegisterForwardPostHook(
    FunctionPostHook hook) {
  forward_post_hooks_.push_back(std::move(hook));
  return std::make_shared<FunctionHookHandleImpl<FunctionPostHook>>(
      &forward_post_hooks_, forward_post_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Function::RegisterBackwardPreHook(
    FunctionPreHook hook) {
  backward_pre_hooks_.push_back(std::move(hook));
  return std::make_shared<FunctionHookHandleImpl<FunctionPreHook>>(
      &backward_pre_hooks_, backward_pre_hooks_.size() - 1);
}

std::shared_ptr<infini_train::HookHandle> Function::RegisterBackwardPostHook(
    FunctionPostHook hook) {
  backward_post_hooks_.push_back(std::move(hook));
  return std::make_shared<FunctionHookHandleImpl<FunctionPostHook>>(
      &backward_post_hooks_, backward_post_hooks_.size() - 1);
}
}  // namespace infini_train::autograd

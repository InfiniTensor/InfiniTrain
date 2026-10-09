#pragma once
#include <atomic>

namespace infini_train::autograd {

class GradMode {
public:
    // Whether to enable Autograd (enabled by default)
    static bool IsEnabled() { return grad_enabled_; }
    static void SetEnabled(bool enabled) { grad_enabled_ = enabled; }
    static bool PropagateRequiresGrad() { return propagate_requires_grad_; }
    static void SetPropagateRequiresGrad(bool enabled) { propagate_requires_grad_ = enabled; }

private:
    // grad mode should be thread_local
    static thread_local bool grad_enabled_;
    static thread_local bool propagate_requires_grad_;
};

// RAII: Disable grad (align with torch.no_grad)
class NoGradGuard {
public:
    NoGradGuard() : prev_(GradMode::IsEnabled()) { GradMode::SetEnabled(false); }
    ~NoGradGuard() { GradMode::SetEnabled(prev_); }

private:
    bool prev_;
};

// RAII: Enable grad (align with torch.enable_grad)
class EnableGradGuard {
public:
    EnableGradGuard() : prev_(GradMode::IsEnabled()) { GradMode::SetEnabled(true); }
    ~EnableGradGuard() { GradMode::SetEnabled(prev_); }

private:
    bool prev_;
};

// RAII: Propagate requires_grad metadata while graph construction is disabled.
// Used by non-reentrant checkpoint recomputation so downstream SetupContext
// calls see the same needs_input_grad_ pattern as the original forward,
// without wiring the recompute graph into the engine.
class PropagateRequiresGradGuard {
public:
    PropagateRequiresGradGuard() : prev_(GradMode::PropagateRequiresGrad()) {
        GradMode::SetPropagateRequiresGrad(true);
    }
    ~PropagateRequiresGradGuard() { GradMode::SetPropagateRequiresGrad(prev_); }

private:
    bool prev_;
};

} // namespace infini_train::autograd

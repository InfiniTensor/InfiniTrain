#pragma once

namespace infini_train::autograd {

class GradMode {
public:
    // Whether to enable Autograd (enabled by default)
    static bool IsEnabled();
    static void SetEnabled(bool enabled);
    static bool PropagateRequiresGrad();
    static void SetPropagateRequiresGrad(bool enabled);

private:
    // grad mode should be thread_local
    static thread_local bool grad_enabled_;
    static thread_local bool propagate_requires_grad_;
};

// RAII: Disable grad (align with torch.no_grad)
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

private:
    bool prev_;
};

// RAII: Enable grad (align with torch.enable_grad)
class EnableGradGuard {
public:
    EnableGradGuard();
    ~EnableGradGuard();

private:
    bool prev_;
};

// RAII: Propagate requires_grad metadata while graph construction is disabled.
// Used by non-reentrant checkpoint recomputation so downstream SetupContext
// calls see the same needs_input_grad_ pattern as the original forward,
// without wiring the recompute graph into the engine. Together with grad mode,
// this gives three behaviors: build a graph, propagate metadata only, or neither.
class PropagateRequiresGradGuard {
public:
    explicit PropagateRequiresGradGuard(bool enabled = true);
    ~PropagateRequiresGradGuard();

private:
    bool prev_ = false;
};

} // namespace infini_train::autograd

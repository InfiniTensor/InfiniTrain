#pragma once
#include <atomic>

namespace infini_train::autograd {

class GradMode {
public:
    // Whether to enable Autograd (enabled by default)
    static bool IsEnabled() { return grad_enabled_; }
    static void SetEnabled(bool enabled) { grad_enabled_ = enabled; }

private:
    // grad mode should be thread_local
    static thread_local bool grad_enabled_;
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

} // namespace infini_train::autograd

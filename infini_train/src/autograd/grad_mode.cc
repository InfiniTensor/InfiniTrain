#include "infini_train/include/autograd/grad_mode.h"

namespace infini_train::autograd {
thread_local bool GradMode::grad_enabled_ = true;
thread_local bool GradMode::propagate_requires_grad_ = false;

bool GradMode::IsEnabled() { return grad_enabled_; }

void GradMode::SetEnabled(bool enabled) { grad_enabled_ = enabled; }

bool GradMode::PropagateRequiresGrad() { return propagate_requires_grad_; }

void GradMode::SetPropagateRequiresGrad(bool enabled) {
  propagate_requires_grad_ = enabled;
}

NoGradGuard::NoGradGuard() : prev_(GradMode::IsEnabled()) {
  GradMode::SetEnabled(false);
}

NoGradGuard::~NoGradGuard() { GradMode::SetEnabled(prev_); }

EnableGradGuard::EnableGradGuard() : prev_(GradMode::IsEnabled()) {
  GradMode::SetEnabled(true);
}

EnableGradGuard::~EnableGradGuard() { GradMode::SetEnabled(prev_); }

PropagateRequiresGradGuard::PropagateRequiresGradGuard(bool enabled)
    : prev_(GradMode::PropagateRequiresGrad()) {
  GradMode::SetPropagateRequiresGrad(enabled);
}

PropagateRequiresGradGuard::~PropagateRequiresGradGuard() {
  GradMode::SetPropagateRequiresGrad(prev_);
}
}  // namespace infini_train::autograd

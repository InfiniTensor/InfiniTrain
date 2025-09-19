#include "infini_train/include/autograd/grad_mode.h"

namespace infini_train::autograd {
thread_local bool GradMode::grad_enabled_ = true;
} // namespace infini_train::autograd

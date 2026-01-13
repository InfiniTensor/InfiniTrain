#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function_hook.h"

namespace infini_train {
class Tensor;

namespace autograd {

// Tensor backward hook: modifies gradient during backward pass
// Returns modified gradient or nullptr to keep original
using TensorBackwardHook = std::function<std::shared_ptr<Tensor>(const std::shared_ptr<Tensor>&)>;

class TensorBackwardHookHandle : public HookHandle {
public:
    TensorBackwardHookHandle(std::vector<TensorBackwardHook>* hooks, size_t id)
        : hooks_(hooks), id_(id) {}

    void Remove() override;

private:
    std::vector<TensorBackwardHook>* hooks_;
    size_t id_;
    bool removed_ = false;
};

} // namespace autograd
} // namespace infini_train

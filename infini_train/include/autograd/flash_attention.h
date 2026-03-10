#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::autograd {

class FlashAttention : public Function {
public:
    static constexpr char kType[] = "FlashAttentionFunction";

    explicit FlashAttention(bool is_causal = false) : Function(kType), is_causal_(is_causal) {}

    std::vector<std::shared_ptr<Tensor>> Forward(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(
        const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    bool is_causal_;
    // Temporary storage to pass L from Forward() to SetupContext()
    std::shared_ptr<Tensor> l_cache_;
};

} // namespace infini_train::autograd

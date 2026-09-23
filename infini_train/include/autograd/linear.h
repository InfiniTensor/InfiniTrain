#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

// Raw dispatcher operations for custom autograd Functions. These calls do not create autograd nodes.
namespace linear {
std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                const std::shared_ptr<Tensor> &bias = nullptr);
std::shared_ptr<Tensor> BackwardInput(const std::shared_ptr<Tensor> &weight, const std::shared_ptr<Tensor> &grad_output,
                                      const std::vector<int64_t> &input_dims);
std::shared_ptr<Tensor> BackwardWeight(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &grad_output,
                                       int64_t in_features, int64_t out_features);
std::shared_ptr<Tensor> BackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_features);
} // namespace linear

class Linear : public Function {
public:
    static constexpr char kType[] = "LinearFunction";

    Linear() : Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    bool bias_ = false;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    std::vector<int64_t> input_dims_;
};
} // namespace infini_train::autograd

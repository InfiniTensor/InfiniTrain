#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

class TopK : public Function {
public:
    static constexpr char kType[] = "TopKFunction";

    explicit TopK(int64_t topk, int64_t dim = -1, bool largest = true, bool sorted = true)
        : Function(kType), topk_(topk), dim_(dim), largest_(largest), sorted_(sorted) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    int64_t topk_ = 1;
    int64_t dim_ = -1;
    bool largest_ = true;
    bool sorted_ = true;
    std::vector<int64_t> input_dims_;
};

} // namespace infini_train::autograd

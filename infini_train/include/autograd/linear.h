#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

struct LinearMeta {
    bool transpose = false;
    bool has_bias = false;
    int64_t in_features = 0;
    int64_t out_features = 0;
    std::vector<int64_t> input_dims;
};

struct LinearGradFlags {
    bool input = false;
    bool weight = false;
    bool bias = false;
};

class Linear : public Function {
public:
    static constexpr char kType[] = "LinearFunction";

    Linear() : Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    LinearMeta meta_;
};
} // namespace infini_train::autograd

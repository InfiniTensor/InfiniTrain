#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {

class Conv2d : public Function {
 public:
  static constexpr char kType[] = "Conv2dFunction";

  Conv2d(std::pair<int64_t, int64_t> stride,
         std::pair<int64_t, int64_t> padding,
         std::pair<int64_t, int64_t> dilation, int64_t groups);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
  void SetupContext(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors,
      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

 private:
  std::pair<int64_t, int64_t> stride_{1, 1};
  std::pair<int64_t, int64_t> padding_{0, 0};
  std::pair<int64_t, int64_t> dilation_{1, 1};
  int64_t groups_ = 1;
  bool has_bias_ = false;
  std::vector<int64_t> input_dims_;
  std::vector<int64_t> weight_dims_;
};

}  // namespace infini_train::autograd

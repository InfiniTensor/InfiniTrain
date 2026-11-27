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

class MaxPool2d : public Function {
 public:
  static constexpr char kType[] = "MaxPool2dFunction";

  MaxPool2d(std::pair<int64_t, int64_t> kernel_size,
            std::pair<int64_t, int64_t> stride,
            std::pair<int64_t, int64_t> padding);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
  void SetupContext(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors,
      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

 private:
  std::pair<int64_t, int64_t> kernel_size_;
  std::pair<int64_t, int64_t> stride_;
  std::pair<int64_t, int64_t> padding_;
  std::vector<int64_t> input_dims_;
};

class AdaptiveAvgPool2d : public Function {
 public:
  static constexpr char kType[] = "AdaptiveAvgPool2dFunction";

  explicit AdaptiveAvgPool2d(std::pair<int64_t, int64_t> output_size);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
  void SetupContext(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors,
      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

 private:
  std::pair<int64_t, int64_t> output_size_;
  std::vector<int64_t> input_dims_;
};

}  // namespace infini_train::autograd

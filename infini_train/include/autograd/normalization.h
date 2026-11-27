#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {
class LayerNorm : public Function {
 public:
  static constexpr char kType[] = "LayerNormFunction";

  explicit LayerNorm(float eps) : Function(kType), eps_(eps) {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
  void SetupContext(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors,
      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

 private:
  const float eps_ = 1e-5f;
};

class BatchNorm2d : public Function {
 public:
  static constexpr char kType[] = "BatchNorm2dFunction";

  BatchNorm2d(std::shared_ptr<Tensor> running_mean,
              std::shared_ptr<Tensor> running_var, bool training,
              float momentum, float eps);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
  void SetupContext(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors,
      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

 private:
  std::shared_ptr<Tensor> running_mean_;
  std::shared_ptr<Tensor> running_var_;
  bool training_ = true;
  float momentum_ = 0.1f;
  float eps_ = 1e-5f;
};
}  // namespace infini_train::autograd

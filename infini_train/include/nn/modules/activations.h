#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Sigmoid : public CloneableModule<Sigmoid> {
 public:
  static constexpr char kType[] = "Sigmoid";
  Sigmoid() : CloneableModule(kType) {}
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class NewGELU : public CloneableModule<NewGELU> {
 public:
  static constexpr char kType[] = "NewGELU";
  NewGELU() : CloneableModule(kType) {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &x) override;
};

class SwiGLU : public CloneableModule<SwiGLU> {
 public:
  static constexpr char kType[] = "SwiGLU";
  SwiGLU() : CloneableModule(kType) {}

  // The last input dimension is packed as [gate, up], matching Megatron-LM.
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &x) override;
};

class ReLU : public CloneableModule<ReLU> {
 public:
  static constexpr char kType[] = "ReLU";
  ReLU() : CloneableModule(kType) {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ReLU6 : public CloneableModule<ReLU6> {
 public:
  static constexpr char kType[] = "ReLU6";
  ReLU6() : CloneableModule(kType) {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
}  // namespace infini_train::nn

#pragma once

#include <stdint.h>

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}  // namespace infini_train

namespace infini_train::nn {
class LayerNorm : public CloneableModule<LayerNorm> {
 public:
  static constexpr char kType[] = "LayerNorm";
  static constexpr char kParamWeightName[] = "weight";
  static constexpr char kParamBiasName[] = "bias";

  LayerNorm(const std::vector<int64_t> &normalized_shape, float eps = 1e-5f,
            Device device = Device());
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

 private:
  void ResetParameters();

  const float eps_ = 1e-5f;
};

class RMSNorm : public CloneableModule<RMSNorm> {
 public:
  static constexpr char kType[] = "RMSNorm";
  static constexpr char kParamWeightName[] = "weight";

  explicit RMSNorm(int64_t dim, float eps = 1e-6f, Device device = Device());

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &x) override;

 private:
  float eps_ = 1e-5f;
};

class BatchNorm2d : public CloneableModule<BatchNorm2d> {
 public:
  static constexpr char kType[] = "BatchNorm2d";
  static constexpr char kParamWeightName[] = "weight";
  static constexpr char kParamBiasName[] = "bias";
  static constexpr char kBufferRunningMeanName[] = "running_mean";
  static constexpr char kBufferRunningVarName[] = "running_var";

  explicit BatchNorm2d(int64_t num_features, float eps = 1e-5f,
                       float momentum = 0.1f, Device device = Device());

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

 private:
  void ResetParameters();

  int64_t num_features_ = 0;
  float eps_ = 1e-5f;
  float momentum_ = 0.1f;
};
}  // namespace infini_train::nn

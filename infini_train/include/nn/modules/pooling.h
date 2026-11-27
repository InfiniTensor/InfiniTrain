#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {

class MaxPool2d : public CloneableModule<MaxPool2d> {
 public:
  static constexpr char kType[] = "MaxPool2d";

  MaxPool2d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0);
  MaxPool2d(std::pair<int64_t, int64_t> kernel_size,
            std::pair<int64_t, int64_t> stride,
            std::pair<int64_t, int64_t> padding = {0, 0});

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

 private:
  std::pair<int64_t, int64_t> kernel_size_;
  std::pair<int64_t, int64_t> stride_;
  std::pair<int64_t, int64_t> padding_;
};

class AdaptiveAvgPool2d : public CloneableModule<AdaptiveAvgPool2d> {
 public:
  static constexpr char kType[] = "AdaptiveAvgPool2d";

  explicit AdaptiveAvgPool2d(int64_t output_size);
  explicit AdaptiveAvgPool2d(std::pair<int64_t, int64_t> output_size);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

 private:
  std::pair<int64_t, int64_t> output_size_;
};

}  // namespace infini_train::nn

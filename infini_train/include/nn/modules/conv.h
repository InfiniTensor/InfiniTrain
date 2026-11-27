#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {

class Conv2d : public CloneableModule<Conv2d> {
 public:
  static constexpr char kType[] = "Conv2d";
  static constexpr char kParamWeightName[] = "weight";
  static constexpr char kParamBiasName[] = "bias";

  Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
         int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
         int64_t groups = 1, bool bias = true, Device device = Device());
  Conv2d(int64_t in_channels, int64_t out_channels,
         std::pair<int64_t, int64_t> kernel_size,
         std::pair<int64_t, int64_t> stride = {1, 1},
         std::pair<int64_t, int64_t> padding = {0, 0},
         std::pair<int64_t, int64_t> dilation = {1, 1}, int64_t groups = 1,
         bool bias = true, Device device = Device());

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

  bool has_bias() const;

 private:
  void ResetParameters();

  int64_t in_channels_ = 0;
  int64_t out_channels_ = 0;
  std::pair<int64_t, int64_t> kernel_size_{1, 1};
  std::pair<int64_t, int64_t> stride_{1, 1};
  std::pair<int64_t, int64_t> padding_{0, 0};
  std::pair<int64_t, int64_t> dilation_{1, 1};
  int64_t groups_ = 1;
  bool bias_ = true;
};

}  // namespace infini_train::nn

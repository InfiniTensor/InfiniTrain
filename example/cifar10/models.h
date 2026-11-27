#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::example::cifar10 {

class BasicBlock : public infini_train::nn::CloneableModule<BasicBlock> {
 public:
  static constexpr char kType[] = "BasicBlock";

  BasicBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1,
             infini_train::Device device = infini_train::Device());

  std::vector<std::shared_ptr<infini_train::Tensor>> Forward(
      const std::vector<std::shared_ptr<infini_train::Tensor>> &input_tensors)
      override;

 private:
  bool has_downsample_ = false;
};

class ResNet18 : public infini_train::nn::CloneableModule<ResNet18> {
 public:
  static constexpr char kType[] = "ResNet18";

  explicit ResNet18(int64_t num_classes = 10,
                    infini_train::Device device = infini_train::Device());

  std::vector<std::shared_ptr<infini_train::Tensor>> Forward(
      const std::vector<std::shared_ptr<infini_train::Tensor>> &input_tensors)
      override;

 private:
  std::shared_ptr<infini_train::nn::Module> MakeLayer(
      int64_t in_channels, int64_t out_channels, int64_t num_blocks,
      int64_t stride, infini_train::Device device);
  void InitWeights();
};

class ConvBNReLU6 : public infini_train::nn::CloneableModule<ConvBNReLU6> {
 public:
  static constexpr char kType[] = "ConvBNReLU6";

  ConvBNReLU6(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
              int64_t stride, int64_t groups = 1,
              infini_train::Device device = infini_train::Device());

  std::vector<std::shared_ptr<infini_train::Tensor>> Forward(
      const std::vector<std::shared_ptr<infini_train::Tensor>> &input_tensors)
      override;
};

class InvertedResidual
    : public infini_train::nn::CloneableModule<InvertedResidual> {
 public:
  static constexpr char kType[] = "InvertedResidual";

  InvertedResidual(int64_t in_channels, int64_t out_channels, int64_t stride,
                   int64_t expand_ratio,
                   infini_train::Device device = infini_train::Device());

  std::vector<std::shared_ptr<infini_train::Tensor>> Forward(
      const std::vector<std::shared_ptr<infini_train::Tensor>> &input_tensors)
      override;

 private:
  bool use_res_connect_ = false;
};

class MobileNetV2 : public infini_train::nn::CloneableModule<MobileNetV2> {
 public:
  static constexpr char kType[] = "MobileNetV2";

  explicit MobileNetV2(int64_t num_classes = 10, float width_mult = 1.0f,
                       infini_train::Device device = infini_train::Device());

  std::vector<std::shared_ptr<infini_train::Tensor>> Forward(
      const std::vector<std::shared_ptr<infini_train::Tensor>> &input_tensors)
      override;

 private:
  void InitWeights();
};

}  // namespace infini_train::example::cifar10

#include "example/cifar10/models.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/conv.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/pooling.h"
#include "infini_train/include/tensor.h"

namespace infini_train::example::cifar10 {
namespace {

using namespace infini_train;

int64_t MakeDivisible(float v, int64_t divisor = 8) {
  int64_t new_v = std::max<int64_t>(
      divisor,
      static_cast<int64_t>(std::floor((v + divisor / 2.0f) / divisor)) *
          divisor);
  if (new_v < static_cast<int64_t>(0.9f * v)) {
    new_v += divisor;
  }
  return new_v;
}

}  // namespace

BasicBlock::BasicBlock(int64_t in_channels, int64_t out_channels,
                       int64_t stride, Device device)
    : CloneableModule(kType) {
  device_ = device;
  modules_["conv1"] = std::make_shared<nn::Conv2d>(
      in_channels, out_channels, 3, stride, 1, 1, 1, /*bias=*/false, device);
  modules_["bn1"] =
      std::make_shared<nn::BatchNorm2d>(out_channels, 1e-5f, 0.1f, device);
  modules_["relu1"] = std::make_shared<nn::ReLU>();
  modules_["conv2"] = std::make_shared<nn::Conv2d>(
      out_channels, out_channels, 3, 1, 1, 1, 1, /*bias=*/false, device);
  modules_["bn2"] =
      std::make_shared<nn::BatchNorm2d>(out_channels, 1e-5f, 0.1f, device);
  modules_["relu2"] = std::make_shared<nn::ReLU>();

  if (stride != 1 || in_channels != out_channels) {
    has_downsample_ = true;
    std::vector<std::shared_ptr<nn::Module>> ds_layers;
    ds_layers.push_back(std::make_shared<nn::Conv2d>(
        in_channels, out_channels, 1, stride, 0, 1, 1, /*bias=*/false, device));
    ds_layers.push_back(
        std::make_shared<nn::BatchNorm2d>(out_channels, 1e-5f, 0.1f, device));
    modules_["downsample"] =
        std::make_shared<nn::Sequential>(std::move(ds_layers));
  }
}

std::vector<std::shared_ptr<Tensor>> BasicBlock::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK_EQ(input_tensors.size(), 1);
  auto identity = input_tensors[0];

  auto out = (*modules_["conv1"])({identity});
  out = (*modules_["bn1"])(out);
  out = (*modules_["relu1"])(out);
  out = (*modules_["conv2"])(out);
  out = (*modules_["bn2"])(out);

  if (has_downsample_) {
    identity = (*modules_["downsample"])({identity})[0];
  }
  auto residual = out[0] + identity;
  return (*modules_["relu2"])({residual});
}

ResNet18::ResNet18(int64_t num_classes, Device device)
    : CloneableModule(kType) {
  device_ = device;
  // CIFAR-10 ResNet-18 stem: 3x3 stride 1, no initial maxpool
  modules_["conv1"] = std::make_shared<nn::Conv2d>(3, 64, 3, 1, 1, 1, 1,
                                                   /*bias=*/false, device);
  modules_["bn1"] = std::make_shared<nn::BatchNorm2d>(64, 1e-5f, 0.1f, device);
  modules_["relu"] = std::make_shared<nn::ReLU>();

  modules_["layer1"] = MakeLayer(64, 64, 2, 1, device);
  modules_["layer2"] = MakeLayer(64, 128, 2, 2, device);
  modules_["layer3"] = MakeLayer(128, 256, 2, 2, device);
  modules_["layer4"] = MakeLayer(256, 512, 2, 2, device);

  modules_["avgpool"] = std::make_shared<nn::AdaptiveAvgPool2d>(1);
  modules_["flatten"] = std::make_shared<nn::Flatten>(1);
  modules_["fc"] = std::make_shared<nn::Linear>(512, num_classes, true, device);

  InitWeights();
}

std::shared_ptr<nn::Module> ResNet18::MakeLayer(int64_t in_channels,
                                                int64_t out_channels,
                                                int64_t num_blocks,
                                                int64_t stride, Device device) {
  std::vector<std::shared_ptr<nn::Module>> blocks;
  blocks.push_back(
      std::make_shared<BasicBlock>(in_channels, out_channels, stride, device));
  for (int64_t i = 1; i < num_blocks; ++i) {
    blocks.push_back(
        std::make_shared<BasicBlock>(out_channels, out_channels, 1, device));
  }
  return std::make_shared<nn::Sequential>(std::move(blocks));
}

void ResNet18::InitWeights() {
  Apply([](nn::Module *m) {
    if (m->type() == nn::Conv2d::kType) {
      nn::init::KaimingNormal(
          *m->mutable_parameter(nn::Conv2d::kParamWeightName), 0.0f,
          nn::init::KaimingMode::kFanOut, nn::init::NonLinearityType::kReLU);
    } else if (m->type() == nn::BatchNorm2d::kType) {
      nn::init::Ones(*m->mutable_parameter(nn::BatchNorm2d::kParamWeightName));
      nn::init::Zeros(*m->mutable_parameter(nn::BatchNorm2d::kParamBiasName));
    }
  });
}

std::vector<std::shared_ptr<Tensor>> ResNet18::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  auto x = (*modules_["conv1"])(input_tensors);
  x = (*modules_["bn1"])(x);
  x = (*modules_["relu"])(x);
  x = (*modules_["layer1"])(x);
  x = (*modules_["layer2"])(x);
  x = (*modules_["layer3"])(x);
  x = (*modules_["layer4"])(x);
  x = (*modules_["avgpool"])(x);
  x = (*modules_["flatten"])(x);
  return (*modules_["fc"])(x);
}

ConvBNReLU6::ConvBNReLU6(int64_t in_channels, int64_t out_channels,
                         int64_t kernel_size, int64_t stride, int64_t groups,
                         Device device)
    : CloneableModule(kType) {
  device_ = device;
  const int64_t padding = (kernel_size - 1) / 2;
  modules_["0"] = std::make_shared<nn::Conv2d>(in_channels, out_channels,
                                               kernel_size, stride, padding, 1,
                                               groups, /*bias=*/false, device);
  modules_["1"] =
      std::make_shared<nn::BatchNorm2d>(out_channels, 1e-5f, 0.1f, device);
  modules_["2"] = std::make_shared<nn::ReLU6>();
}

std::vector<std::shared_ptr<Tensor>> ConvBNReLU6::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  auto x = (*modules_["0"])(input_tensors);
  x = (*modules_["1"])(x);
  return (*modules_["2"])(x);
}

InvertedResidual::InvertedResidual(int64_t in_channels, int64_t out_channels,
                                   int64_t stride, int64_t expand_ratio,
                                   Device device)
    : CloneableModule(kType),
      use_res_connect_(stride == 1 && in_channels == out_channels) {
  device_ = device;
  const int64_t hidden_dim = in_channels * expand_ratio;
  std::vector<std::shared_ptr<nn::Module>> layers;
  if (expand_ratio != 1) {
    // 1x1 pointwise expansion
    layers.push_back(std::make_shared<ConvBNReLU6>(in_channels, hidden_dim, 1,
                                                   1, 1, device));
  }
  // 3x3 depthwise convolution
  layers.push_back(std::make_shared<ConvBNReLU6>(hidden_dim, hidden_dim, 3,
                                                 stride, hidden_dim, device));
  // 1x1 pointwise linear projection (no ReLU6)
  layers.push_back(std::make_shared<nn::Conv2d>(
      hidden_dim, out_channels, 1, 1, 0, 1, 1, /*bias=*/false, device));
  layers.push_back(
      std::make_shared<nn::BatchNorm2d>(out_channels, 1e-5f, 0.1f, device));
  modules_["conv"] = std::make_shared<nn::Sequential>(std::move(layers));
}

std::vector<std::shared_ptr<Tensor>> InvertedResidual::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK_EQ(input_tensors.size(), 1);
  auto x = input_tensors[0];
  auto out = (*modules_["conv"])({x})[0];
  if (use_res_connect_) {
    return {x + out};
  }
  return {out};
}

MobileNetV2::MobileNetV2(int64_t num_classes, float width_mult, Device device)
    : CloneableModule(kType) {
  device_ = device;
  struct StageCfg {
    int64_t t;
    int64_t c;
    int64_t n;
    int64_t s;
  };
  // CIFAR-10 adapted MobileNetV2 strides (stem s=1, stage2 s=1)
  const std::vector<StageCfg> cfgs = {
      {1, 16, 1, 1},   //
      {6, 24, 2, 1},   //
      {6, 32, 3, 2},   //
      {6, 64, 4, 2},   //
      {6, 96, 3, 1},   //
      {6, 160, 3, 2},  //
      {6, 320, 1, 1},  //
  };

  int64_t input_channel = MakeDivisible(32.0f * width_mult, 8);
  const int64_t last_channel =
      MakeDivisible(1280.0f * std::max(1.0f, width_mult), 8);

  std::vector<std::shared_ptr<nn::Module>> features;
  features.push_back(
      std::make_shared<ConvBNReLU6>(3, input_channel, 3, 1, 1, device));

  for (const auto &cfg : cfgs) {
    const int64_t output_channel = MakeDivisible(cfg.c * width_mult, 8);
    for (int64_t i = 0; i < cfg.n; ++i) {
      const int64_t stride = (i == 0) ? cfg.s : 1;
      features.push_back(std::make_shared<InvertedResidual>(
          input_channel, output_channel, stride, cfg.t, device));
      input_channel = output_channel;
    }
  }
  features.push_back(std::make_shared<ConvBNReLU6>(input_channel, last_channel,
                                                   1, 1, 1, device));

  modules_["features"] = std::make_shared<nn::Sequential>(std::move(features));
  modules_["avgpool"] = std::make_shared<nn::AdaptiveAvgPool2d>(1);
  modules_["flatten"] = std::make_shared<nn::Flatten>(1);
  modules_["classifier"] =
      std::make_shared<nn::Linear>(last_channel, num_classes, true, device);

  InitWeights();
}

void MobileNetV2::InitWeights() {
  Apply([](nn::Module *m) {
    if (m->type() == nn::Conv2d::kType) {
      nn::init::KaimingNormal(
          *m->mutable_parameter(nn::Conv2d::kParamWeightName), 0.0f,
          nn::init::KaimingMode::kFanOut, nn::init::NonLinearityType::kReLU);
    } else if (m->type() == nn::BatchNorm2d::kType) {
      nn::init::Ones(*m->mutable_parameter(nn::BatchNorm2d::kParamWeightName));
      nn::init::Zeros(*m->mutable_parameter(nn::BatchNorm2d::kParamBiasName));
    } else if (m->type() == nn::Linear::kType) {
      nn::init::Normal(*m->mutable_parameter(nn::Linear::kParamWeightName),
                       0.0f, 0.01f);
      if (m->has_parameter(nn::Linear::kParamBiasName)) {
        nn::init::Zeros(*m->mutable_parameter(nn::Linear::kParamBiasName));
      }
    }
  });
}

std::vector<std::shared_ptr<Tensor>> MobileNetV2::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  auto x = (*modules_["features"])(input_tensors);
  x = (*modules_["avgpool"])(x);
  x = (*modules_["flatten"])(x);
  return (*modules_["classifier"])(x);
}

}  // namespace infini_train::example::cifar10

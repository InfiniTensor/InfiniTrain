#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <utility>

#include "infini_train/include/tensor.h"

namespace infini_train::nn::init {
std::shared_ptr<Tensor> Normal(const std::shared_ptr<Tensor> &tensor, float mean = 0.0, float std = 1.0,
                               std::optional<std::mt19937> generator = std::nullopt);

std::pair<int64_t, int64_t> CalculateFanInAndFanOut(const std::shared_ptr<Tensor> &tensor);

enum class KaimingMode : int8_t {
    kFanIn,
    kFanOut,
};

enum class NonLinearityType : int8_t {
    kLinear,
    kConv1D,
    kConv2D,
    kConv3D,
    kConvTransposed1d,
    kConvTransposed2d,
    kConvTransposed3d,
    kSigmoid,
    kTanh,
    kReLU,
    kLeakyReLU,
    kSELU,
};

std::shared_ptr<Tensor> KaimingUniform(const std::shared_ptr<Tensor> &tensor, float a = 0.0f,
                                       KaimingMode mode = KaimingMode::kFanIn,
                                       NonLinearityType non_linearity = NonLinearityType::kLeakyReLU,
                                       std::optional<std::mt19937> generator = std::nullopt);

std::shared_ptr<Tensor> Uniform(const std::shared_ptr<Tensor> &tensor, float a = 0.0f, float b = 1.0f,
                                std::optional<std::mt19937> generator = std::nullopt);

std::shared_ptr<Tensor> Ones(const std::shared_ptr<Tensor> &tensor);

std::shared_ptr<Tensor> Zeros(const std::shared_ptr<Tensor> &tensor);

std::shared_ptr<Tensor> Arange(int64_t start, int64_t end, DataType dtype, Device device = Device());
} // namespace infini_train::nn::init

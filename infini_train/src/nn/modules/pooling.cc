#include "infini_train/include/nn/modules/pooling.h"

#include <memory>
#include <utility>
#include <vector>

#include "infini_train/include/autograd/pooling.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

MaxPool2d::MaxPool2d(int64_t kernel_size, int64_t stride, int64_t padding)
    : MaxPool2d({kernel_size, kernel_size},
                stride > 0
                    ? std::pair<int64_t, int64_t>{stride, stride}
                    : std::pair<int64_t, int64_t>{kernel_size, kernel_size},
                {padding, padding}) {}

MaxPool2d::MaxPool2d(std::pair<int64_t, int64_t> kernel_size,
                     std::pair<int64_t, int64_t> stride,
                     std::pair<int64_t, int64_t> padding)
    : CloneableModule(kType),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {}

std::vector<std::shared_ptr<Tensor>> MaxPool2d::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  auto outputs =
      std::make_shared<autograd::MaxPool2d>(kernel_size_, stride_, padding_)
          ->Apply(input_tensors);
  return {outputs[0]};
}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(int64_t output_size)
    : AdaptiveAvgPool2d({output_size, output_size}) {}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(std::pair<int64_t, int64_t> output_size)
    : CloneableModule(kType), output_size_(output_size) {}

std::vector<std::shared_ptr<Tensor>> AdaptiveAvgPool2d::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  return std::make_shared<autograd::AdaptiveAvgPool2d>(output_size_)
      ->Apply(input_tensors);
}

}  // namespace infini_train::nn

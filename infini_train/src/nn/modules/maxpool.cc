#include "infini_train/include/nn/modules/maxpool.h"

#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/maxpool.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
MaxPool2D::MaxPool2D(size_t kernel_size, size_t stride, size_t padding, size_t dilation, Device device)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation) {
    device_ = device;
    ResetParameters();
}

std::vector<std::shared_ptr<Tensor>> MaxPool2D::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return std::make_shared<autograd::MaxPool2D>(kernel_size_, stride_, padding_, dilation_)->Apply({input_tensors[0]});
}

void MaxPool2D::ResetParameters() {}
} // namespace infini_train::nn
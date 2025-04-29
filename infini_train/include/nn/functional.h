#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::nn::function {

// Returns the lower triangular part of a 2D tensor or a batch of matrices.
//
// The lower triangular part includes elements on and below the specified
// diagonal. Elements above the diagonal are set to zero.
//
// Args:
//   input: The input tensor.
//   diagonal: Diagonal offset (default 0). Positive means above the main diagonal,
//             negative means below.
//
// Returns:
//   A tensor with the same shape as input, with upper-triangular values zeroed.
std::shared_ptr<Tensor> Tril(const std::shared_ptr<Tensor> &input, int64_t diagonal = 0);

// Returns a tensor filled with ones of the specified shape.
//
// Args:
//   size: A vector specifying the shape of the output tensor.
//
// Returns:
//   A tensor of the given shape filled with the scalar value 1.
std::shared_ptr<Tensor> Ones(const std::vector<int64_t> size);

// Returns a new tensor with the hyperbolic tangent of each element in the input.
//
// Args:
//   input: The input tensor.
//
// Returns:
//   A tensor containing tanh applied element-wise to the input.
std::shared_ptr<Tensor> Tanh(const std::shared_ptr<Tensor> &input);

// Raises each element of the input tensor to the specified power.
//
// Args:
//   input: The input tensor.
//   exponent: The exponent to apply to each element.
//
// Returns:
//   A tensor with each element raised to the given exponent.
std::shared_ptr<Tensor> Pow(const std::shared_ptr<Tensor> &input, float exponent);

// Applies the softmax function along the specified dimension.
//
// The softmax function maps input values to the range [0, 1] and ensures they sum to 1.
//
// Args:
//   input: The input tensor.
//   dim: The dimension along which softmax is computed (default -1).
//
// Returns:
//   A tensor with softmax applied along the specified dimension.
std::shared_ptr<Tensor> Softmax(const std::shared_ptr<Tensor> &input, int64_t dim = -1);

// Returns a slice of the input tensor defined by start, end, and step per dimension.
//
// Args:
//   input: The input tensor.
//   starts: Start indices for each dimension.
//   ends: End indices for each dimension (exclusive).
//   steps: Step sizes for each dimension.
//
// Returns:
//   A sliced view of the input tensor.
std::shared_ptr<Tensor> Slice(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                              const std::vector<int64_t> &ends, const std::vector<int64_t> &steps);

} // namespace infini_train::nn::function

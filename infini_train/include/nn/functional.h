#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace infini_train {
class Tensor;
}

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

// Returns the upper triangular part of a 2D tensor or a batch of matrices.
//
// The upper triangular part includes elements on and above the specified
// diagonal. Elements below the diagonal are set to zero.
//
// Args:
//   input: The input tensor.
//   diagonal: Diagonal offset (default 0). Positive means above the main diagonal,
//             negative means below.
//
// Returns:
//   A tensor with the same shape as input, with lower-triangular values zeroed.
std::shared_ptr<Tensor> Triu(const std::shared_ptr<Tensor> &input, int64_t diagonal = 0);

// Returns a tensor filled with ones of the specified shape.
//
// Args:
//   size: A vector specifying the shape of the output tensor.
//
// Returns:
//   A tensor of the given shape filled with the scalar value 1.
std::shared_ptr<Tensor> Ones(const std::vector<int64_t> size);

// Returns a new tensor with the reciprocal of the elements of input.
//
// Args:
//   input: The input tensor.
//
// Returns:
//   A tensor containing reciprocal applied element-wise to the input.
std::shared_ptr<Tensor> Reciprocal(const std::shared_ptr<Tensor> &input);

// Returns a new tensor with the sine of each element in the input.
//
// Args:
//   input: The input tensor.
//
// Returns:
//   A tensor containing sin applied element-wise to the input.
std::shared_ptr<Tensor> Sin(const std::shared_ptr<Tensor> &input);

// Returns a new tensor with the cosine of each element in the input.
//
// Args:
//   input: The input tensor.
//
// Returns:
//   A tensor containing cos applied element-wise to the input.
std::shared_ptr<Tensor> Cos(const std::shared_ptr<Tensor> &input);

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

// Raises the specified base to the power of each element in the input tensor.
//
// Args:
//   base: The scalar base value.
//   input: The input tensor providing the exponents.
//
// Returns:
//   A tensor where each element is computed as base raised to the power of the corresponding input element.
std::shared_ptr<Tensor> Pow(float base, const std::shared_ptr<Tensor> &input);

// Returns a new tensor with reciprocal of the square-root of each element in the input.
//
// Args:
//   input: The input tensor.
//
// Returns:
//   A tensor containing reciprocal square-root applied element-wise to the input.
std::shared_ptr<Tensor> Rsqrt(const std::shared_ptr<Tensor> &input);

// Returns the aggregate of all elements in the input tensor.
//
// Args:
//   input: The input tensor.
//   dim: The dimension to reduce.
//   keep_dim: Whether the output tensor has dim retained or not (default false).
//
// Returns:
//   A new tensor with the aggregate values computed along the specified dimension.
std::shared_ptr<Tensor> Mean(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim = false);
std::shared_ptr<Tensor> Sum(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim = false);
std::shared_ptr<Tensor> Min(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim = false);
std::shared_ptr<Tensor> Max(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim = false);

// Returns a new tensor with the sigmoid of each element in the input.
//
// The sigmoid function is defined as 1 / (1 + exp(-x)).
//
// Args:
//   input: The input tensor.
//
// Returns:
//   A tensor containing sigmoid applied element-wise to the input.
std::shared_ptr<Tensor> Sigmoid(const std::shared_ptr<Tensor> &input);

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

// Concatenates a sequence of tensors along a new dimension.
//
// Args:
//   inputs: The sequence of tensors to concatenate. All tensors need to be of the same size.
//   dim: dimension to insert (defualt 0).
//
// Returns:
//   Concatenation of the input tensors.
std::shared_ptr<Tensor> Stack(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim = 0);

// Concatenates the given sequence of tensors in tensors in the given dimension.
//
// Args:
//   inputs: The sequence of tensors to concatenate. All tensors must either have the same shape (except in the
//   concatenating dimension) or be a 1-D empty tensor with size (0,). dim: dimension to insert (defualt 0).
//   dim: dimension to insert (defualt 0).
//
// Returns:
//   Concatenation of the input tensors.
std::shared_ptr<Tensor> Concat(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim = 0);

} // namespace infini_train::nn::function

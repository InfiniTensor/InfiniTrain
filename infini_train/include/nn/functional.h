#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::nn::function {
/*
     Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
     :attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

     The lower triangular part of the matrix is defined as the elements on and
     below the diagonal.

     The argument :attr:`diagonal` controls which diagonal to consider. If
     :attr:`diagonal` = 0, all elements on and below the main diagonal are
     retained. A positive value includes just as many diagonals above the main
     diagonal, and similarly a negative value excludes just as many diagonals below
     the main diagonal. The main diagonal are the set of indices
     :math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
     :math:`d_{1}, d_{2}` are the dimensions of the matrix.
 */
std::shared_ptr<Tensor> Tril(const std::shared_ptr<Tensor> &input, int64_t diagonal = 0);

/*
    Returns a tensor filled with the scalar value `1`, with the shape defined
    by the variable argument :attr:`size`.
*/
std::shared_ptr<Tensor> Ones(const std::vector<int64_t> size);

/*
    Returns a new tensor with the hyperbolic tangent of the elements
    of :attr:`input`.
*/
std::shared_ptr<Tensor> Tanh(const std::shared_ptr<Tensor> &input);

/*
    Takes the power of each element in :attr:`input` with :attr:`exponent` and
    returns a tensor with the result.
*/
std::shared_ptr<Tensor> Pow(const std::shared_ptr<Tensor> &input, float exponent);

/*
    Apply a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
*/
std::shared_ptr<Tensor> Softmax(const std::shared_ptr<Tensor> &input, int64_t dim = -1);

std::shared_ptr<Tensor> Slice(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                              const std::vector<int64_t> &ends, const std::vector<int64_t> &steps);
} // namespace infini_train::nn::function

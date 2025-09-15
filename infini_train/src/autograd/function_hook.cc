#include "infini_train/include/autograd/function_hook.h"

#include <memory>

#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
void AllReducePostAccumulateHook::operator()(const std::shared_ptr<Tensor> &tensor) {
    infini_train::nn::parallel::function::AllReduce(tensor, reduce_op_);
}
} // namespace infini_train::autograd

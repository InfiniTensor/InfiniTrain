#include "infini_train/include/autograd/function_hook.h"

#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
AllReducePostAccumulateHook::AllReducePostAccumulateHook(infini_train::nn::parallel::function::ReduceOpType reduce_op,
                                                         const infini_train::nn::parallel::ProcessGroup *pg)
    : reduce_op_(reduce_op),
      pg_(pg ? pg : infini_train::nn::parallel::ProcessGroupFactory::Instance()->GetDefaultProcessGroup()) {}

void AllReducePostAccumulateHook::operator()(const std::shared_ptr<Tensor> &tensor) {
    infini_train::nn::parallel::function::AllReduce(tensor, reduce_op_, pg_);
}
} // namespace infini_train::autograd

#include "infini_train/include/autograd/function_hook.h"

#include <utility>

#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
AllReducePostAccumulateHook::AllReducePostAccumulateHook(infini_train::nn::parallel::function::ReduceOpType reduce_op,
                                                         const infini_train::nn::parallel::ProcessGroup *pg,
                                                         std::shared_ptr<const std::atomic_bool> enabled)
    : reduce_op_(reduce_op),
      pg_(pg ? pg : infini_train::nn::parallel::ProcessGroupFactory::Instance()->GetDefaultProcessGroup()),
      enabled_(std::move(enabled)) {}

void AllReducePostAccumulateHook::operator()(const std::shared_ptr<Tensor> &tensor) {
    if (enabled_ && !enabled_->load(std::memory_order_relaxed)) {
        return;
    }
    infini_train::nn::parallel::function::AllReduce(tensor, reduce_op_, pg_);
}
} // namespace infini_train::autograd

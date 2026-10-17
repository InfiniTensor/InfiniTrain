#include "infini_train/include/autograd/function_hook.h"

#include <utility>

#include "infini_train/include/nn/parallel/comm.h"
#include "infini_train/include/nn/parallel/process_group.h"

namespace infini_train::nn::parallel::comm {
enum class ReduceOpType : int8_t;
} // namespace infini_train::nn::parallel::comm

namespace infini_train::autograd {
bool PreAccumulateGradHook::TryBypassAccumulate(const std::shared_ptr<Tensor> &, const std::shared_ptr<Tensor> &, bool,
                                                float) {
    return false;
}

AllReducePostAccumulateHook::AllReducePostAccumulateHook(infini_train::nn::parallel::comm::ReduceOpType reduce_op,
                                                         const infini_train::nn::parallel::ProcessGroup *pg,
                                                         std::shared_ptr<const std::atomic_bool> enabled)
    : reduce_op_(reduce_op),
      pg_(pg ? pg : infini_train::nn::parallel::ProcessGroupFactory::Instance()->GetDefaultProcessGroup()),
      enabled_(std::move(enabled)) {}

void AllReducePostAccumulateHook::operator()(const std::shared_ptr<Tensor> &tensor) {
    if (enabled_ && !enabled_->load(std::memory_order_relaxed)) {
        return;
    }
    infini_train::nn::parallel::comm::AllReduce(tensor, reduce_op_, pg_);
}
} // namespace infini_train::autograd

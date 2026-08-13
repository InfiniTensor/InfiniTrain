#include "infini_train/include/nn/parallel/comm.h"

#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel::comm {

std::shared_ptr<Work> AllReduce(const std::shared_ptr<Tensor> &tensor, ReduceOpType reduce_op, const ProcessGroup *pg,
                                bool async_op) {
    if (pg == nullptr) {
        pg = ProcessGroupFactory::Instance(tensor->GetDevice().type())->GetDefaultProcessGroup();
    }
    return pg->AllReduce(tensor, reduce_op, async_op);
}

std::shared_ptr<Work> AllGather(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                const ProcessGroup *pg, bool async_op) {
    if (pg == nullptr) {
        pg = ProcessGroupFactory::Instance(output->GetDevice().type())->GetDefaultProcessGroup();
    }
    return pg->AllGather(output, input, async_op);
}

std::shared_ptr<Work> ReduceScatter(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                    ReduceOpType reduce_op, const ProcessGroup *pg, bool async_op) {
    if (pg == nullptr) {
        pg = ProcessGroupFactory::Instance(output->GetDevice().type())->GetDefaultProcessGroup();
    }
    return pg->ReduceScatter(output, input, reduce_op, async_op);
}

} // namespace infini_train::nn::parallel::comm

#pragma once

#include <stdint.h>

#include <memory>

namespace infini_train {
class Tensor;
}

namespace infini_train::nn::parallel {
class ProcessGroup;
class Work;
}  // namespace infini_train::nn::parallel

namespace infini_train::nn::parallel::comm {
enum class ReduceOpType : int8_t;

std::shared_ptr<Work> AllReduce(const std::shared_ptr<Tensor> &tensor,
                                ReduceOpType reduce_op,
                                const ProcessGroup *pg = nullptr,
                                bool async_op = false);

std::shared_ptr<Work> AllGather(const std::shared_ptr<Tensor> &output,
                                const std::shared_ptr<Tensor> &input,
                                const ProcessGroup *pg = nullptr,
                                bool async_op = false);

std::shared_ptr<Work> ReduceScatter(const std::shared_ptr<Tensor> &output,
                                    const std::shared_ptr<Tensor> &input,
                                    ReduceOpType reduce_op,
                                    const ProcessGroup *pg = nullptr,
                                    bool async_op = false);

std::shared_ptr<Work> AlltoAll(const std::shared_ptr<Tensor> &output,
                               const std::shared_ptr<Tensor> &input,
                               const ProcessGroup *pg = nullptr,
                               bool async_op = false);

}  // namespace infini_train::nn::parallel::comm

#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train {
class Tensor;
namespace nn {
class Module;
}
} // namespace infini_train

namespace infini_train::nn::parallel::function {

std::shared_ptr<Work> AllReduce(const std::shared_ptr<Tensor> &tensor, ReduceOpType reduce_op,
                                const ProcessGroup *pg = nullptr, bool async_op = false);

std::shared_ptr<Work> AllGather(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                const ProcessGroup *pg = nullptr, bool async_op = false);

std::shared_ptr<Work> ReduceScatter(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                    ReduceOpType reduce_op, const ProcessGroup *pg = nullptr, bool async_op = false);

std::vector<std::vector<std::shared_ptr<Tensor>>> Scatter(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                                          const std::vector<Device> &device_ids, int dim);

std::vector<std::shared_ptr<Tensor>> Gather(const std::vector<std::vector<std::shared_ptr<Tensor>>> &outputs,
                                            Device target_device, int dim);

std::vector<std::vector<std::shared_ptr<Tensor>>>
BroadcastCoalescedReshape(const std::vector<std::shared_ptr<Tensor>> &tensors, const std::vector<Device> &devices);

std::vector<std::shared_ptr<Module>> Replicate(const std::shared_ptr<Module> &network,
                                               const std::vector<Device> &devices);

} // namespace infini_train::nn::parallel::function

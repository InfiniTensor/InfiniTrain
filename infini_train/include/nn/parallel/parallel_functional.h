#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train {
class Tensor;
class Device;
namespace nn {
class Module;
}
} // namespace infini_train

namespace infini_train::nn::parallel::function {
std::vector<std::vector<std::shared_ptr<Tensor>>> Scatter(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                                          const std::vector<const Device *> &device_ids, int dim);

std::vector<std::shared_ptr<Tensor>> Gather(const std::vector<std::vector<std::shared_ptr<Tensor>>> &outputs,
                                            const Device *target_device, int dim);

void AllReduce(const std::shared_ptr<Tensor> &tensor, ReduceOpType reduce_op, const ProcessGroup *pg);

std::vector<std::vector<std::shared_ptr<Tensor>>>
BroadcastCoalescedReshape(const std::vector<std::shared_ptr<Tensor>> &tensors,
                          const std::vector<const Device *> &devices);

std::vector<std::shared_ptr<Module>> Replicate(const std::shared_ptr<Module> &network,
                                               const std::vector<const Device *> &devices);

} // namespace infini_train::nn::parallel::function

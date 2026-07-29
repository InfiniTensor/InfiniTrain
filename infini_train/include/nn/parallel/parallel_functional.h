#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train {
class Tensor;
namespace nn {
class Module;
namespace parallel {
class ProcessGroup;
}
} // namespace nn
} // namespace infini_train

namespace infini_train::nn::parallel::function {

// Concatenates equal-shaped tensors from the group along the first dimension.
// Backward uses SUM ReduceScatter.
std::shared_ptr<Tensor> AllGather(const std::shared_ptr<Tensor> &input, const ProcessGroup *pg);

// Reduces and scatters input along the first dimension.
// Backward is currently supported only for SUM reduction, where it is implemented as AllGather.
std::shared_ptr<Tensor> ReduceScatter(const std::shared_ptr<Tensor> &input, comm::ReduceOpType reduce_op,
                                      const ProcessGroup *pg);

std::vector<std::vector<std::shared_ptr<Tensor>>> Scatter(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                                          const std::vector<Device> &device_ids, int dim);

std::vector<std::shared_ptr<Tensor>> Gather(const std::vector<std::vector<std::shared_ptr<Tensor>>> &outputs,
                                            Device target_device, int dim);

std::vector<std::vector<std::shared_ptr<Tensor>>>
BroadcastCoalescedReshape(const std::vector<std::shared_ptr<Tensor>> &tensors, const std::vector<Device> &devices);

std::vector<std::shared_ptr<Module>> Replicate(const std::shared_ptr<Module> &network,
                                               const std::vector<Device> &devices);

} // namespace infini_train::nn::parallel::function

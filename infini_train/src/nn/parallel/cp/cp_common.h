#pragma once

#include <cstdint>
#include <memory>

namespace infini_train {
class Tensor;
}  // namespace infini_train

namespace infini_train::nn::parallel {
class ProcessGroup;

namespace detail {

std::shared_ptr<Tensor> SliceCPRank(const std::shared_ptr<Tensor> &input,
                                    int64_t dim, int rank, int cp_size);
std::shared_ptr<Tensor> ReorderCPSequence(const std::shared_ptr<Tensor> &input,
                                          int64_t dim, int cp_size,
                                          bool to_sequence_order);
const ProcessGroup *GetCPGroup(const std::shared_ptr<Tensor> &tensor);
std::shared_ptr<Tensor> GatherAlongFirstDim(
    const std::shared_ptr<Tensor> &tensor, const ProcessGroup *cp_group);
std::shared_ptr<Tensor> GatherFromCPRegionFunc(
    const std::shared_ptr<Tensor> &input);
std::shared_ptr<Tensor> AllToAllSeqToHeadCPRegionFunc(
    const std::shared_ptr<Tensor> &input, const ProcessGroup *cp_group);
std::shared_ptr<Tensor> AllToAllHeadToSeqCPRegionFunc(
    const std::shared_ptr<Tensor> &input, const ProcessGroup *cp_group);
std::shared_ptr<Tensor> ApplyAttnWithCPAndKVP2P(
    const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask,
    const ProcessGroup *cp_group);

}  // namespace detail
}  // namespace infini_train::nn::parallel

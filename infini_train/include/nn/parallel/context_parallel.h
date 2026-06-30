#pragma once

#include <cstdint>
#include <memory>

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::nn::parallel {

extern thread_local int cp_rank;

int GetContextParallelRank();

int64_t GetContextParallelSequenceStart(int64_t local_sequence_length);

std::shared_ptr<Tensor> SliceAlongCPRegionFunc(const std::shared_ptr<Tensor> &input, int64_t dim);

std::shared_ptr<Tensor> GatherFromCPRegionFunc(const std::shared_ptr<Tensor> &input);

std::shared_ptr<Tensor> ContextParallelAttentionFunc(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                                     const std::shared_ptr<Tensor> &v,
                                                     const std::shared_ptr<Tensor> &mask, bool mask_true_means_invalid,
                                                     float scale, int64_t n_rep);

std::shared_ptr<Tensor> AttnFuncWithCPAndKVP2P(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                               const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask,
                                               bool mask_true_means_invalid, float scale, int64_t n_rep);

std::shared_ptr<Tensor> AttnFuncWithCPAndKVAllGather(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                                     const std::shared_ptr<Tensor> &v,
                                                     const std::shared_ptr<Tensor> &mask, bool mask_true_means_invalid,
                                                     float scale, int64_t n_rep);

std::shared_ptr<Tensor> AttnFuncWithCPAndQKVOA2A(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                                 const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask,
                                                 bool mask_true_means_invalid, float scale, int64_t n_rep);

} // namespace infini_train::nn::parallel

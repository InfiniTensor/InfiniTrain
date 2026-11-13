#pragma once

#include <cstdint>
#include <memory>

namespace infini_train {
class Tensor;
class Device;
}  // namespace infini_train

namespace infini_train::nn::parallel {

extern thread_local int cp_rank;

int GetContextParallelRank();

std::shared_ptr<Tensor> SliceAlongCPRegionFunc(
    const std::shared_ptr<Tensor> &input, int64_t dim);

std::shared_ptr<Tensor> MakeCausalMaskAlongCPRegionFunc(int64_t local_t,
                                                        const Device &device);

std::shared_ptr<Tensor> AttnForwardFuncWithCP(
    const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
    const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &mask);

}  // namespace infini_train::nn::parallel

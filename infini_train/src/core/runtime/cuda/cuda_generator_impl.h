#pragma once

#include <cstdint>
#include <memory>

#include "infini_train/include/generator.h"
#include "infini_train/include/generator_impl.h"

namespace infini_train::core::cuda {

class CUDAGeneratorImpl final : public GeneratorImpl {
public:
    explicit CUDAGeneratorImpl(int8_t device_index, uint64_t seed = Generator::kDefaultSeed);
    ~CUDAGeneratorImpl() override = default;

    void set_current_seed(uint64_t seed) override;
    uint64_t current_seed() const override;
    uint64_t Seed() override;
    void set_state(const Tensor &state) override;
    std::shared_ptr<Tensor> get_state() const override;

    // The caller must hold mutex_ while reserving Philox subsequences.
    uint64_t ReservePhiloxSubsequence(uint64_t increment);

private:
    CUDAGeneratorImpl *CloneImpl() const override;

    uint64_t seed_ = Generator::kDefaultSeed;
    uint64_t next_philox_subsequence_ = 0;
};

const Generator &GetDefaultCudaGenerator(int8_t device_index = -1);
Generator CreateCudaGenerator(int8_t device_index, uint64_t seed = Generator::kDefaultSeed);
void ManualSeedAll(uint64_t seed);

} // namespace infini_train::core::cuda

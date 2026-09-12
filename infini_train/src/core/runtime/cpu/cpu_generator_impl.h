#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>

#include "infini_train/include/generator.h"
#include "infini_train/include/generator_impl.h"
#include "infini_train/include/tensor.h"

namespace infini_train::core::cpu {

// CPU generator backed by std::mt19937 with cached Box-Muller samples.
class CPUGeneratorImpl final : public GeneratorImpl {
public:
    explicit CPUGeneratorImpl(uint64_t seed = Generator::kDefaultSeed);
    ~CPUGeneratorImpl() override = default;

    void set_current_seed(uint64_t seed) override;
    uint64_t current_seed() const override;
    uint64_t Seed() override;
    void set_state(const Tensor &state) override;
    std::shared_ptr<Tensor> get_state() const override;

    std::shared_ptr<CPUGeneratorImpl> Clone() const;

    uint32_t Random();
    uint64_t Random64();

    std::optional<float> next_float_normal_sample() const;
    std::optional<double> next_double_normal_sample() const;
    void set_next_float_normal_sample(std::optional<float> randn);
    void set_next_double_normal_sample(std::optional<double> randn);

private:
    CPUGeneratorImpl *CloneImpl() const override;

    std::mt19937 engine() const { return engine_; }
    void set_engine(std::mt19937 engine);

    std::mt19937 engine_;
    uint64_t seed_ = Generator::kDefaultSeed;
    std::optional<float> next_float_normal_sample_;
    std::optional<double> next_double_normal_sample_;
};

const Generator &GetDefaultCpuGenerator();
Generator CreateCpuGenerator(uint64_t seed);
void ManualSeed(uint64_t seed);

} // namespace infini_train::core::cpu

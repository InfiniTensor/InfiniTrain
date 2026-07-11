#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>

#include "infini_train/include/generator.h"

namespace infini_train::core::cpu {

// ============================================================
// CPUGeneratorImpl — CPU RNG 后端（仿 at::CPUGeneratorImpl）
// ============================================================
// 包装 std::mt19937 Mersenne Twister 引擎。
// 缓存 Box-Muller 正态分布样本以优化性能。
//
class CPUGeneratorImpl final : public GeneratorImpl {
public:
    explicit CPUGeneratorImpl(uint64_t seed = Generator::kDefaultSeed);
    ~CPUGeneratorImpl() override = default;

    // ---- GeneratorImpl 接口 ----
    void set_current_seed(uint64_t seed) override;
    uint64_t current_seed() const override;
    uint64_t seed() override;
    void set_state(const Tensor &state) override;
    std::shared_ptr<Tensor> get_state() const override;

    // ---- clone（类型安全版本）----
    std::shared_ptr<CPUGeneratorImpl> clone() const;

    // ---- 类型标识（用于 check_generator<T> 模板）----
    static Device::DeviceType device_type();

    // ---- 随机数生成 ----
    uint32_t random();
    uint64_t random64();

    // ---- Box-Muller 正态缓存 ----
    std::optional<float> next_float_normal_sample() const;
    std::optional<double> next_double_normal_sample() const;
    void set_next_float_normal_sample(std::optional<float> randn);
    void set_next_double_normal_sample(std::optional<double> randn);

private:
    CPUGeneratorImpl *clone_impl() const override;

    // ---- 引擎（private：比赛要求公共接口不暴露 std::mt19937）----
    std::mt19937 engine() const { return engine_; }
    void set_engine(std::mt19937 engine);

    std::mt19937 engine_;
    uint64_t seed_ = Generator::kDefaultSeed;
    std::optional<float> next_float_normal_sample_;
    std::optional<double> next_double_normal_sample_;
};

// ---- 默认 Generator 管理（仿 PyTorch detail 命名空间）----
const Generator &getDefaultCPUGenerator();
Generator createCPUGenerator(uint64_t seed);
void manual_seed(uint64_t seed);

} // namespace infini_train::core::cpu

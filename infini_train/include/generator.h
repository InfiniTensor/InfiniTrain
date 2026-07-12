#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "infini_train/include/device.h"

namespace infini_train {

// 前向声明，避免循环依赖（仿 PyTorch at::Tensor 前向声明）
class Tensor;

namespace detail {

// Validates the common Tensor contract for serialized RNG states.
void check_rng_state(const Tensor &state);

} // namespace detail

// ============================================================
// GeneratorImpl — 抽象基类（仿 c10::GeneratorImpl）
// ============================================================
// 定义所有 RNG 后端必须实现的纯虚接口。
//
// 拷贝/移动已删除，只能通过 clone() 显式深拷贝。
// clone 采用 NVI（Non-Virtual Interface）模式：
//   - clone() 公有非虚，内部调用 protected clone_impl()
//   - 子类只需覆写 clone_impl() 返回堆上分配的同类型拷贝
//
// 线程安全：mutex_ 是 public 的，调用方对多步原子操作自行加锁。
//
class GeneratorImpl {
public:
    explicit GeneratorImpl(Device device) : device_(device) {}
    virtual ~GeneratorImpl() = default;

    // 禁止拷贝 / 移动，避免意外覆盖 RNG 状态
    GeneratorImpl(const GeneratorImpl &other) = delete;
    GeneratorImpl(GeneratorImpl &&other) = delete;
    GeneratorImpl &operator=(const GeneratorImpl &other) = delete;
    GeneratorImpl &operator=(GeneratorImpl &&other) = delete;

    // ---- 纯虚接口（子类必须实现）----
    virtual void set_current_seed(uint64_t seed) = 0;
    virtual uint64_t current_seed() const = 0;
    virtual uint64_t seed() = 0;
    virtual void set_state(const Tensor &state) = 0;
    virtual std::shared_ptr<Tensor> get_state() const = 0;

    // ---- NVI clone ----
    std::shared_ptr<GeneratorImpl> clone() const {
        return std::shared_ptr<GeneratorImpl>(clone_impl());
    }

    // ---- 设备 ----
    Device device() const { return device_; }

    // 线程安全（public，调用方自行加锁）
    std::mutex mutex_;

protected:
    Device device_;

    // 子类覆写点：返回堆上分配的同类型拷贝
    virtual GeneratorImpl *clone_impl() const = 0;
};

// ============================================================
// Generator — 值类型壳（仿 at::Generator）
// ============================================================
// 轻量级、可拷贝的值类型。拷贝是浅拷贝——两个 Generator
// 共享同一个 GeneratorImpl，推进一个对另一个可见。
//
// 默认构造的 Generator 处于 "undefined" 状态（impl_ == nullptr）。
// 使用 make_generator<CPUGeneratorImpl>(seed) 或 Generator(impl)
// 来创建可用的实例。
//
class Generator {
public:
    // 默认种子：一个大数，bit 分布均匀（仿 PyTorch default_rng_seed_val）
    static constexpr uint64_t kDefaultSeed = 67280421310721;

    // 默认构造：undefined 状态
    Generator() = default;

    // 从已有 Impl 构造（impl 不能为 nullptr，实现在 .cc 做检查）
    explicit Generator(std::shared_ptr<GeneratorImpl> impl);

    // 拷贝 / 移动（浅拷贝，共享 impl_）
    Generator(const Generator &) = default;
    Generator &operator=(const Generator &) = default;
    Generator(Generator &&) = default;
    Generator &operator=(Generator &&) = default;

    ~Generator() = default;

    // ---- 种子 ----
    void set_current_seed(uint64_t seed) const { impl_->set_current_seed(seed); }
    uint64_t current_seed() const { return impl_->current_seed(); }
    uint64_t seed() { return impl_->seed(); }

    // ---- 状态序列化（实现在 .cc，需要 Tensor 完整定义）----
    void set_state(const Tensor &state);
    std::shared_ptr<Tensor> get_state() const;

    // ---- 设备 ----
    Device device() const { return impl_->device(); }

    // ---- 克隆（深拷贝 Impl）----
    Generator clone() const { return Generator(impl_->clone()); }

    // ---- 线程安全 ----
    std::mutex &mutex() const { return impl_->mutex_; }

    // ---- 非检查 downcast ----
    // 调用方必须先确认 Impl 类型与 Generator 的设备类型匹配。
    // 算子代码应使用 check_generator<T>()。
    template <typename T>
    T *get() const {
        return static_cast<T *>(impl_.get());
    }

    // ---- 底层访问 ----
    GeneratorImpl *unsafeGetGeneratorImpl() const { return impl_.get(); }
    bool defined() const { return impl_ != nullptr; }

    // ---- 比较 ----
    friend bool operator==(const Generator &a, const Generator &b) {
        return a.impl_ == b.impl_;
    }
    friend bool operator!=(const Generator &a, const Generator &b) {
        return !(a == b);
    }

private:
    std::shared_ptr<GeneratorImpl> impl_;
};

// ============================================================
// 工具函数
// ============================================================

// 工厂函数：make_generator<CPUGeneratorImpl>(seed)
template <class Impl, class... Args>
Generator make_generator(Args &&...args) {
    return Generator(std::make_shared<Impl>(std::forward<Args>(args)...));
}

// 检查 Generator 已定义且属于 T 对应的设备后端，再进行 downcast。
template <typename T>
T *check_generator(const Generator &generator) {
    if (!generator.defined()) {
        throw std::invalid_argument("Generator with undefined implementation is not allowed");
    }
    if (T::device_type() != generator.device().type()) {
        throw std::invalid_argument("Generator device type does not match the requested backend");
    }

    auto *impl = dynamic_cast<T *>(generator.unsafeGetGeneratorImpl());
    if (impl == nullptr) {
        throw std::invalid_argument("Generator implementation does not match the requested backend");
    }
    return impl;
}

// 显式 Generator 优先；未提供或未定义时使用该设备的默认 Generator。
template <typename T>
T *get_generator_or_default(const std::optional<Generator> &generator,
                            const Generator &default_generator) {
    return generator.has_value() && generator->defined()
        ? check_generator<T>(*generator)
        : check_generator<T>(default_generator);
}

// Reset the default generators for all enabled devices.
void manual_seed(uint64_t seed);

} // namespace infini_train

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace infini_train {
namespace {

// ============================================================
// uniform_real_distribution — 均匀分布函子
// ============================================================
// 仿 PyTorch at::uniform_real_distribution
// （aten/src/ATen/core/DistributionsHelper.h:99）
//
// 调用 generator->random()（float）或 generator->random64()（double）
// 获取原始随机比特，变换到 [from, to) 区间。
//
// 模板参数 T = float | double
// 模板参数 RNG = CPUGeneratorImpl* | CUDAGeneratorImpl* | ...
//
template <typename T>
struct uniform_real_distribution {
    uniform_real_distribution(T from, T to) : from_(from), to_(to) {
        assert(from <= to);
        assert(to - from <= std::numeric_limits<T>::max());
    }
    //
    // 不允许重新绑定 from/to，因为分布函子是无状态的
    uniform_real_distribution(const uniform_real_distribution &) = default;
    uniform_real_distribution &operator=(const uniform_real_distribution &) = delete;

    template <typename RNG>
    T operator()(RNG *generator) const {
        if constexpr (std::is_same_v<T, double>) {
            return transform(generator->random64());
        } else {
            return transform(generator->random());
        }
    }

private:
    T from_;
    T to_;

    // 变换：raw bits → [0, 1) → [from_, to_)
    // 仿 PyTorch at::transformation::uniform_real
    // （aten/src/ATen/core/TransformationHelper.h:84）
    template <typename V>
    T transform(V val) const {
        constexpr auto MASK
            = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
        constexpr auto DIVISOR
            = static_cast<T>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
        T x = (val & MASK) * DIVISOR;
        return x * (to_ - from_) + from_;
    }
};

// ============================================================
// Box-Muller 正态分布缓存辅助（SFINAE）
// ============================================================
// 仿 PyTorch at::maybe_get_next_normal_sample / maybe_set_next_normal_sample
// （aten/src/ATen/core/DistributionsHelper.h:120-163）
//
// 如果 RNG 有 next_float_normal_sample() 系列方法（如 CPUGeneratorImpl），
// 则 Box-Muller 第二个样本被缓存到 Generator 里。
// 如果 RNG 没有这些方法（如 CUDA curand state），则 SFINAE 回退到 no-op，
// 每次生成两个样本但只返回一个（丢弃另一个）。
//

// ---- get: 尝试读取缓存 ----

template <typename RNG,
          typename = decltype(&RNG::next_double_normal_sample),
          typename = decltype(&RNG::set_next_double_normal_sample)>
bool maybe_get_next_normal_sample(RNG *generator, double *ret) {
    const auto sample = generator->next_double_normal_sample();
    if (!sample.has_value())
        return false;
    *ret = sample.value();
    generator->set_next_double_normal_sample(std::nullopt);
    return true;
}

template <typename RNG,
          typename = decltype(&RNG::next_float_normal_sample),
          typename = decltype(&RNG::set_next_float_normal_sample)>
bool maybe_get_next_normal_sample(RNG *generator, float *ret) {
    const auto sample = generator->next_float_normal_sample();
    if (!sample.has_value())
        return false;
    *ret = sample.value();
    generator->set_next_float_normal_sample(std::nullopt);
    return true;
}

// 兜底：不支持缓存时总是返回 false
template <typename RNG>
bool maybe_get_next_normal_sample(RNG * /*generator*/, void * /*ret*/) {
    return false;
}

// ---- set: 写入缓存 ----

template <typename RNG, typename = decltype(&RNG::set_next_double_normal_sample)>
void maybe_set_next_normal_sample(RNG *generator, const double *cache) {
    generator->set_next_double_normal_sample(*cache);
}

template <typename RNG, typename = decltype(&RNG::set_next_float_normal_sample)>
void maybe_set_next_normal_sample(RNG *generator, const float *cache) {
    generator->set_next_float_normal_sample(*cache);
}

// 兜底：不支持缓存时 no-op
template <typename RNG>
void maybe_set_next_normal_sample(RNG * /*generator*/, const void * /*cache*/) {}

// ============================================================
// normal_distribution — 正态分布函子（Box-Muller）
// ============================================================
// 仿 PyTorch at::normal_distribution
// （aten/src/ATen/core/DistributionsHelper.h:172）
//
// Box-Muller 每次产生两个正态样本，第二个缓存到 Generator。
//
template <typename T>
struct normal_distribution {//用struct能记录
    normal_distribution(T mean, T stdv) : mean_(mean), stdv_(stdv) {
        assert(stdv >= 0);
    }
    //
    normal_distribution(const normal_distribution &) = default;
    normal_distribution &operator=(const normal_distribution &) = delete;

    template <typename RNG>
    T operator()(RNG *generator) const {
        T ret;
        // 先检查缓存
        if (maybe_get_next_normal_sample(generator, &ret)) {
            return ret * stdv_ + mean_;
        }

        // 生成两个 [0, 1) 均匀样本
        uniform_real_distribution<T> uniform(static_cast<T>(0), static_cast<T>(1));
        const T u1 = uniform(generator);
        const T u2 = uniform(generator);

        // Box-Muller 变换
        const T r = std::sqrt(static_cast<T>(-2.0) * std::log1p(-u2));
        constexpr T kTwoPi = static_cast<T>(2.0 * M_PI);
        const T theta = kTwoPi * u1;
        const T sample = r * std::sin(theta);

        // 缓存第二个样本
        maybe_set_next_normal_sample(generator, &sample);

        ret = r * std::cos(theta);
        return ret * stdv_ + mean_;
    }

private:
    T mean_;
    T stdv_;
};

} // namespace
} // namespace infini_train

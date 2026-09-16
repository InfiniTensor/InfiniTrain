#pragma once

// Host-side uniform and normal distributions for generators exposing Random()
// and Random64(). Box-Muller's second sample is cached when supported by the generator.

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>

#include "glog/logging.h"

namespace infini_train::common::cpu {

template <typename T> struct UniformRealDistribution {
    UniformRealDistribution(T from, T to) : from_(from), to_(to) {
        CHECK_LE(from, to);
        CHECK_LE(to - from, std::numeric_limits<T>::max());
    }

    UniformRealDistribution(const UniformRealDistribution &) = default;
    UniformRealDistribution &operator=(const UniformRealDistribution &) = delete;

    template <typename RNG> T operator()(RNG *generator) const {
        if constexpr (std::is_same_v<T, double>) {
            return Transform(generator->Random64());
        } else {
            return Transform(generator->Random());
        }
    }

private:
    T from_;
    T to_;

    template <typename V> T Transform(V val) const {
        constexpr auto kMask = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
        constexpr auto kDivisor = static_cast<T>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
        T x = (val & kMask) * kDivisor;
        return x * (to_ - from_) + from_;
    }
};

template <typename RNG, typename = decltype(&RNG::next_double_normal_sample),
          typename = decltype(&RNG::set_next_double_normal_sample)>
bool MaybeGetNextNormalSample(RNG *generator, double *ret) {
    const auto sample = generator->next_double_normal_sample();
    if (!sample.has_value()) {
        return false;
    }
    *ret = sample.value();
    generator->set_next_double_normal_sample(std::nullopt);
    return true;
}

template <typename RNG, typename = decltype(&RNG::next_float_normal_sample),
          typename = decltype(&RNG::set_next_float_normal_sample)>
bool MaybeGetNextNormalSample(RNG *generator, float *ret) {
    const auto sample = generator->next_float_normal_sample();
    if (!sample.has_value()) {
        return false;
    }
    *ret = sample.value();
    generator->set_next_float_normal_sample(std::nullopt);
    return true;
}

// Fallback: RNG without cache support never has a cached sample.
template <typename RNG> bool MaybeGetNextNormalSample(RNG * /*generator*/, void * /*ret*/) { return false; }

template <typename RNG, typename = decltype(&RNG::set_next_double_normal_sample)>
void MaybeSetNextNormalSample(RNG *generator, const double *cache) {
    generator->set_next_double_normal_sample(*cache);
}

template <typename RNG, typename = decltype(&RNG::set_next_float_normal_sample)>
void MaybeSetNextNormalSample(RNG *generator, const float *cache) {
    generator->set_next_float_normal_sample(*cache);
}

// Fallback: RNG without cache support discards the second sample.
template <typename RNG> void MaybeSetNextNormalSample(RNG * /*generator*/, const void * /*cache*/) {}

template <typename T> struct NormalDistribution {
    NormalDistribution(T mean, T stdv) : mean_(mean), stdv_(stdv) { CHECK_GE(stdv, static_cast<T>(0)); }

    NormalDistribution(const NormalDistribution &) = default;
    NormalDistribution &operator=(const NormalDistribution &) = delete;

    template <typename RNG> T operator()(RNG *generator) const {
        T ret;
        if (MaybeGetNextNormalSample(generator, &ret)) {
            return ret * stdv_ + mean_;
        }

        UniformRealDistribution<T> uniform(static_cast<T>(0), static_cast<T>(1));
        const T u1 = uniform(generator);
        const T u2 = uniform(generator);

        const T r = std::sqrt(static_cast<T>(-2.0) * std::log1p(-u2));
        constexpr T kTwoPi = static_cast<T>(2.0 * std::numbers::pi_v<double>);
        const T theta = kTwoPi * u1;
        const T sample = r * std::sin(theta);

        MaybeSetNextNormalSample(generator, &sample);

        ret = r * std::cos(theta);
        return ret * stdv_ + mean_;
    }

private:
    T mean_;
    T stdv_;
};

} // namespace infini_train::common::cpu

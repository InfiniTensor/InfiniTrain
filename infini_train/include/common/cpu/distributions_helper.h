#pragma once

// Host-side uniform and normal distributions for generators exposing random()
// and random64(). Box-Muller's second sample is cached when supported by the generator.

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>

#include "glog/logging.h"

namespace infini_train::common::cpu {

template <typename T> struct uniform_real_distribution {
    uniform_real_distribution(T from, T to) : from_(from), to_(to) {
        CHECK_LE(from, to);
        CHECK_LE(to - from, std::numeric_limits<T>::max());
    }

    uniform_real_distribution(const uniform_real_distribution &) = default;
    uniform_real_distribution &operator=(const uniform_real_distribution &) = delete;

    template <typename RNG> T operator()(RNG *generator) const {
        if constexpr (std::is_same_v<T, double>) {
            return transform(generator->random64());
        } else {
            return transform(generator->random());
        }
    }

private:
    T from_;
    T to_;

    template <typename V> T transform(V val) const {
        constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
        constexpr auto DIVISOR = static_cast<T>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
        T x = (val & MASK) * DIVISOR;
        return x * (to_ - from_) + from_;
    }
};

template <typename RNG, typename = decltype(&RNG::next_double_normal_sample),
          typename = decltype(&RNG::set_next_double_normal_sample)>
bool maybe_get_next_normal_sample(RNG *generator, double *ret) {
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
bool maybe_get_next_normal_sample(RNG *generator, float *ret) {
    const auto sample = generator->next_float_normal_sample();
    if (!sample.has_value()) {
        return false;
    }
    *ret = sample.value();
    generator->set_next_float_normal_sample(std::nullopt);
    return true;
}

// Fallback: RNG without cache support never has a cached sample.
template <typename RNG> bool maybe_get_next_normal_sample(RNG * /*generator*/, void * /*ret*/) { return false; }

template <typename RNG, typename = decltype(&RNG::set_next_double_normal_sample)>
void maybe_set_next_normal_sample(RNG *generator, const double *cache) {
    generator->set_next_double_normal_sample(*cache);
}

template <typename RNG, typename = decltype(&RNG::set_next_float_normal_sample)>
void maybe_set_next_normal_sample(RNG *generator, const float *cache) {
    generator->set_next_float_normal_sample(*cache);
}

// Fallback: RNG without cache support discards the second sample.
template <typename RNG> void maybe_set_next_normal_sample(RNG * /*generator*/, const void * /*cache*/) {}

template <typename T> struct normal_distribution {
    normal_distribution(T mean, T stdv) : mean_(mean), stdv_(stdv) { CHECK_GE(stdv, static_cast<T>(0)); }

    normal_distribution(const normal_distribution &) = default;
    normal_distribution &operator=(const normal_distribution &) = delete;

    template <typename RNG> T operator()(RNG *generator) const {
        T ret;
        if (maybe_get_next_normal_sample(generator, &ret)) {
            return ret * stdv_ + mean_;
        }

        uniform_real_distribution<T> uniform(static_cast<T>(0), static_cast<T>(1));
        const T u1 = uniform(generator);
        const T u2 = uniform(generator);

        const T r = std::sqrt(static_cast<T>(-2.0) * std::log1p(-u2));
        constexpr T kTwoPi = static_cast<T>(2.0 * std::numbers::pi_v<double>);
        const T theta = kTwoPi * u1;
        const T sample = r * std::sin(theta);

        maybe_set_next_normal_sample(generator, &sample);

        ret = r * std::cos(theta);
        return ret * stdv_ + mean_;
    }

private:
    T mean_;
    T stdv_;
};

} // namespace infini_train::common::cpu

#pragma once

#include <cstdint>

namespace infini_train::nn::parallel::function {
enum class ReduceOpType : int8_t {
    kSum,
    kProd,
    kMin,
    kMax,
    kAvg,
};

} // namespace infini_train::nn::parallel::function

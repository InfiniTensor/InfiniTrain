#pragma once

#include <cstdint>

namespace infini_train::nn::parallel::comm {
enum class ReduceOpType : int8_t {
    kSum,
    kProd,
    kMin,
    kMax,
    kAvg,
};

} // namespace infini_train::nn::parallel::comm

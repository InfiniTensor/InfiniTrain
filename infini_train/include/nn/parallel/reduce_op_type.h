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

struct AllreduceOptions {
    ReduceOpType reduce_op_type = ReduceOpType::kSum;
    bool async_op = false;
};

} // namespace infini_train::nn::parallel::function

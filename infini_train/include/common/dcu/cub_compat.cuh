#pragma once

#include <functional>
#include <hipcub/hipcub.hpp>

namespace infini_train::kernels::dcu {

using CubSumOp = hipcub::Sum;
using CubMaxOp = hipcub::Max;
using CubMinOp = hipcub::Min;

} // namespace infini_train::kernels::dcu

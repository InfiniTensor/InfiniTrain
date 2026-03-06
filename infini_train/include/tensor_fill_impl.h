#pragma once

#include <cstring>

#include "infini_train/include/common/cpu/common_cpu.h"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype_dispatch.h"

namespace infini_train {

inline void Tensor::Fill(double value) {
    auto device = GetDevice();
    core::DeviceGuard guard(device);

    uint64_t storage = 0;

    DispatchFunc<INFINI_ALL_TYPES>(Dtype(), [&storage, value]<typename TargetT>() {
        TargetT casted_value = common::cpu::Cast<TargetT>(static_cast<float>(value));
        std::memcpy((void *)(&storage), &casted_value, sizeof(TargetT));
    });

    auto kernel = Dispatcher::Instance().GetKernel({device.type(), "Fill"});
    kernel.Call<void>(shared_from_this(), static_cast<void *>(&storage));
}

} // namespace infini_train

#pragma once

#include <string_view>
#include <unordered_map>

#include "common/common.h"
#include "datatype.h"
#include "device.h"
#include "tensor.h"

#ifdef USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace infini_train {
namespace {
inline std::string_view GetBaseOpName(std::string_view op) {
    constexpr std::string_view forward_suffix = "Forward";
    constexpr std::string_view backward_suffix = "Backward";

    // Check for "Forward" suffix
    if (op.size() >= forward_suffix.size()) {
        const auto suffix_pos = op.size() - forward_suffix.size();
        if (op.substr(suffix_pos) == forward_suffix) {
            return op.substr(0, suffix_pos);
        }
    }

    // Check for "Backward" suffix
    if (op.size() >= backward_suffix.size()) {
        const auto suffix_pos = op.size() - backward_suffix.size();
        if (op.substr(suffix_pos) == backward_suffix) {
            return op.substr(0, suffix_pos);
        }
    }

    return op;
}
}; // namespace

enum class CastPolicy : uint8_t {
    kLowerPrecision = 0,
    kFP32,
    kPromoteWidest,
    kCount,
};

inline constexpr std::array kLowerPrecisionOps = {"Matmul", "Linear"};
inline constexpr std::array kFP32Ops = {"Sin",   "Cos",   "Tan",   "Asin", "Acos", "Atan", "Sinh",  "Cosh", "Tanh",
                                        "Asinh", "Acosh", "Atanh", "Exp",  "Log",  "Sqrt", "Rsqrt", "Pow"};

inline const std::unordered_map<std::string_view, CastPolicy> kOpCastPolicyMap = {
    {"Matmul", CastPolicy::kLowerPrecision},
    {"Linear", CastPolicy::kLowerPrecision},
    {"Sin", CastPolicy::kFP32},
    {"Cos", CastPolicy::kFP32},
    {"Tan", CastPolicy::kFP32},
    {"Asin", CastPolicy::kFP32},
    {"Acos", CastPolicy::kFP32},
    {"Atan", CastPolicy::kFP32},
    {"Sinh", CastPolicy::kFP32},
    {"Cosh", CastPolicy::kFP32},
    {"Tanh", CastPolicy::kFP32},
    {"Asinh", CastPolicy::kFP32},
    {"Acosh", CastPolicy::kFP32},
    {"Atanh", CastPolicy::kFP32},
    {"Tanh", CastPolicy::kFP32},
    {"Exp", CastPolicy::kFP32},
    {"Log", CastPolicy::kFP32},
    {"Sqrt", CastPolicy::kFP32},
    {"Reciprocal", CastPolicy::kFP32},
    {"Rsqrt", CastPolicy::kFP32},
    {"Prod", CastPolicy::kFP32},
    {"Pow", CastPolicy::kFP32},
    {"CrossEntropy", CastPolicy::kFP32},
    {"Layernorm", CastPolicy::kFP32},
};

inline constexpr std::array<DataType, static_cast<size_t>(DeviceType::kCount)> kDeviceDefaultDtype = {
    DataType::kBFLOAT16, // CPU
    DataType::kFLOAT16,  // CUDA.
};

// Thread-local context to track autocast state
struct AutocastContext {
    bool enabled = false;                          // Whether autocast is active in the current thread
    DeviceType device_type = DeviceType::kCPU;     // Target device type (CPU/GPU)
    DataType autocast_dtype = DataType::kBFLOAT16; // The data type used for autocasting

    template <typename... ArgsT> void Autocast(std::pair<DeviceType, std::string> key, ArgsT &...args) {
        if (!enabled) {
            return;
        }

        if (device_type != key.first) {
            LOG_LOC(FATAL, "In AutocastContext::Autocast(): the AutocastContext device_type is different from the one "
                           "passed in. Don't know what to do.");
            return;
        }

        auto map_it = kOpCastPolicyMap.find(GetBaseOpName(key.second));
        if (map_it == kOpCastPolicyMap.end()) {
            return;
        }

        CastPolicy policy = map_it->second;

        auto is_floating_point = [](DataType dtype) -> bool {
            return dtype == DataType::kFLOAT32 || dtype == DataType::kFLOAT16 || dtype == DataType::kBFLOAT16;
        };

        auto get_target_dtype = [&]() {
            switch (policy) {
            case CastPolicy::kLowerPrecision:
                return autocast_dtype;
            case CastPolicy::kFP32:
                return DataType::kFLOAT32;
            case CastPolicy::kPromoteWidest:
                throw std::runtime_error("kPromoteWidest not implemented");
            default:
                throw std::runtime_error("Invalid cast policy");
            }
        };

        // Process each argument
        auto cast_arg = [&](auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::shared_ptr<Tensor>>) {
                if (arg) {
                    DataType current_dtype = arg->Dtype();
                    if (is_floating_point(current_dtype)) {
                        DataType target_dtype = get_target_dtype();
                        if (current_dtype != target_dtype) {
                            arg = std::make_shared<Tensor>(arg->To(target_dtype));
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, Tensor>) {
                DataType current_dtype = arg.Dtype();
                if (is_floating_point(current_dtype)) {
                    DataType target_dtype = get_target_dtype();
                    if (current_dtype != target_dtype) {
                        arg = arg.To(target_dtype);
                    }
                }
            }
        };

        // Apply casting to each argument
        (cast_arg(args), ...);
    }
};

// Global thread-local storage for autocast context
inline thread_local AutocastContext tls_autocast_context;

// RAII guard to enable/disable autocast in a scope
class AutocastGuard {
public:
    AutocastGuard(DeviceType device_type, DataType autocast_dtype) {
        saved_context_ = tls_autocast_context;
        tls_autocast_context.enabled = true;
        tls_autocast_context.device_type = device_type;
        tls_autocast_context.autocast_dtype = autocast_dtype;
    }

    AutocastGuard(DeviceType device_type)
        : AutocastGuard(device_type, kDeviceDefaultDtype[static_cast<size_t>(device_type)]) {}

    // Disable autocast (restore previous state)
    ~AutocastGuard() { tls_autocast_context = saved_context_; }

    AutocastGuard(const AutocastGuard &) = delete;
    AutocastGuard &operator=(const AutocastGuard &) = delete;

    void Enable() const { tls_autocast_context.enabled = true; }
    void Disable() const { tls_autocast_context.enabled = false; }
    bool IsEnabled() const { return tls_autocast_context.enabled; }

private:
    AutocastContext saved_context_;
};

} // namespace infini_train

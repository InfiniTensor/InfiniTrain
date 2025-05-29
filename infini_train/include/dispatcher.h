#pragma once

#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif

/**
 * General Utility Macros
 */
#define EXPAND(X) X
// This macro lets you pass an arbitrary expression that may contain internal
// commas to another macro without having the commas causing the expression
// to be interpreted as being multiple arguments
// Basically an alternative for __VA_OPTS__ before C++20
// ref: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch_v2.h
#define WRAP(...) __VA_ARGS__
#define CAT(a, b) CAT_(a, b)
#define CAT_(a, b) a##b

/**
 * Data Type Macros
 * Defines common categories of data types for dispatching
 */
#define INFINI_FLOATING_TYPES DataType::kFLOAT32, DataType::kFLOAT64
#define INFINI_REDUCED_FLOATING_TYPES DataType::kFLOAT16, DataType::kBFLOAT16
#define INFINI_ALL_FLOATING_TYPES EXPAND(INFINI_FLOATING_TYPES), EXPAND(INFINI_REDUCED_FLOATING_TYPES)
#define INFINI_SIGNED_INTEGRAL_TYPES DataType::kINT8, DataType::kINT16, DataType::kINT32, DataType::kINT64
#define INFINI_UNSIGNED_INTEGRAL_TYPES DataType::kUINT8, DataType::kUINT16, DataType::kUINT32, DataType::kUINT64
#define INFINI_ALL_INTEGRAL_TYPES EXPAND(INFINI_SIGNED_INTEGRAL_TYPES), EXPAND(INFINI_UNSIGNED_INTEGRAL_TYPES)
#define INFINI_ALL_TYPES EXPAND(INFINI_ALL_FLOATING_TYPES), EXPAND(INFINI_ALL_INTEGRAL_TYPES)

/**
 * Dispatch Macros
 */
#define DISPATCH_WITH_DEFAULT(DTYPE_EXPR, BODY, DEFAULT_BODY, ...)                                                     \
    switch (DTYPE_EXPR) {                                                                                              \
        CAT(DISPATCH_CASE_, PP_NARG(__VA_ARGS__))(__VA_ARGS__, WRAP(BODY)) default : { WRAP(DEFAULT_BODY); }           \
    }

// dispatch with switch and arbitrary number of cases
#define DISPATCH(DTYPE_EXPR, BODY, ...)                                                                                \
    DISPATCH_WITH_DEFAULT(                                                                                             \
        DTYPE_EXPR, WRAP(BODY),                                                                                        \
        EXPAND(LOG(FATAL) << "Unsupported data type at " << __FILE__ << ":" << __LINE__; return nullptr;),             \
        __VA_ARGS__)

// dispatch a single case
#define DISPATCH_CASE(BODY, ...) CAT(DISPATCH_CASE_, PP_NARG(__VA_ARGS__))(__VA_ARGS__, WRAP(BODY))

// Helper macros to count the number of arguments
#define PP_NARG(...) PP_NARG_(__VA_ARGS__, PP_RSEQ_N())
#define PP_NARG_(...) PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22,  \
                 _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42,   \
                 _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62,   \
                 _63, N, ...)                                                                                          \
    N
#define PP_RSEQ_N()                                                                                                    \
    63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36,    \
        35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,  \
        7, 6, 5, 4, 3, 2, 1, 0

// Macros to generate case labels
// Should have up to number of DataType cases (currently 12)
#define DISPATCH_CASE_1(T1, BODY)                                                                                      \
    case T1: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_2(T1, T2, BODY)                                                                                  \
    case T1:                                                                                                           \
    case T2: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_3(T1, T2, T3, BODY)                                                                              \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_4(T1, T2, T3, T4, BODY)                                                                          \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_5(T1, T2, T3, T4, T5, BODY)                                                                      \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4:                                                                                                           \
    case T5: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_6(T1, T2, T3, T4, T5, T6, BODY)                                                                  \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4:                                                                                                           \
    case T5:                                                                                                           \
    case T6: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_7(T1, T2, T3, T4, T5, T6, T7, BODY)                                                              \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4:                                                                                                           \
    case T5:                                                                                                           \
    case T6:                                                                                                           \
    case T7: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_8(T1, T2, T3, T4, T5, T6, T7, T8, BODY)                                                          \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4:                                                                                                           \
    case T5:                                                                                                           \
    case T6:                                                                                                           \
    case T7:                                                                                                           \
    case T8: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_9(T1, T2, T3, T4, T5, T6, T7, T8, T9, BODY)                                                      \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4:                                                                                                           \
    case T5:                                                                                                           \
    case T6:                                                                                                           \
    case T7:                                                                                                           \
    case T8:                                                                                                           \
    case T9: {                                                                                                         \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_10(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, BODY)                                                \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4:                                                                                                           \
    case T5:                                                                                                           \
    case T6:                                                                                                           \
    case T7:                                                                                                           \
    case T8:                                                                                                           \
    case T9:                                                                                                           \
    case T10: {                                                                                                        \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_11(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, BODY)                                           \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4:                                                                                                           \
    case T5:                                                                                                           \
    case T6:                                                                                                           \
    case T7:                                                                                                           \
    case T8:                                                                                                           \
    case T9:                                                                                                           \
    case T10:                                                                                                          \
    case T11: {                                                                                                        \
        BODY break;                                                                                                    \
    }

#define DISPATCH_CASE_12(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, BODY)                                      \
    case T1:                                                                                                           \
    case T2:                                                                                                           \
    case T3:                                                                                                           \
    case T4:                                                                                                           \
    case T5:                                                                                                           \
    case T6:                                                                                                           \
    case T7:                                                                                                           \
    case T8:                                                                                                           \
    case T9:                                                                                                           \
    case T10:                                                                                                          \
    case T11:                                                                                                          \
    case T12: {                                                                                                        \
        BODY break;                                                                                                    \
    }

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) : func_ptr_(reinterpret_cast<void *>(func)) {}

    // TODO(dcj): support auto-deduction of return type and parameter types
    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
#ifdef PROFILE_MODE
        const auto &ctx = GetProfileContext();
        Profiler::Instance().StartRecord(ctx.name, ctx.device);
#endif

        using FuncT = RetT (*)(ArgsT...);
        auto fn = reinterpret_cast<FuncT>(func_ptr_);

        if constexpr (std::is_void_v<RetT>) {
            fn(std::forward<ArgsT>(args)...);

#ifdef PROFILE_MODE
            Profiler::Instance().EndRecord(ctx.name, ctx.device);
#endif
            return;
        } else {
            RetT ret = fn(std::forward<ArgsT>(args)...);

#ifdef PROFILE_MODE
            Profiler::Instance().EndRecord(ctx.name, ctx.device);
#endif
            return ret;
        }
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    const KernelFunction &GetKernel(KeyT key) const {
        CHECK(key_to_kernel_map_.contains(key))
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
#ifdef PROFILE_MODE
        SetProfileContext(key.second, key.first);
#endif
        return key_to_kernel_map_.at(key);
    }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        CHECK(!key_to_kernel_map_.contains(key))
            << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
        key_to_kernel_map_.emplace(key, kernel);
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static const bool _##kernel_name##_registered##__COUNTER__ = []() {                                                \
        infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                            \
        return true;                                                                                                   \
    }();

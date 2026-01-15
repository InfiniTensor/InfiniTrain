#pragma once

#include <format>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/common/common.h"
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
#define INFINI_8_BIT_TYPES DataType::kINT8, DataType::kUINT8
#define INFINI_16_BIT_TYPES DataType::kINT16, DataType::kUINT16, DataType::kFLOAT16, DataType::kBFLOAT16
#define INFINI_32_BIT_TYPES DataType::kINT32, DataType::kUINT32, DataType::kFLOAT32
#define INFINI_64_BIT_TYPES DataType::kINT64, DataType::kUINT64, DataType::kFLOAT64

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

template <DataType... DTypes> struct DataTypeList {};

template <DataType Dtype, typename List> struct IsDataTypeInList;

template <DataType Dtype, DataType... DTypes>
struct IsDataTypeInList<Dtype, DataTypeList<DTypes...>> : std::disjunction<std::bool_constant<Dtype == DTypes>...> {};

template <DataType Dtype, typename List>
inline constexpr bool IsDataTypeInList_v = IsDataTypeInList<Dtype, List>::value;

// function to check if a type is in a list of types
template <typename T, typename... Ts> inline constexpr bool IsTypeInList = (std::is_same_v<T, Ts> || ...);

/**
 * @brief Dispatches a functor call based on runtime DataType, restricted to specified allowed types.
 *
 * This function:
 * 1. Maps runtime DataType to compile-time C++ types using TypeMap_t
 * 2. Only processes types specified in AllowedDTypes template parameter
 * 3. Calls functor with resolved type and forwarded arguments
 *
 * @tparam AllowedDTypes List of DataType enums to support
 * @param dtype Runtime data type to dispatch
 * @param func Templated functor to call (must accept operator()<T>)
 * @param context_identifier Optional string for context in error messages
 * @param args Arguments to be forwarded to the functor
 *
 * Behavior:
 * - For allowed types: Instantiates functor with mapped C++ type
 * - For disallowed and unknown types: Logs error and returns
 *
 * @see TypeMap for DataType to C++ type mapping
 */
template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    switch (dtype) {

#define CASE_FOR_TYPE(DType)                                                                                           \
    case DType: {                                                                                                      \
        if constexpr (IsTypeInList<TypeMap_t<DType>, TypeMap_t<AllowedDTypes>...>) {                                   \
            return std::forward<Functor>(func).template operator()<TypeMap_t<DType>>(std::forward<Args>(args)...);     \
        } else {                                                                                                       \
            break;                                                                                                     \
        }                                                                                                              \
    }

        CASE_FOR_TYPE(DataType::kUINT8)
        CASE_FOR_TYPE(DataType::kINT8)
        CASE_FOR_TYPE(DataType::kUINT16)
        CASE_FOR_TYPE(DataType::kINT16)
        CASE_FOR_TYPE(DataType::kUINT32)
        CASE_FOR_TYPE(DataType::kINT32)
        CASE_FOR_TYPE(DataType::kUINT64)
        CASE_FOR_TYPE(DataType::kINT64)
        CASE_FOR_TYPE(DataType::kFLOAT32)
        CASE_FOR_TYPE(DataType::kFLOAT64)
#ifdef USE_CUDA
        CASE_FOR_TYPE(DataType::kBFLOAT16)
        CASE_FOR_TYPE(DataType::kFLOAT16)
#endif
#undef CASE_FOR_TYPE
    }
    LOG_UNSUPPORTED_DTYPE(dtype, context_identifier);
    // prevent the compiler warning about control reaching the end of non-void function
    std::abort();
}

namespace {
/**
 * @brief Responsible for resolving a list of data types and invoking a functor with the corresponding C++ types.
 *
 * @tparam index            Current index in the `dtypes` vector.
 * @tparam AllowedListTuple Tuple of allowed `DataType` sets per dispatch level.
 * @tparam ResolvedTypes    Accumulated resolved C++ types.
 */
template <size_t index, typename AllowedListTuple, typename... ResolvedTypes> struct DtypeDispatcher {

    /**
     * @brief Dispatches based on runtime data types and invokes the functor with resolved C++ types.
     *
     * Recursively matches each `DataType` in `dtypes` against the corresponding allowed list in
     * `AllowedListTuple`. For each match, maps the `DataType` to a C++ type using `TypeMap_t`.
     * Once all types are resolved, invokes the functor.
     *
     * @param dtypes              Vector of runtime data types to dispatch on.
     * @param func                Functor to invoke with resolved template types.
     * @param context_identifier  String used for logging or error context.
     * @param args                Additional arguments forwarded to the functor.
     * @return Result of invoking the functor with resolved types and forwarded arguments.
     */
    template <typename Functor, typename... Args>
    static auto call(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier,
                     Args &&...args) {
        constexpr size_t num_lists = std::tuple_size_v<AllowedListTuple>;

        if constexpr (index == num_lists) {
            // Base case: All types resolved, invoke the functor
            return std::forward<Functor>(func).template operator()<ResolvedTypes...>(std::forward<Args>(args)...);
        } else {
            // Recursive case: Resolve the next type
            using CurrentList = std::tuple_element_t<index, AllowedListTuple>;
            DataType dtype = dtypes[index];

            switch (dtype) {
#define CASE_FOR_TYPE(DType)                                                                                           \
    case DType:                                                                                                        \
        if constexpr (IsDataTypeInList_v<DType, CurrentList>) {                                                        \
            using T = TypeMap_t<DType>;                                                                                \
            return DtypeDispatcher<index + 1, AllowedListTuple, ResolvedTypes..., T>::call(                            \
                dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);                 \
        } else {                                                                                                       \
            break;                                                                                                     \
        }

                CASE_FOR_TYPE(DataType::kUINT8)
                CASE_FOR_TYPE(DataType::kINT8)
                CASE_FOR_TYPE(DataType::kUINT16)
                CASE_FOR_TYPE(DataType::kINT16)
                CASE_FOR_TYPE(DataType::kUINT32)
                CASE_FOR_TYPE(DataType::kINT32)
                CASE_FOR_TYPE(DataType::kUINT64)
                CASE_FOR_TYPE(DataType::kINT64)
                CASE_FOR_TYPE(DataType::kFLOAT32)
                CASE_FOR_TYPE(DataType::kFLOAT64)
#ifdef USE_CUDA
                CASE_FOR_TYPE(DataType::kBFLOAT16)
                CASE_FOR_TYPE(DataType::kFLOAT16)
#endif
#undef CASE_FOR_TYPE
            }
            LOG_UNSUPPORTED_DTYPE(dtype, context_identifier);
            // prevent the compiler warning about control reaching the end of non-void function
            std::abort();
        }
    }
};
} // namespace

/**
 * @brief Dispatches a functor based on a list of runtime data types.
 *
 * Given a vector of `DataType` values and corresponding allowed type lists, this function resolves
 * each data type to its mapped C++ type using `TypeMap_t`, then invokes the provided functor with
 * those types as template parameters.
 *
 * @tparam AllowedTypeLists   Variadic list of allowed data type sets per dispatch level.
 * @tparam Functor            Callable object with a templated call operator.
 * @tparam Args               Additional arguments to forward to the functor.
 *
 * @param dtypes              Vector of runtime data types to dispatch on.
 * @param func                Functor to invoke after resolving types.
 * @param context_identifier  Optional context string for error reporting/logging.
 * @param args                Additional arguments to pass to the functor.
 * @return Result of invoking the functor with resolved template types and arguments.
 *
 * Example functor using a templated lambda: [=]<typename T1, typename T2>() { ... }
 */
template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                  Args &&...args) {
    constexpr size_t num_lists = sizeof...(AllowedTypeLists);
    if (dtypes.size() != num_lists) {
        LOG(FATAL) << std::format("DispatchFunc expects {} dtypes, but only got {} in {}", num_lists, dtypes.size(),
                                  context_identifier);
        std::abort();
    }

    using AllowedListTuple = std::tuple<AllowedTypeLists...>;
    return DtypeDispatcher<0, AllowedListTuple>::call(dtypes, std::forward<Functor>(func), context_identifier,
                                                      std::forward<Args>(args)...);
}

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
    using KeyT = std::pair<Device::DeviceType, std::string>;

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

    template <typename RetT, class... ArgsT> RetT Call(KeyT key, ArgsT... args) const {
        auto kernel = this->GetKernel(key);
        tls_autocast_context.Autocast(key, args...);
        return kernel.Call<RetT>(std::forward<ArgsT>(args)...);
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

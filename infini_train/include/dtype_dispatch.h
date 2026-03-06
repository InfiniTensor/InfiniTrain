#pragma once

#include <format>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/common.h"
#include "infini_train/include/datatype.h"

// ---------------------------------------------------------------------------
// Dispatch macros for dtype switch statements
// ---------------------------------------------------------------------------

#define LOG_UNSUPPORTED_DTYPE(DTYPE, CONTEXT_IDENTIFIER)                                                               \
    LOG_LOC(FATAL, std::string(CONTEXT_IDENTIFIER)                                                                     \
                       + ": Unsupported data type: " + kDataTypeToDesc.at(static_cast<infini_train::DataType>(dtype)))

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

#define DISPATCH_CASE(BODY, ...) CAT(DISPATCH_CASE_, PP_NARG(__VA_ARGS__))(__VA_ARGS__, WRAP(BODY))

#define DISPATCH_WITH_DEFAULT(DTYPE_EXPR, BODY, DEFAULT_BODY, ...)                                                     \
    switch (DTYPE_EXPR) {                                                                                              \
        CAT(DISPATCH_CASE_, PP_NARG(__VA_ARGS__))(__VA_ARGS__, WRAP(BODY)) default : { WRAP(DEFAULT_BODY); }           \
    }

#define DISPATCH(DTYPE_EXPR, BODY, ...)                                                                                \
    DISPATCH_WITH_DEFAULT(                                                                                             \
        DTYPE_EXPR, WRAP(BODY),                                                                                        \
        EXPAND(LOG(FATAL) << "Unsupported data type at " << __FILE__ << ":" << __LINE__; return nullptr;),             \
        __VA_ARGS__)

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

/**
 * Data Type Macros
 * Defines common categories of data types for dispatching
 */
#define INFINI_FLOATING_TYPES DataType::kFLOAT32, DataType::kFLOAT64
#define INFINI_REDUCED_FLOATING_TYPES DataType::kFLOAT16, DataType::kBFLOAT16
#define INFINI_ALL_FLOATING_TYPES INFINI_FLOATING_TYPES, INFINI_REDUCED_FLOATING_TYPES
#define INFINI_SIGNED_INTEGRAL_TYPES DataType::kINT8, DataType::kINT16, DataType::kINT32, DataType::kINT64
#define INFINI_UNSIGNED_INTEGRAL_TYPES DataType::kUINT8, DataType::kUINT16, DataType::kUINT32, DataType::kUINT64
#define INFINI_ALL_INTEGRAL_TYPES INFINI_SIGNED_INTEGRAL_TYPES, INFINI_UNSIGNED_INTEGRAL_TYPES
#define INFINI_ALL_TYPES INFINI_ALL_FLOATING_TYPES, INFINI_ALL_INTEGRAL_TYPES
#define INFINI_8_BIT_TYPES DataType::kINT8, DataType::kUINT8
#define INFINI_16_BIT_TYPES DataType::kINT16, DataType::kUINT16, DataType::kFLOAT16, DataType::kBFLOAT16
#define INFINI_32_BIT_TYPES DataType::kINT32, DataType::kUINT32, DataType::kFLOAT32
#define INFINI_64_BIT_TYPES DataType::kINT64, DataType::kUINT64, DataType::kFLOAT64

template <DataType... DTypes> struct DataTypeList {};

template <DataType Dtype, typename List> struct IsDataTypeInList;

template <DataType Dtype, DataType... DTypes>
struct IsDataTypeInList<Dtype, DataTypeList<DTypes...>> : std::disjunction<std::bool_constant<Dtype == DTypes>...> {};

template <DataType Dtype, typename List>
inline constexpr bool IsDataTypeInList_v = IsDataTypeInList<Dtype, List>::value;

template <typename T, typename... Ts> inline constexpr bool IsTypeInList = (std::is_same_v<T, Ts> || ...);

template <template <DataType> class TypeMap, DataType DType> using MappedType_t = typename TypeMap<DType>::type;

// -----------------------------------------------------------------------------
// Generic single-dtype dispatch by custom type map
// -----------------------------------------------------------------------------
template <template <DataType> class TypeMap, DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchByTypeMap(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    switch (dtype) {
#define CASE_FOR_TYPE(DType)                                                                                           \
    case DType: {                                                                                                      \
        if constexpr (IsTypeInList<MappedType_t<TypeMap, DType>, MappedType_t<TypeMap, AllowedDTypes>...>) {           \
            return std::forward<Functor>(func).template operator()<MappedType_t<TypeMap, DType>>(                      \
                std::forward<Args>(args)...);                                                                          \
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
        CASE_FOR_TYPE(DataType::kBFLOAT16)
        CASE_FOR_TYPE(DataType::kFLOAT16)
#undef CASE_FOR_TYPE
    }

    LOG_UNSUPPORTED_DTYPE(dtype, context_identifier);
    std::abort();
}

namespace detail {

template <template <DataType> class TypeMap, size_t Index, typename AllowedListTuple, typename... ResolvedTypes>
struct TypeMapDispatcher {
    template <typename Functor, typename... Args>
    static auto call(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier,
                     Args &&...args) {
        constexpr size_t kNumLists = std::tuple_size_v<AllowedListTuple>;

        if constexpr (Index == kNumLists) {
            return std::forward<Functor>(func).template operator()<ResolvedTypes...>(std::forward<Args>(args)...);
        } else {
            using CurrentList = std::tuple_element_t<Index, AllowedListTuple>;
            const DataType dtype = dtypes[Index];

            switch (dtype) {
#define CASE_FOR_TYPE(DType)                                                                                           \
    case DType:                                                                                                        \
        if constexpr (IsDataTypeInList_v<DType, CurrentList>) {                                                        \
            using T = MappedType_t<TypeMap, DType>;                                                                    \
            return TypeMapDispatcher<TypeMap, Index + 1, AllowedListTuple, ResolvedTypes..., T>::call(                 \
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
                CASE_FOR_TYPE(DataType::kBFLOAT16)
                CASE_FOR_TYPE(DataType::kFLOAT16)
#undef CASE_FOR_TYPE
            }

            LOG_UNSUPPORTED_DTYPE(dtype, context_identifier);
            std::abort();
        }
    }
};

} // namespace detail

// -----------------------------------------------------------------------------
// Generic multi-dtype dispatch by custom type map
// -----------------------------------------------------------------------------
template <template <DataType> class TypeMap, typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchByTypeMap(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                       Args &&...args) {
    constexpr size_t kNumLists = sizeof...(AllowedTypeLists);

    if (dtypes.size() != kNumLists) {
        LOG(FATAL) << std::format("DispatchByTypeMap expects {} dtypes, but only got {} in {}", kNumLists,
                                  dtypes.size(), context_identifier);
        std::abort();
    }

    using AllowedListTuple = std::tuple<AllowedTypeLists...>;
    return detail::TypeMapDispatcher<TypeMap, 0, AllowedListTuple>::call(
        dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

// -----------------------------------------------------------------------------
// Default framework dispatch using TypeMap
// -----------------------------------------------------------------------------
template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    return DispatchByTypeMap<TypeMap, AllowedDTypes...>(dtype, std::forward<Functor>(func), context_identifier,
                                                        std::forward<Args>(args)...);
}

template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                  Args &&...args) {
    return DispatchByTypeMap<TypeMap, AllowedTypeLists...>(dtypes, std::forward<Functor>(func), context_identifier,
                                                           std::forward<Args>(args)...);
}

} // namespace infini_train

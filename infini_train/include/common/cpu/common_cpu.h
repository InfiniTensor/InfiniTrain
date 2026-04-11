#pragma once

#include <type_traits>
#include <utility>

#include "infini_train/include/datatype.h"

namespace infini_train::common::cpu {

namespace detail {

// FP16/BF16 don't support implicit conversion, so we route through float.
template <typename DST, typename SRC> DST CastImpl(SRC &&x) {
    using SrcBase = std::remove_cvref_t<SRC>;
    if constexpr (std::is_same_v<DST, SrcBase>) {
        return x;
    } else if constexpr (std::is_same_v<DST, FP16> || std::is_same_v<DST, BF16>) {
        // Destination is a framework 16-bit type: convert via float
        return DST(static_cast<float>(std::forward<SRC>(x)));
    } else if constexpr (std::is_same_v<SrcBase, FP16> || std::is_same_v<SrcBase, BF16>) {
        // Source is a framework 16-bit type: widen to float first
        return static_cast<DST>(static_cast<float>(x));
    } else {
        return static_cast<DST>(std::forward<SRC>(x));
    }
}

} // namespace detail

/**
 * Converts a value between arbitrary types, including framework FP16/BF16.
 *
 * @tparam DST Destination type
 * @tparam SRC Source type (deduced)
 * @param x Input value
 * @return Value converted to DST type
 */
template <typename DST, typename SRC> DST Cast(SRC &&x) {
    static_assert(!std::is_reference_v<DST>, "Cast cannot return reference types");
    return detail::CastImpl<DST>(std::forward<SRC>(x));
}

} // namespace infini_train::common::cpu

#pragma once

#include <type_traits>
#include <utility>

#include "infini_train/include/datatype.h"

namespace infini_train::common::cpu {
/**
 * Converts a value between arbitrary types. This offers perfect
 * forwarding which preserves value categories (lvalues/rvalues)
 *
 * @tparam DST Destination type (deduced)
 * @tparam SRC Source type (deduced)
 * @param x Input value (preserves const/volatile and value category)
 * @return Value converted to DST type
 */
template <typename DST, typename SRC> DST Cast(SRC &&x) {
    static_assert(!std::is_reference_v<DST>, "Cast cannot return reference types");

    using Dst = std::remove_cv_t<std::remove_reference_t<DST>>;
    if constexpr (is_bfloat16<Dst>::value || is_fp16<Dst>::value) {
        // TODO(lzm): add cpu-version fp16 and bf16
        return Dst(static_cast<float>(std::forward<SRC>(x)));
    } else {
        return static_cast<DST>(std::forward<SRC>(x));
    }
}
} // namespace infini_train::common::cpu

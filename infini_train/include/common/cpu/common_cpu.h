#pragma once

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

    // TODO(lzm): add cpu-version fp16 and bf16
    return (DST)(std::forward<SRC>(x));
}
} // namespace infini_train::common::cpu

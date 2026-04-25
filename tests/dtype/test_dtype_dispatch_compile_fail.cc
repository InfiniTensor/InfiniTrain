#include "infini_train/include/datatype.h"
#include "infini_train/include/dtype_dispatch.h"

using namespace infini_train;

// ============================================================================
// Compile-fail: dispatching an unregistered low-precision dtype must be
//               intercepted at compile time
// ============================================================================

// Models a backend that has registered standard floating types but has NOT
// yet provided a mapping for the low-precision dtypes FP16 / BF16.
template <DataType DType> struct LowPrecisionMissingTypeMap;

template <> struct LowPrecisionMissingTypeMap<DataType::kFLOAT32> {
    using type = float;
};

int main() {
    // Dispatching kFLOAT16 through LowPrecisionMissingTypeMap must trigger the
    // static_assert inside DispatchByTypeMap, failing this translation unit
    // before MappedType_t<TypeMap, kFLOAT16> is ever instantiated.
    DispatchByTypeMap<LowPrecisionMissingTypeMap, DataType::kFLOAT16>(
        DataType::kFLOAT16,
        []<typename T>() { (void)sizeof(T); },
        "compile-fail: unregistered low-precision dtype");
    return 0;
}

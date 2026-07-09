// distribution.cc — Non-template distribution utilities.
//
// The core template machinery lives in distribution.h (all inline/template).
// This translation unit exists for:
//   1. Future non-template distribution helpers (e.g. bounds checking)
//   2. Ensuring the header compiles cleanly as a standalone TU
//   3. Providing a home for distribution-related constants if needed

#include "infini_train/include/core/distribution.h"

namespace infini_train::core::distribution {

// Currently all distribution logic is header-only (template + inline).
// This TU is reserved for future non-template distribution utilities,
// such as:
//   - check_uniform_bounds (parameter validation, cf. PyTorch's
//     CHECK_OUT_OF_BOUNDS / check_from_to_in_range)
//   - Distribution registration for multi-dtype dispatch

}  // namespace infini_train::core::distribution

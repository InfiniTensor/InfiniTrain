#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/dtype_dispatch.h"

#include "infini_train/src/core/runtime/cpu/cpu_dispatch.h"

using namespace infini_train;

// ============================================================================
// Test 1: HasMappedType_v intercepts backends missing FP16 / BF16
// ============================================================================

// A backend TypeMap that only registers kFLOAT32 — FP16 / BF16 are absent.
template <DataType DType> struct LowPrecisionAbsentTypeMap;

template <> struct LowPrecisionAbsentTypeMap<DataType::kFLOAT32> {
    using type = float;
};

static_assert(HasMappedType_v<LowPrecisionAbsentTypeMap, DataType::kFLOAT32>,
              "sanity: registered dtype must be detected as present");
static_assert(!HasMappedType_v<LowPrecisionAbsentTypeMap, DataType::kFLOAT16>,
              "unregistered kFLOAT16 must be intercepted by HasMappedType_v");
static_assert(!HasMappedType_v<LowPrecisionAbsentTypeMap, DataType::kBFLOAT16>,
              "unregistered kBFLOAT16 must be intercepted by HasMappedType_v");

// ============================================================================
// Test 2: CpuTypeMap resolves FP16 / BF16 to framework scalar types
// ============================================================================

static_assert(std::is_same_v<MappedType_t<core::cpu::CpuTypeMap, DataType::kFLOAT16>, FP16>,
              "CpuTypeMap<kFLOAT16> must resolve to framework FP16");
static_assert(std::is_same_v<MappedType_t<core::cpu::CpuTypeMap, DataType::kBFLOAT16>, BF16>,
              "CpuTypeMap<kBFLOAT16> must resolve to framework BF16");

// ============================================================================
// Test 3: Runtime dispatch of kFLOAT16 / kBFLOAT16
// ============================================================================

void TestRuntimeDispatchLowPrecision() {
    std::cout << "\n=== Test 3: Runtime dispatch of kFLOAT16 / kBFLOAT16 ===" << std::endl;

    // kFLOAT16 must dispatch to framework FP16
    bool called_fp16 = false;
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16>(
        DataType::kFLOAT16,
        [&called_fp16]<typename T>() {
            if constexpr (std::is_same_v<T, FP16>) {
                called_fp16 = true;
            }
        },
        "dispatch kFLOAT16");
    CHECK(called_fp16) << "DispatchCpuFunc did not invoke functor for kFLOAT16";

    // kBFLOAT16 must dispatch to framework BF16
    bool called_bf16 = false;
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16>(
        DataType::kBFLOAT16,
        [&called_bf16]<typename T>() {
            if constexpr (std::is_same_v<T, BF16>) {
                called_bf16 = true;
            }
        },
        "dispatch kBFLOAT16");
    CHECK(called_bf16) << "DispatchCpuFunc did not invoke functor for kBFLOAT16";

    std::cout << "Low-precision dispatch OK." << std::endl;
}

// ============================================================================
// Test 4: Runtime dispatch of a low-precision dtype outside AllowedDTypes
//         must fatal
// ============================================================================

// Sub-process entry: tries to dispatch kFLOAT16 with only kFLOAT32 allowed.
void TriggerRuntimeUnsupportedLowPrecisionFatal() {
    core::cpu::DispatchCpuFunc<DataType::kFLOAT32>(
        DataType::kFLOAT16,
        []<typename T>() { (void)sizeof(T); },
        "intercept kFLOAT16 when only kFLOAT32 is allowed");
}

void TestRuntimeInterceptLowPrecision(const char *argv0) {
    std::cout << "\n=== Test 4: Runtime intercept of kFLOAT16 outside AllowedDTypes ===" << std::endl;
    const std::string cmd = std::string(argv0) + " --expect-runtime-fatal > /dev/null 2>&1";
    const int status = std::system(cmd.c_str());
    CHECK_NE(status, 0) << "Expected non-zero exit when dispatching an unallowed low-precision dtype";
    std::cout << "Low-precision runtime intercept OK." << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    if (argc > 1 && std::string(argv[1]) == "--expect-runtime-fatal") {
        TriggerRuntimeUnsupportedLowPrecisionFatal();
        return 0;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  Low-precision Dtype Dispatch Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "Compile-time checks: PASSED" << std::endl;

    TestRuntimeDispatchLowPrecision();
    TestRuntimeInterceptLowPrecision(argv[0]);

    std::cout << "\nAll low-precision dtype dispatch tests passed." << std::endl;
    return 0;
}

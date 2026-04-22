#include <gtest/gtest.h>

#include <type_traits>

#include "infini_train/include/datatype.h"
#include "infini_train/include/dtype_dispatch.h"
#include "infini_train/src/core/runtime/cpu/cpu_dispatch.h"
#include "test_utils.h"

using namespace infini_train;

// ============================================================================
// Compile-time checks (static_assert, always verified at build time)
// ============================================================================

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

static_assert(std::is_same_v<MappedType_t<core::cpu::CpuTypeMap, DataType::kFLOAT16>, FP16>,
              "CpuTypeMap<kFLOAT16> must resolve to framework FP16");
static_assert(std::is_same_v<MappedType_t<core::cpu::CpuTypeMap, DataType::kBFLOAT16>, BF16>,
              "CpuTypeMap<kBFLOAT16> must resolve to framework BF16");

// ============================================================================
// Runtime tests
// ============================================================================

class DtypeDispatchTest : public infini_train::test::InfiniTrainTest {};

TEST_P(DtypeDispatchTest, DispatchFP16) {
    bool called = false;
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16>(
        DataType::kFLOAT16,
        [&called]<typename T>() {
            if constexpr (std::is_same_v<T, FP16>) {
                called = true;
            }
        },
        "dispatch kFLOAT16");
    EXPECT_TRUE(called);
}

TEST_P(DtypeDispatchTest, DispatchBF16) {
    bool called = false;
    core::cpu::DispatchCpuFunc<DataType::kFLOAT16, DataType::kBFLOAT16>(
        DataType::kBFLOAT16,
        [&called]<typename T>() {
            if constexpr (std::is_same_v<T, BF16>) {
                called = true;
            }
        },
        "dispatch kBFLOAT16");
    EXPECT_TRUE(called);
}

TEST_P(DtypeDispatchTest, UnallowedDtypeFatals) {
    EXPECT_DEATH(core::cpu::DispatchCpuFunc<DataType::kFLOAT32>(
                     DataType::kFLOAT16, []<typename T>() { (void)sizeof(T); },
                     "intercept kFLOAT16 when only kFLOAT32 is allowed"),
                 "");
}

INFINI_TRAIN_REGISTER_TEST(DtypeDispatchTest);

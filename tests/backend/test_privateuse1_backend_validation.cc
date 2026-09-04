#include "gtest/gtest.h"

#include "infini_train/include/core/privateuse1_backend.h"

namespace infini_train::test {
namespace {

void UnusedRegistrationCallback() {}

core::PrivateUse1BackendRegistration MinimalRegistration() {
    core::PrivateUse1BackendRegistration registration;
    registration.name = "valid_name";
    registration.default_autocast_dtype = DataType::kBFLOAT16;
    registration.register_runtime = &UnusedRegistrationCallback;
    registration.register_kernels = &UnusedRegistrationCallback;
    return registration;
}

TEST(PrivateUse1BackendValidationTest, RejectsReservedBackendNames) {
    auto registration = MinimalRegistration();
    registration.name = "cuda";
    EXPECT_DEATH(core::RegisterPrivateUse1Backend(registration), "non-reserved");
}

TEST(PrivateUse1BackendValidationTest, RejectsNonAsciiBackendNames) {
    auto registration = MinimalRegistration();
    registration.name = "invalid-name";
    EXPECT_DEATH(core::RegisterPrivateUse1Backend(registration), "lowercase ASCII");
}

TEST(PrivateUse1BackendValidationTest, RequiresDefaultAutocastDtype) {
    auto registration = MinimalRegistration();
    registration.default_autocast_dtype.reset();
    EXPECT_DEATH(core::RegisterPrivateUse1Backend(registration), "must declare a default autocast dtype");
}

TEST(PrivateUse1BackendValidationTest, RejectsNonLowPrecisionDefaultAutocastDtype) {
    auto registration = MinimalRegistration();
    registration.default_autocast_dtype = DataType::kFLOAT32;
    EXPECT_DEATH(core::RegisterPrivateUse1Backend(registration), "must be float16 or bfloat16");
}

TEST(PrivateUse1BackendValidationTest, RejectsDefaultAutocastDtypeQueryBeforeRegistration) {
    EXPECT_DEATH(core::GetPrivateUse1BackendDefaultAutocastDtype(), "is not registered");
}

} // namespace
} // namespace infini_train::test

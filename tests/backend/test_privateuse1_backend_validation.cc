#include "gtest/gtest.h"

#include "infini_train/include/core/privateuse1_backend.h"

namespace infini_train::test {
namespace {

void UnusedRegistrationCallback() {}

core::PrivateUse1BackendRegistration MinimalRegistration() {
    core::PrivateUse1BackendRegistration registration;
    registration.name = "valid_name";
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

} // namespace
} // namespace infini_train::test

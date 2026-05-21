#include <memory>

#include "gtest/gtest.h"

#include "infini_train/include/core/generator/cpu_generator_impl.h"

using infini_train::core::CPUGeneratorImpl;

TEST(CPUGeneratorImplTest, ManualSeedSetsCurrentAndInitialSeed) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(42);
    EXPECT_EQ(impl.CurrentSeed(), 42u);
    EXPECT_EQ(impl.InitialSeed(), 42u);
}

TEST(CPUGeneratorImplTest, EngineAdvancesAcrossCalls) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(123);
    auto &eng = impl.engine();
    const uint64_t a = eng();
    const uint64_t b = eng();
    EXPECT_NE(a, b);
}

TEST(CPUGeneratorImplTest, SameManualSeedReproduces) {
    CPUGeneratorImpl a(0), b(0);
    a.SetCurrentSeed(7);
    b.SetCurrentSeed(7);
    EXPECT_EQ(a.engine()(), b.engine()());
    EXPECT_EQ(a.engine()(), b.engine()());
}

TEST(CPUGeneratorImplTest, SeedRandomizesCurrentButPreservesInitial) {
    CPUGeneratorImpl impl(0);
    impl.SetCurrentSeed(99);
    const uint64_t initial_before = impl.InitialSeed();
    const uint64_t new_seed = impl.Seed();
    EXPECT_EQ(impl.CurrentSeed(), new_seed);
    EXPECT_EQ(impl.InitialSeed(), initial_before);
}

TEST(CPUGeneratorImplTest, IndexNonZeroAborts) {
    EXPECT_DEATH(CPUGeneratorImpl(1), "");
}

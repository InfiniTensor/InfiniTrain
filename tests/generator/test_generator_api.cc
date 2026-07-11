// Generator handle class API tests.
//
// These tests verify the basic semantics of the Generator wrapper/handle class
// (e.g. Defined(), comparison operators, copy/move semantics, and polymorphic
// dispatch using a simple stub implementation) without depending on the concrete
// CPU or CUDA generator backends.

#include <memory>
#include <vector>

#include "infini_train/include/core/generator.h"
#include "infini_train/include/device.h"

#include "gtest/gtest.h"

using namespace infini_train;

namespace {

// A simple stub generator implementation for verifying handle dispatch.
class StubGeneratorImpl : public core::GeneratorImpl {
public:
    static constexpr Device::DeviceType kDeviceType = Device::DeviceType::kCPU;

    explicit StubGeneratorImpl(uint64_t seed = 42) : seed_(seed), offset_(0) {}

    void SetCurrentSeed(uint64_t seed) override { seed_ = seed; }

    uint64_t CurrentSeed() const override { return seed_; }

    uint64_t Seed() override { return ++seed_; }

    void SetOffset(uint64_t offset) override { offset_ = offset; }

    uint64_t GetOffset() const override { return offset_; }

    std::vector<uint8_t> GetState() const override { return std::vector<uint8_t>{0xBE, 0xEF}; }

    void SetState(const std::vector<uint8_t> &state) override { (void)state; }

    Device GetDevice() const override { return Device(Device::DeviceType::kCPU, 0); }

    std::shared_ptr<core::GeneratorImpl> Clone() const override {
        auto cloned = std::make_shared<StubGeneratorImpl>(seed_);
        cloned->offset_ = offset_;
        return cloned;
    }

private:
    uint64_t seed_;
    uint64_t offset_;
};

} // namespace

TEST(GeneratorAPITest, DefaultConstructedIsUndefined) {
    core::Generator gen;
    EXPECT_FALSE(gen.Defined());
}

TEST(GeneratorAPITest, ConstructorRejectsNullptr) {
    EXPECT_DEATH(core::Generator(nullptr), "nullptr is not supported");
}

TEST(GeneratorAPITest, DefinedWhenImplProvided) {
    auto impl = std::make_shared<StubGeneratorImpl>(123);
    core::Generator gen(impl);
    EXPECT_TRUE(gen.Defined());
    EXPECT_EQ(gen.CurrentSeed(), 123u);
}

TEST(GeneratorAPITest, CopySharesImplAndState) {
    auto impl = std::make_shared<StubGeneratorImpl>(10);
    core::Generator gen1(impl);
    core::Generator gen2 = gen1;

    EXPECT_TRUE(gen1.Defined());
    EXPECT_TRUE(gen2.Defined());
    EXPECT_EQ(gen1, gen2);
    EXPECT_EQ(gen1.UnsafeGetImpl(), gen2.UnsafeGetImpl());

    // Modifying via one handle affects the other since they share the impl.
    gen1.SetCurrentSeed(20);
    EXPECT_EQ(gen2.CurrentSeed(), 20u);
}

TEST(GeneratorAPITest, MoveTransfersImpl) {
    auto impl = std::make_shared<StubGeneratorImpl>(10);
    core::Generator gen1(impl);
    core::Generator gen2 = std::move(gen1);

    EXPECT_FALSE(gen1.Defined());
    EXPECT_TRUE(gen2.Defined());
    EXPECT_EQ(gen2.CurrentSeed(), 10u);
}

TEST(GeneratorAPITest, CloneCreatesIndependentImpl) {
    auto impl = std::make_shared<StubGeneratorImpl>(10);
    core::Generator gen1(impl);
    core::Generator gen2 = gen1.Clone();

    EXPECT_TRUE(gen1.Defined());
    EXPECT_TRUE(gen2.Defined());
    EXPECT_NE(gen1, gen2);
    EXPECT_NE(gen1.UnsafeGetImpl(), gen2.UnsafeGetImpl());

    EXPECT_EQ(gen1.CurrentSeed(), gen2.CurrentSeed());

    // Mutating gen1 does not affect gen2 because they are clones.
    gen1.SetCurrentSeed(30);
    EXPECT_EQ(gen1.CurrentSeed(), 30u);
    EXPECT_EQ(gen2.CurrentSeed(), 10u);
}

TEST(GeneratorAPITest, BasicAPIForwarding) {
    auto impl = std::make_shared<StubGeneratorImpl>(100);
    core::Generator gen(impl);

    EXPECT_EQ(gen.CurrentSeed(), 100u);

    gen.ManualSeed(200);
    EXPECT_EQ(gen.CurrentSeed(), 200u);

    uint64_t old_seed = gen.CurrentSeed();
    uint64_t new_seed = gen.Seed();
    EXPECT_NE(new_seed, old_seed);
    EXPECT_EQ(new_seed, gen.CurrentSeed());

    EXPECT_EQ(gen.GetOffset(), 0u);
    gen.SetOffset(99);
    EXPECT_EQ(gen.GetOffset(), 99u);

    std::vector<uint8_t> expected_state{0xBE, 0xEF};
    EXPECT_EQ(gen.GetState(), expected_state);

    EXPECT_EQ(gen.GetDevice(), Device(Device::DeviceType::kCPU, 0));
}

TEST(GeneratorAPITest, CheckGeneratorDeath) {
    std::optional<core::Generator> null_gen = std::nullopt;
    EXPECT_DEATH(core::CheckGenerator<StubGeneratorImpl>(null_gen), "Expected a Generator but received std::nullopt");

    core::Generator undef_gen;
    EXPECT_DEATH(core::CheckGenerator<StubGeneratorImpl>(undef_gen), "undefined implementation");
}

TEST(GeneratorAPITest, MakeGeneratorTest) {
    auto gen = core::MakeGenerator<StubGeneratorImpl>(999);
    EXPECT_TRUE(gen.Defined());
    EXPECT_EQ(gen.CurrentSeed(), 999u);
}

TEST(GeneratorAPITest, UnsafeGetImplBehavior) {
    auto gen = core::MakeGenerator<StubGeneratorImpl>(111);
    auto *impl = gen.UnsafeGetImpl();
    EXPECT_NE(impl, nullptr);
    EXPECT_EQ(impl->CurrentSeed(), 111u);

    // Modifying via UnsafeGetImpl directly changes the generator state
    impl->SetCurrentSeed(222);
    EXPECT_EQ(gen.CurrentSeed(), 222u);
}

TEST(GeneratorAPITest, CurrentSeedRelationship) {
    auto gen = core::MakeGenerator<StubGeneratorImpl>(100);
    EXPECT_EQ(gen.CurrentSeed(), 100u);

    uint64_t old_seed = gen.CurrentSeed();
    uint64_t new_seed = gen.Seed();
    EXPECT_NE(new_seed, old_seed);
    EXPECT_EQ(new_seed, gen.CurrentSeed());
}

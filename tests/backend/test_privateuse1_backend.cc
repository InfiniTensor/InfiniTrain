#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/core/privateuse1_backend.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::test {
namespace {

int g_fake_runtime_initialize_count = 0;

class FakeStream final : public core::Stream {};
class FakeEvent final : public core::Event {};

class FakePrivateUse1GuardImpl final : public core::DeviceGuardImpl {
public:
    void Initialize() override {
        std::call_once(initialize_once_, [] { ++g_fake_runtime_initialize_count; });
    }

    Device GetDevice() const override { return Device(Type(), current_device_); }

    void SetDevice(Device device) const override {
        CHECK(device.type() == Type());
        CHECK_EQ(device.index(), 0);
        current_device_ = device.index();
    }

    int DeviceCount() const override { return 1; }

    Device::DeviceType Type() const override { return Device::DeviceType::kPrivateUse1; }

    core::Stream *GetStream(Device device) const override {
        CHECK(device.type() == Type());
        return const_cast<FakeStream *>(&default_stream_);
    }

    core::Stream *CreateStream(Device device) const override {
        CHECK(device.type() == Type());
        return new FakeStream();
    }

    core::Stream *CreateStreamWithPriority(Device device, int) const override { return CreateStream(device); }

    void DestroyStream(core::Stream *stream) const override { delete stream; }

    void GetStreamPriorityRange(int *low, int *high) const override {
        *low = 0;
        *high = 0;
    }

    void EventCreate(core::Event **event) const override { *event = new FakeEvent(); }

    void EventCreateWithFlags(core::Event **event, core::EventFlag) const override { EventCreate(event); }

    void EventDestroy(core::Event *event) const override { delete event; }

    void EventRecord(core::Event *, core::Stream *) const override {}

    void StreamWaitEvent(core::Stream *, core::Event *, uint32_t) const override {}

    core::RuntimeStatus EventSynchronize(core::Event *) const override { return core::RuntimeStatus::kSuccess; }

    core::RuntimeStatus EventQuery(core::Event *) const override { return core::RuntimeStatus::kSuccess; }

    float EventElapsedTime(core::Event *, core::Event *) const override { return 0.0F; }

    void SynchronizeDevice(Device) const override {}

    void SynchronizeStream(core::Stream *) const override {}

    void Malloc(void **dev_ptr, size_t size) override { *dev_ptr = ::operator new(size == 0 ? 1 : size); }

    void MallocAsync(void **dev_ptr, size_t size, core::Stream *) override { Malloc(dev_ptr, size); }

    void Free(void *dev_ptr) override { ::operator delete(dev_ptr); }

    void FreeAsync(void *dev_ptr, core::Stream *) override { Free(dev_ptr); }

    void Memcpy(void *dst, const void *src, size_t count, core::MemcpyKind) override { std::memcpy(dst, src, count); }

    void MemcpyAsync(void *dst, const void *src, size_t count, core::MemcpyKind kind, core::Stream *) override {
        Memcpy(dst, src, count, kind);
    }

    void ResetMemPoolHighWatermarks(Device) const override {}

    std::pair<size_t, size_t> GetMemPoolPeakMB(Device) const override { return {0, 0}; }

private:
    inline static thread_local int8_t current_device_ = 0;
    std::once_flag initialize_once_;
    FakeStream default_stream_;
};

void FakeFill(std::shared_ptr<Tensor> tensor, Scalar value) {
    CHECK(tensor->Dtype() == DataType::kFLOAT32);
    auto *data = static_cast<float *>(tensor->DataPtr());
    for (size_t i = 0; i < tensor->NumElements(); ++i) { data[i] = value.to<float>(); }
}

std::shared_ptr<Tensor> FakeCast(std::shared_ptr<Tensor> input, DataType dtype) {
    CHECK(input->Dtype() == dtype);
    auto output = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());
    std::memcpy(output->DataPtr(), input->DataPtr(), input->SizeInBytes());
    return output;
}

std::shared_ptr<Tensor> FakeNoOpForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &dims) {
    return std::make_shared<Tensor>(*input, 0, dims);
}

std::shared_ptr<Tensor> FakeNoOpBackward(const std::vector<int64_t> &dims, const std::shared_ptr<Tensor> &grad_output) {
    return std::make_shared<Tensor>(*grad_output, 0, dims);
}

void RegisterFakeRuntime() {
    CHECK_EQ(core::GetPrivateUse1BackendName(), "fake");
    CHECK_EQ(Device(Device::DeviceType::kPrivateUse1, 0).ToString(), "Device(fake, 0)");
    INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kPrivateUse1, FakePrivateUse1GuardImpl)
}

void RegisterFakeKernels() {
    REGISTER_KERNEL(Device::DeviceType::kPrivateUse1, Cast, FakeCast)
    REGISTER_KERNEL(Device::DeviceType::kPrivateUse1, Fill, FakeFill)
    REGISTER_KERNEL(Device::DeviceType::kPrivateUse1, NoOpForward, FakeNoOpForward)
    REGISTER_KERNEL(Device::DeviceType::kPrivateUse1, NoOpBackward, FakeNoOpBackward)
}

void InitializeFakeBackend() {
    static std::once_flag once;
    std::call_once(once, [] {
        core::PrivateUse1BackendRegistration registration;
        registration.name = "fake";
        registration.register_runtime = &RegisterFakeRuntime;
        registration.register_kernels = &RegisterFakeKernels;
        core::RegisterPrivateUse1Backend(registration);
    });
}

TEST(PrivateUse1BackendTest, RegistersRuntimeMetadataAndBasicKernels) {
    InitializeFakeBackend();

    EXPECT_EQ(g_fake_runtime_initialize_count, 0);
    EXPECT_TRUE(core::HasPrivateUse1Backend());
    EXPECT_EQ(core::GetPrivateUse1BackendName(), "fake");
    EXPECT_EQ(Device::ParseType("fake"), Device::DeviceType::kPrivateUse1);
    EXPECT_EQ(Device::ParseType("privateuse1"), Device::DeviceType::kPrivateUse1);
    EXPECT_FALSE(Device::ParseType("missing").has_value());

    Device device(Device::DeviceType::kPrivateUse1, 0);
    EXPECT_EQ(device.ToString(), "Device(fake, 0)");
    EXPECT_TRUE(core::DeviceGuardImplRegistry::Instance().Has(device.type()));
    EXPECT_EQ(core::GetDeviceGuardImpl(device.type())->Type(), device.type());
    EXPECT_EQ(g_fake_runtime_initialize_count, 1);

    {
        AutocastGuard guard(device.type(), DataType::kBFLOAT16);
        EXPECT_EQ(GetCurrentAutocastContext().autocast_dtype, DataType::kBFLOAT16);
    }
    EXPECT_DEATH({ AutocastGuard guard(device.type()); }, "requires an explicit dtype");

    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, device);
    tensor->Fill(Scalar(3.0F));
    const auto *data = static_cast<const float *>(tensor->DataPtr());
    for (size_t i = 0; i < tensor->NumElements(); ++i) { EXPECT_FLOAT_EQ(data[i], 3.0F); }

    auto view = tensor->View({3, 2});
    EXPECT_EQ(view->Dims(), (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(view->DataPtr(), tensor->DataPtr());
}

} // namespace
} // namespace infini_train::test

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"

using namespace infini_train;

namespace {

bool EnvFlagEnabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return !(value[0] == '\0' || value[0] == '0');
}

bool CacheReuseExpected() {
    if (EnvFlagEnabled("INFINI_MACA_DISABLE_CACHE")) {
        return false;
    }
    const char *max_mb = std::getenv("INFINI_MACA_CACHE_MAX_MB");
    return !(max_mb != nullptr && max_mb[0] == '0' && max_mb[1] == '\0');
}

Device MakeMacaDevice(int device_index) { return Device(Device::DeviceType::kMACA, static_cast<int8_t>(device_index)); }

void TestSingleThreadReuse(core::DeviceGuardImpl *impl, Device device) {
    core::DeviceGuard guard(device);
    auto *stream = impl->GetStream(device);

    void *first = nullptr;
    impl->MallocAsync(&first, 1 << 20, stream);
    CHECK_NOTNULL(first);
    impl->FreeAsync(first, stream);
    impl->SynchronizeDevice(device);

    void *second = nullptr;
    impl->MallocAsync(&second, 1 << 20, stream);
    CHECK_NOTNULL(second);
    if (CacheReuseExpected()) {
        CHECK_EQ(first, second) << "Expected same-size allocation to reuse cached MACA block.";
    }
    impl->FreeAsync(second, stream);
    impl->SynchronizeDevice(device);
}

void TestCrossStreamFree(core::DeviceGuardImpl *impl, Device device) {
    core::DeviceGuard guard(device);
    auto *stream_a = impl->CreateStream(device);
    auto *stream_b = impl->CreateStream(device);

    std::vector<uint8_t> host(4096, 7);
    void *ptr = nullptr;
    impl->MallocAsync(&ptr, 2 << 20, stream_a);
    CHECK_NOTNULL(ptr);
    impl->MemcpyAsync(ptr, host.data(), host.size(), core::MemcpyKind::kH2D, stream_a);
    impl->FreeAsync(ptr, stream_a);

    void *next = nullptr;
    impl->MallocAsync(&next, 2 << 20, stream_b);
    CHECK_NOTNULL(next);
    impl->FreeAsync(next, stream_b);
    impl->SynchronizeDevice(device);

    impl->DestroyStream(stream_a);
    impl->DestroyStream(stream_b);
}

void TestMultiThreadedAllocFree(core::DeviceGuardImpl *impl, int device_count) {
    constexpr int kThreads = 8;
    constexpr int kIters = 64;
    std::atomic<int> ready{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for (int tid = 0; tid < kThreads; ++tid) {
        workers.emplace_back([impl, device_count, tid, &ready] {
            const int device_index = tid % device_count;
            Device device = MakeMacaDevice(device_index);
            core::DeviceGuard guard(device);
            auto *stream = impl->CreateStream(device);
            std::vector<uint8_t> host(8192, static_cast<uint8_t>(tid));

            ready.fetch_add(1, std::memory_order_release);
            while (ready.load(std::memory_order_acquire) < kThreads) {}

            for (int iter = 0; iter < kIters; ++iter) {
                const size_t bytes = 512 + static_cast<size_t>((tid + 1) * (iter + 3) % 1024) * 4096;
                void *ptr = nullptr;
                impl->MallocAsync(&ptr, bytes, stream);
                CHECK_NOTNULL(ptr);
                impl->MemcpyAsync(ptr, host.data(), std::min(bytes, host.size()), core::MemcpyKind::kH2D, stream);
                impl->SynchronizeStream(stream);
                impl->FreeAsync(ptr, stream);
            }

            impl->SynchronizeStream(stream);
            impl->DestroyStream(stream);
        });
    }

    for (auto &worker : workers) { worker.join(); }
}

} // namespace

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    auto *impl = core::GetDeviceGuardImpl(Device::DeviceType::kMACA);
    CHECK_NOTNULL(impl);
    const int device_count = impl->DeviceCount();
    CHECK_GT(device_count, 0);

    Device device0 = MakeMacaDevice(0);
    TestSingleThreadReuse(impl, device0);
    TestCrossStreamFree(impl, device0);
    TestMultiThreadedAllocFree(impl, device_count);

    std::cout << "MACA allocator stress test passed" << std::endl;
    return 0;
}

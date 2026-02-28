#include "infini_train/include/nn/parallel/work.h"

#include <stdexcept>
#include <string>
#include <thread>

#include "glog/logging.h"

#include "infini_train/include/core/ccl/ccl.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"

namespace infini_train::nn::parallel {
namespace {
std::exception_ptr makeRuntimeError(const char *api, core::RuntimeStatus status) {
    return std::make_exception_ptr(
        std::runtime_error(std::string(api) + " failed with status " + core::RuntimeStatusToString(status)));
}
} // namespace

Work::Work(Device device, core::CclComm *comm) : device_(device), comm_(comm) {
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    impl->EventCreateWithFlags(&ready_event_, core::EventFlag::kDisableTiming);
    impl->EventCreateWithFlags(&done_event_, core::EventFlag::kDisableTiming);
}

Work::~Work() {
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    if (ready_event_) {
        impl->EventDestroy(ready_event_);
    }
    if (done_event_) {
        impl->EventDestroy(done_event_);
    }
}

bool Work::WaitBlocking(std::chrono::milliseconds timeout) {
    // Block wait on host
    core::DeviceGuard guard(device_);
    auto *impl = core::GetDeviceGuardImpl(device_.type());

    // If timeout is not set, then wait till it finishes
    if (timeout <= std::chrono::milliseconds::zero()) {
        if (const auto status = impl->EventSynchronize(done_event_); status != core::RuntimeStatus::kSuccess) {
            SetException(makeRuntimeError("EventSynchronize", status));
            return false;
        }
        return CheckCclStatus();
    }

    // If timeout is set, keep querying till time's up
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        const auto query = impl->EventQuery(done_event_);
        if (query == core::RuntimeStatus::kSuccess) {
            return CheckCclStatus();
        }
        if (query != core::RuntimeStatus::kNotReady) {
            SetException(makeRuntimeError("EventQuery", query));
            return false;
        }

        // NOTE(zbl): sleep for a while in case of busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    if (exception_) {
        // NOTE(zbl): do not throw any c++ exception
        LOG(FATAL) << "Work::WaitBlocking: Error occurs while wait(). ";
    }

    return false;
}

bool Work::WaitNonBlocking() {
    // Non-blocking wait on compute stream
    core::DeviceGuard guard(device_);
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    auto *stream = impl->GetStream(device_);
    impl->StreamWaitEvent(stream, done_event_, 0);
    return true;
}

void Work::Synchronize() const {
    core::DeviceGuard guard(device_);
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    const auto status = impl->EventSynchronize(done_event_);
    CHECK(status == core::RuntimeStatus::kSuccess)
        << "Work::Synchronize failed, status=" << core::RuntimeStatusToString(status);
}

bool Work::IsCompleted() const {
    if (completed_.load(std::memory_order_acquire)) {
        return true;
    }

    core::DeviceGuard guard(device_);
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    const auto query = impl->EventQuery(done_event_);
    if (query == core::RuntimeStatus::kSuccess) {
        const_cast<Work *>(this)->CheckCclStatus();
        return true;
    }
    if (query != core::RuntimeStatus::kNotReady) {
        const_cast<Work *>(this)->SetException(makeRuntimeError("EventQuery", query));
        return true;
    }
    return false;
}

bool Work::IsSuccess() const {
    if (!IsCompleted()) {
        return false;
    }
    return success_.load(std::memory_order_acquire) && !exception_;
}

bool Work::CheckCclStatus() {
    if (comm_ != nullptr) {
        auto *impl = core::GetCclImpl(device_.type());
        core::CclStatus async_error = core::CclStatus::kSuccess;
        impl->CommGetAsyncError(comm_, &async_error);
        if (async_error != core::CclStatus::kSuccess) {
            SetException(std::make_exception_ptr(
                std::runtime_error(std::string("CCL async error: ") + core::CclStatusToString(async_error))));
            return false;
        }
    }
    success_.store(true, std::memory_order_release);
    completed_.store(true, std::memory_order_release);
    return true;
}

void Work::SetException(std::exception_ptr e) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!exception_) {
        exception_ = std::move(e);
    }
    completed_.store(true, std::memory_order_release);
    success_.store(false, std::memory_order_release);
}

} // namespace infini_train::nn::parallel

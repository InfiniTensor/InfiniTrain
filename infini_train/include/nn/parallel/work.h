#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>

#include "infini_train/include/core/ccl/ccl_common.h"
#include "infini_train/include/core/event.h"
#include "infini_train/include/device.h"

namespace infini_train::nn::parallel {

class Work {
public:
    explicit Work(Device device, core::CclComm *comm = nullptr);
    ~Work();

    bool WaitBlocking(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    bool WaitNonBlocking();

    bool IsCompleted() const;
    bool IsSuccess() const;

    void Synchronize() const;

    std::exception_ptr exception() const { return exception_; }

    core::Event *ready_event() const { return ready_event_; }
    core::Event *done_event() const { return done_event_; }

private:
    bool CheckCclStatus();
    void SetException(std::exception_ptr e);

private:
    Device device_;
    core::Event *ready_event_ = nullptr;
    core::Event *done_event_ = nullptr;
    core::CclComm *comm_ = nullptr;

    mutable std::mutex mutex_;
    std::exception_ptr exception_;
    std::atomic<bool> completed_{false};
    std::atomic<bool> success_{false};
};

} // namespace infini_train::nn::parallel

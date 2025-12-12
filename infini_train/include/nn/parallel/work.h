#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef USE_NCCL
#include <nccl.h>
#endif

namespace infini_train {
class Device;
} // namespace infini_train

namespace infini_train::nn::parallel {

class Work {
public:
    virtual ~Work() = default;

    virtual bool WaitBlocking(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) = 0;
    virtual bool WaitNonBlocking() = 0;

    virtual bool IsCompleted() const = 0;
    virtual bool IsSuccess() const = 0;

    virtual void Synchronize() const = 0;

    virtual std::exception_ptr exception() const = 0;

    virtual void *ready_event() const = 0;
    virtual void *done_event() const = 0;
};

#ifdef USE_NCCL
class WorkNccl final : public Work {
public:
    WorkNccl(const Device *device, ncclComm_t comm);
    ~WorkNccl() override;

    bool WaitBlocking(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) override;
    bool WaitNonBlocking() override;

    bool IsCompleted() const override;
    bool IsSuccess() const override;

    void Synchronize() const override;

    std::exception_ptr exception() const override { return exception_; };

    void *ready_event() const override { return reinterpret_cast<void *>(ready_event_); };
    void *done_event() const override { return reinterpret_cast<void *>(done_event_); };

private:
    bool CheckNcclStatus();
    void SetException(std::exception_ptr e);

private:
    const Device *device_ = nullptr;
    cudaEvent_t ready_event_;
    cudaEvent_t done_event_;
    ncclComm_t comm_;

    mutable std::mutex mutex_;
    std::exception_ptr exception_;
    std::atomic<bool> completed_{false};
    std::atomic<bool> success_{false};
};
#endif

} // namespace infini_train::nn::parallel

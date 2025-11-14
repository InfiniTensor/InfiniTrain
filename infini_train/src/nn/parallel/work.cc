#include "infini_train/include/nn/parallel/work.h"

#include "glog/logging.h"

#ifdef USE_CUDA
#include "infini_train/include/common/cuda/common_cuda.h"
#endif

namespace infini_train::nn::parallel {
#ifdef USE_NCCL
namespace {
std::exception_ptr makeCudaError(cudaError_t err) {
    return std::make_exception_ptr(std::runtime_error(cudaGetErrorString(err)));
}
} // namespace

WorkNccl::WorkNccl(const Device *device, ncclComm_t comm) : comm_(comm) {
    CUDA_CHECK(cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_event_, cudaEventDisableTiming));
}

WorkNccl::~WorkNccl() {
    if (ready_event_) {
        CUDA_CHECK(cudaEventDestroy(ready_event_));
    }
    if (done_event_) {
        CUDA_CHECK(cudaEventDestroy(done_event_));
    }
}

bool WorkNccl::Wait(std::chrono::milliseconds timeout) {
    // Block wait on host
    device_->SetDevice();

    // If timeout is not set, then wait till it finishes
    if (timeout <= std::chrono::milliseconds::zero()) {
        if (auto status = cudaEventSynchronize(done_event_); status != cudaSuccess) {
            SetException(makeCudaError(status));
            return false;
        }
        // Check NCCL status
        return CheckNcclStatus();
    }

    // If timeout is set, keep querying till time's up
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        cudaError_t query = cudaEventQuery(done_event_);
        if (query == cudaSuccess) {
            return CheckNcclStatus();
        }
        if (query != cudaErrorNotReady) {
            SetException(makeCudaError(query));
            return false;
        }
        // NOTE(zbl): sleep for a while in case of busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    if (exception_) {
        // NOTE(zbl): do not throw any c++ exception
        LOG(FATAL) << "Error occurs while wait(). ";
    }

    return false;
}

void WorkNccl::Synchronize() const { CUDA_CHECK(cudaEventSynchronize(done_event_)); }

bool WorkNccl::IsCompleted() const {
    if (completed_.load(std::memory_order_acquire)) {
        return true;
    }
    cudaError_t query = cudaEventQuery(done_event_);
    if (query == cudaSuccess) {
        const_cast<WorkNccl *>(this)->completed_.store(true, std::memory_order_release);
        const_cast<WorkNccl *>(this)->success_.store(true, std::memory_order_release);
        return true;
    }
    if (query != cudaErrorNotReady) {
        const_cast<WorkNccl *>(this)->SetException(makeCudaError(query));
        return true;
    }
    return false;
}

bool WorkNccl::IsSuccess() const {
    if (!IsCompleted()) {
        return false;
    }
    return success_.load(std::memory_order_acquire) && !exception_;
}

bool WorkNccl::CheckNcclStatus() {
    ncclResult_t async_error;
    if (comm_ && ncclCommGetAsyncError(comm_, &async_error) == ncclSuccess && async_error != ncclSuccess) {
        SetException(std::make_exception_ptr(
            std::runtime_error(std::string("NCCL async error: ") + ncclGetErrorString(async_error))));
        return false;
    }
    success_.store(true, std::memory_order_release);
    completed_.store(true, std::memory_order_release);
    return true;
}

void WorkNccl::SetException(std::exception_ptr e) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!exception_) {
        exception_ = std::move(e);
    }
    completed_.store(true, std::memory_order_release);
    success_.store(false, std::memory_order_release);
}
#endif

} // namespace infini_train::nn::parallel

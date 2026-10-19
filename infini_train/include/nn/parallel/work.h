#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>

#include "infini_train/include/device.h"

namespace infini_train::core {
class CclComm;
class Event;
}  // namespace infini_train::core

namespace infini_train::nn::parallel {

class Work {
 public:
  explicit Work(Device device, core::CclComm *comm = nullptr);
  ~Work();

  bool WaitBlocking(
      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
  bool WaitNonBlocking();

  bool IsCompleted() const;
  bool IsSuccess() const;

  void Synchronize() const;

  std::exception_ptr exception() const;

  core::Event *ready_event() const;
  core::Event *done_event() const;

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

}  // namespace infini_train::nn::parallel

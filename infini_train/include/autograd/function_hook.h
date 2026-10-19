#pragma once

#include <stdint.h>

#include <atomic>
#include <memory>

namespace infini_train::nn::parallel::comm {
enum class ReduceOpType : int8_t;
}  // namespace infini_train::nn::parallel::comm

namespace infini_train {
class Tensor;

namespace nn::parallel {
class ProcessGroup;
}  // namespace nn::parallel
}  // namespace infini_train

namespace infini_train::autograd {

class PreAccumulateGradHook {
 public:
  virtual void operator()(const std::shared_ptr<Tensor> &grad_output) = 0;

  // Return true if this hook has handled the current gradient accumulation.
  virtual bool TryBypassAccumulate(const std::shared_ptr<Tensor> &,
                                   const std::shared_ptr<Tensor> &, bool,
                                   float);

  virtual ~PreAccumulateGradHook() = default;
};

class PostAccumulateGradHook {
 public:
  virtual void operator()(const std::shared_ptr<Tensor> &tensor) = 0;

  virtual ~PostAccumulateGradHook() = default;
};

class AllReducePostAccumulateHook : public PostAccumulateGradHook {
 public:
  AllReducePostAccumulateHook(
      infini_train::nn::parallel::comm::ReduceOpType reduce_op,
      const infini_train::nn::parallel::ProcessGroup *pg = nullptr,
      std::shared_ptr<const std::atomic_bool> enabled = nullptr,
      float gradient_scaling_factor = 1.f);

  void operator()(const std::shared_ptr<Tensor> &tensor) override;

 private:
  infini_train::nn::parallel::comm::ReduceOpType reduce_op_;
  const infini_train::nn::parallel::ProcessGroup *pg_ = nullptr;
  std::shared_ptr<const std::atomic_bool> enabled_;
  float gradient_scaling_factor_ = 1.f;
};
}  // namespace infini_train::autograd

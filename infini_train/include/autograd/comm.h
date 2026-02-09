#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/device.h"

namespace infini_train {
class Tensor;
class Device;

namespace nn::parallel {
class ProcessGroup;
} // namespace nn::parallel
} // namespace infini_train

namespace infini_train::autograd {
class Scatter : public autograd::Function {
public:
    static constexpr char kType[] = "ScatterFunction";

    explicit Scatter(const std::vector<Device> &target_gpus, int64_t dim,
                     const infini_train::nn::parallel::ProcessGroup *pg = nullptr);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const infini_train::nn::parallel::ProcessGroup *pg_ = nullptr;
    std::vector<Device> target_gpus_;
    Device input_device_;
    int64_t dim_ = 0;
};

class Gather : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFunction";

    explicit Gather(Device target_device, int64_t dim, const infini_train::nn::parallel::ProcessGroup *pg = nullptr);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const infini_train::nn::parallel::ProcessGroup *pg_ = nullptr;
    Device target_device_;
    std::vector<Device> input_gpus_;
    int64_t dim_ = 0;
    bool unsqueezed_scalar_ = false;
};

class Broadcast : public autograd::Function {
public:
    static constexpr char kType[] = "BroadcastFunction";

    explicit Broadcast(const std::vector<Device> &target_gpus,
                       const infini_train::nn::parallel::ProcessGroup *pg = nullptr);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const infini_train::nn::parallel::ProcessGroup *pg_ = nullptr;
    std::vector<Device> target_gpus_;
    int64_t num_inputs_ = 0;
    Device input_device_;
};

class ReduceAddCoalesced : public autograd::Function {
public:
    static constexpr char kType[] = "ReduceAddCoalescedFunction";

    explicit ReduceAddCoalesced(Device destination, int64_t num_inputs,
                                const infini_train::nn::parallel::ProcessGroup *pg = nullptr);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const infini_train::nn::parallel::ProcessGroup *pg_ = nullptr;
    Device destination_;
    std::vector<Device> target_gpus_;
    int64_t num_inputs_ = 0;
};
} // namespace infini_train::autograd

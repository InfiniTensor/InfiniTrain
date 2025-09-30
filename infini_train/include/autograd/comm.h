#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
class Scatter : public autograd::Function {
public:
    static constexpr char kType[] = "ScatterFunction";

    explicit Scatter(const std::vector<const Device *> &target_gpus, int64_t dim)
        : autograd::Function(kType), target_gpus_(target_gpus), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::vector<const Device *> target_gpus_;
    const Device *input_device_ = nullptr;
    int64_t dim_ = 0;
};

class Gather : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFunction";

    explicit Gather(const Device *target_device, int64_t dim)
        : autograd::Function(kType), target_device_(target_device), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const Device *target_device_ = nullptr;
    std::vector<const Device *> input_gpus_;
    int64_t dim_ = 0;
    bool unsqueezed_scalar_ = false;
};

class Broadcast : public autograd::Function {
public:
    static constexpr char kType[] = "BroadcastFunction";

    explicit Broadcast(const std::vector<const Device *> &target_gpus)
        : autograd::Function(kType), target_gpus_(target_gpus) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::vector<const Device *> target_gpus_;
    int64_t num_inputs_ = 0;
    const Device *input_device_ = nullptr;
};

class ReduceAddCoalesced : public autograd::Function {
public:
    static constexpr char kType[] = "ReduceAddCoalescedFunction";

    explicit ReduceAddCoalesced(const Device *destination, int64_t num_inputs)
        : autograd::Function(kType), destination_(destination), num_inputs_(num_inputs) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const Device *destination_ = nullptr;
    std::vector<const Device *> target_gpus_;
    int64_t num_inputs_ = 0;
};
} // namespace infini_train::autograd

#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
class ThreadDDP : public Module {
public:
    ThreadDDP(const std::shared_ptr<Module> &module, int dim = 0);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                    const std::vector<std::shared_ptr<Tensor>> &targets, const std::shared_ptr<Module> &loss_fn,
                    Optimizer &optimizer) override;

private:
    int dim_ = 0;
    std::vector<const Device *> devices_;
    const Device *output_device_ = nullptr;
    const Device *src_device_ = nullptr;
    std::vector<std::shared_ptr<Module>> replicas_;
};
} // namespace infini_train::nn::parallel

#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
class ThreadDistributedDataParallel : public Module {
public:
    ThreadDistributedDataParallel(const std::shared_ptr<Module> &module, int dim = 0);

    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

    std::vector<std::shared_ptr<Tensor>> Buffers() const override;

    void To(const Device *device) override;

    void To(DataType dtype) override;

    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                    const std::vector<std::shared_ptr<Tensor>> &targets,
                    const std::shared_ptr<Module> &loss_fn) override;

private:
    int dim_ = 0;
    std::vector<const Device *> devices_;
    const Device *output_device_ = nullptr;
    const Device *src_device_ = nullptr;
    std::vector<std::shared_ptr<Module>> replicas_;
};
} // namespace infini_train::nn::parallel

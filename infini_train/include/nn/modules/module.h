#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Module {
public:
    virtual ~Module(){};

    std::vector<std::shared_ptr<Tensor>> Parameters() const;

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;

    void To(Device device);

protected:
    Device device_; // CPU by default
    std::unordered_map<std::string, std::unique_ptr<Module>> modules_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> parameters_;
};
} // namespace infini_train::nn

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

    Module *AddNamedLayer(const std::string &name, std::unique_ptr<Module> &&layer);
    std::unique_ptr<Module> &GetLayer(const std::string &name);

    void AddNamedParameter(const std::string &name, const std::vector<int64_t> &dims, const DataType dtype);
    std::vector<Tensor *> Parameters() const;
    Tensor *GetParameter(const std::string &name) const;

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;

    void To(Device device);

protected:
    virtual void ToImpl(Device device) {}

    Device device_; // CPU by default
    std::unordered_map<std::string, std::unique_ptr<Module>> named_layers_;
    std::unordered_map<std::string, std::unique_ptr<Tensor>> named_parameters_;
};
} // namespace infini_train::nn

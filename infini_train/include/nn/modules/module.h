#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Module {
public:
    static constexpr char kUndefinedType[] = "Undefined";

    Module(DataType dtype = DataType::kFLOAT32) : type_(kUndefinedType), dtype_(dtype) {}
    explicit Module(const std::string &type, DataType dtype = DataType::kFLOAT32) : type_(type), dtype_(dtype) {}

    virtual ~Module(){};

    const std::string &type() const;

    std::vector<std::shared_ptr<Tensor>> Parameters() const;
    bool has_parameter(const std::string &name) const;
    std::shared_ptr<Tensor> *mutable_parameter(const std::string &name);
    const std::shared_ptr<Tensor> &parameter(const std::string &name) const;

    std::vector<Module *> modules() const;
    Module *mutable_module(const std::string &name);
    const Module &module(const std::string &name) const;

    std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict() const;

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;

    virtual void To(Device device);

    virtual void To(DataType dtype);

    void Apply(std::function<void(Module *)> fn);

protected:
    Device device_;  // CPU by default
    DataType dtype_; // DataType::kFLOAT32 by default
    const std::string type_ = kUndefinedType;
    std::unordered_map<std::string, std::unique_ptr<Module>> modules_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> parameters_;
};
} // namespace infini_train::nn

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "infini_train/include/datatype.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn {
class Module;

namespace parallel::function {
std::vector<std::shared_ptr<Module>> Replicate(const std::shared_ptr<Module> &network,
                                               const std::vector<const Device *> &devices);
} // namespace parallel::function

class Module : public std::enable_shared_from_this<Module> {
public:
    static constexpr char kUndefinedType[] = "Undefined";

    explicit Module();
    explicit Module(const std::string &type);
    Module(const Module &) = default;

    virtual ~Module(){};

    const std::string &type() const;

    virtual std::vector<std::shared_ptr<Tensor>> Parameters() const;
    bool has_parameter(const std::string &name) const;
    std::shared_ptr<Tensor> *mutable_parameter(const std::string &name);
    const std::shared_ptr<Tensor> &parameter(const std::string &name) const;

    virtual std::vector<std::shared_ptr<Tensor>> Buffers() const;

    std::vector<std::shared_ptr<Module>> modules();
    std::shared_ptr<Module> mutable_module(const std::string &name);
    const Module &module(const std::string &name) const;

    std::unordered_map<std::string, std::shared_ptr<Tensor>> StateDict() const;

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors);

    virtual float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                            const std::vector<std::shared_ptr<Tensor>> &targets, const std::shared_ptr<Module> &loss_fn,
                            DataType dtype) {
        return 0.0f;
    };

    virtual std::vector<std::shared_ptr<Module>> BuildChunks(int pp_rank);

    virtual void To(const Device *device);

    virtual void To(DataType dtype);

    void Apply(std::function<void(Module *)> fn);

    virtual std::shared_ptr<Module> ReplicateForDataParallel(int device_idx) const;

protected:
    const Device *device_ = nullptr;
    const std::string type_ = kUndefinedType;
    std::unordered_map<std::string, std::shared_ptr<Module>> modules_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> parameters_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> buffers_;

private:
    std::unordered_map<std::string, std::shared_ptr<Module>>
    NamedModules(const std::string &prefix = "", bool remove_duplicate = true,
                 std::unordered_set<Module *> *memory = nullptr);

    friend std::vector<std::shared_ptr<Module>>
    parallel::function::Replicate(const std::shared_ptr<Module> &network, const std::vector<const Device *> &devices);
};

template <typename Derived> class CloneableModule : public Module {
public:
    CloneableModule() = default;
    explicit CloneableModule(const std::string &type) : Module(type) {}

    std::shared_ptr<Module> ReplicateForDataParallel(int device_idx) const override {
        return std::make_shared<Derived>(static_cast<const Derived &>(*this));
    }
};
} // namespace infini_train::nn

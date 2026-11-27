#pragma once

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::nn {
class Sequential : public CloneableModule<Sequential> {
 public:
  static constexpr char kType[] = "Sequential";
  // TODO(dcj): Use better ctor signature later.
  explicit Sequential(std::vector<std::shared_ptr<Module>> &&layers);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleDict : public CloneableModule<ModuleDict> {
 public:
  static constexpr char kType[] = "ModuleDict";
  // TODO(dcj): in torch, there is a dict with the order of insertion
  explicit ModuleDict(
      std::unordered_map<std::string, std::shared_ptr<Module>> modules);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleList : public CloneableModule<ModuleList> {
 public:
  static constexpr char kType[] = "ModuleList";

  explicit ModuleList(std::vector<std::shared_ptr<Module>> &&layers);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

  std::vector<std::shared_ptr<Module>>::iterator begin();
  std::vector<std::shared_ptr<Module>>::iterator end();
  std::vector<std::shared_ptr<Module>>::const_iterator begin() const;
  std::vector<std::shared_ptr<Module>>::const_iterator end() const;

  std::shared_ptr<Module> &operator[](std::size_t idx);

 private:
  std::vector<std::shared_ptr<Module>> module_list_;
};

class Flatten : public CloneableModule<Flatten> {
 public:
  static constexpr char kType[] = "Flatten";

  explicit Flatten(int64_t start_dim = 1, int64_t end_dim = -1);

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

 private:
  int64_t start_dim_ = 1;
  int64_t end_dim_ = -1;
};
}  // namespace infini_train::nn

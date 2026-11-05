#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/nn/modules/transformer/transformer_config.h"
namespace infini_train {
class Optimizer;
namespace nn {
class Module;
}
}  // namespace infini_train
namespace infini_train::regression {
class Output {
 public:
  Output(const std::string &metrics_file, const std::string &snapshot_dir,
         const std::string &model, const std::string &dtype,
         const nn::TransformerConfig &config, int rank = 0, int dp = 1,
         int tp = 1, int pp = 1, int sp = 1, int dp_rank = 0, int tp_rank = 0,
         int pp_rank = 0, int vpp = 1,
         const std::string &optimizer_type = "adam");

  bool enabled() const;

  void Step(int64_t step, double loss, double lr, int64_t global_tokens,
            double step_ms, size_t used_mb, size_t reserved_mb,
            size_t consumed_samples);

  void Complete(int64_t step, size_t consumed_samples);

  void Snapshot(int64_t step, const std::shared_ptr<nn::Module> &model,
                const Optimizer *optimizer = nullptr);

 private:
  std::string CanonicalName(std::string name) const;

  void WriteShard(std::ostream &out, const std::string &name,
                  const std::vector<int64_t> &shape) const;

  std::ofstream metrics_;
  std::string snapshot_dir_, context_;
  nn::TransformerConfig config_;
  int rank_, dp_, tp_, pp_, tp_rank_, pp_rank_;
  std::vector<int> global_layers_;
};
}  // namespace infini_train::regression

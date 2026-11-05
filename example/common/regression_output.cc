#include "example/common/regression_output.h"

#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/ddp/param_and_grad_buffer.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::regression {
namespace {
std::string JsonString(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      out << '\\' << c;
    } else if (c < 32) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<int>(c) << std::dec;
    } else {
      out << c;
    }
  }
  return out.str() + '"';
}

// Tensor::SaveAsNpy currently only accepts FP32. Diagnostic state must preserve
// integer counters (including values above 2^24) and double-precision state.
std::string SaveSnapshotNpy(const Tensor &cpu, std::ostream &out) {
  static_assert(std::endian::native == std::endian::little);
  const std::map<DataType, std::pair<std::string, std::string>> types = {
      {DataType::kFLOAT32, {"<f4", "float32"}},
      {DataType::kFLOAT64, {"<f8", "float64"}},
      {DataType::kINT64, {"<i8", "int64"}},
      {DataType::kUINT64, {"<u8", "uint64"}},
      {DataType::kINT32, {"<i4", "int32"}},
      {DataType::kUINT32, {"<u4", "uint32"}},
      {DataType::kINT16, {"<i2", "int16"}},
      {DataType::kUINT16, {"<u2", "uint16"}},
      {DataType::kINT8, {"|i1", "int8"}},
      {DataType::kUINT8, {"|u1", "uint8"}},
      {DataType::kBOOL, {"|b1", "bool"}}};
  const auto &[descriptor, dtype] = types.at(cpu.Dtype());
  std::ostringstream header;
  header << "{'descr': '" << descriptor
         << "', 'fortran_order': False, 'shape': (";
  for (auto dimension : cpu.Dims()) {
    header << dimension << ',';
  }
  header << "), }";
  auto text = header.str();
  text.append((64 - (10 + text.size() + 1) % 64) % 64, ' ');
  text += '\n';
  CHECK_LT(text.size(), 65536);
  out.write("\x93NUMPY\x01\x00", 8);
  out.put(static_cast<char>(text.size() & 255));
  out.put(static_cast<char>((text.size() >> 8) & 255));
  out.write(text.data(), text.size());
  out.write(static_cast<const char *>(cpu.DataPtr()), cpu.SizeInBytes());
  CHECK(out) << "Failed writing tensor snapshot";
  return dtype;
}

}  // namespace
Output::Output(const std::string &metrics_file, const std::string &snapshot_dir,
               const std::string &model, const std::string &dtype,
               const nn::TransformerConfig &config, int rank, int dp, int tp,
               int pp, int sp, int dp_rank, int tp_rank, int pp_rank, int vpp,
               const std::string &optimizer_type)
    : snapshot_dir_(snapshot_dir),
      config_(config),
      rank_(rank),
      dp_(dp),
      tp_(tp),
      pp_(pp),
      tp_rank_(tp_rank),
      pp_rank_(pp_rank) {
  std::ostringstream context;
  context << "\"schema_version\":1,\"model\":" << JsonString(model)
          << ",\"dtype\":" << JsonString(dtype) << ",\"rank\":" << rank
          << ",\"dp\":" << dp << ",\"tp\":" << tp << ",\"pp\":" << pp
          << ",\"sp\":" << sp << ",\"dp_rank\":" << dp_rank
          << ",\"tp_rank\":" << tp_rank << ",\"pp_rank\":" << pp_rank
          << ",\"vpp\":" << vpp << ",\"world_size\":" << dp * tp * pp
          << ",\"memory_resolution_bytes\":1048576,\"optimizer_type\":"
          << JsonString(optimizer_type) << ",\"optimizer_stateful\":"
          << (optimizer_type == "sgd" ? "false" : "true");
  context_ = context.str();
  if (!metrics_file.empty()) {
    const std::filesystem::path path(metrics_file + ".rank_" +
                                     std::to_string(rank) + ".jsonl");
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }
    metrics_.open(path, std::ios::trunc);
    CHECK(metrics_) << "Cannot open regression metrics: " << path;
    metrics_ << std::setprecision(17);
  }
  const auto info = nn::parallel::PipelineParallel::GetStageInfo(
      config.n_layer, pp, pp_rank, vpp);
  for (auto [begin, end] : info.layer_ranges_per_chunk) {
    for (int i = begin; i < end; ++i) {
      global_layers_.push_back(i);
    }
  }
}

bool Output::enabled() const {
  return metrics_.is_open() || !snapshot_dir_.empty();
}

void Output::Step(int64_t step, double loss, double lr, int64_t global_tokens,
                  double step_ms, size_t used_mb, size_t reserved_mb,
                  size_t consumed_samples) {
  if (!metrics_.is_open()) {
    return;
  }
  // JSON has no NaN/Inf literals: invalid metrics fail at their source.
  CHECK(std::isfinite(lr) && std::isfinite(step_ms) && step_ms > 0);
  if (pp_rank_ == pp_ - 1) {
    CHECK(std::isfinite(loss)) << "Non-finite training loss";
  }
  metrics_ << '{' << context_ << ",\"type\":\"step\",\"step\":" << step
           << ",\"loss\":";
  if (pp_rank_ == pp_ - 1) {
    metrics_ << loss;
  } else {
    metrics_ << "null";
  }
  metrics_ << ",\"lr\":" << lr << ",\"global_tokens\":" << global_tokens
           << ",\"step_ms\":" << step_ms
           << ",\"peak_memory_bytes\":" << used_mb * 1048576ULL
           << ",\"peak_reserved_bytes\":" << reserved_mb * 1048576ULL
           << ",\"consumed_samples\":" << consumed_samples << "}\n";
  metrics_.flush();
  CHECK(metrics_) << "Failed writing regression metrics";
}

void Output::Complete(int64_t step, size_t consumed_samples) {
  if (!metrics_.is_open()) {
    return;
  }
  metrics_ << '{' << context_ << ",\"type\":\"complete\",\"step\":" << step
           << ",\"consumed_samples\":" << consumed_samples << "}\n";
  metrics_.flush();
  CHECK(metrics_);
}

void Output::Snapshot(int64_t step, const std::shared_ptr<nn::Module> &model,
                      const Optimizer *optimizer) {
  if (snapshot_dir_.empty()) {
    return;
  }
  // Both ZeRO-1 and ZeRO-2 reduce-scatter gradients. Reconstruct only for
  // diagnostics; never change the
  // optimizer's shard storage or the parameter's own gradient binding.
  std::unordered_map<const Tensor *, std::shared_ptr<Tensor>> gradients;
  std::vector<std::shared_ptr<nn::Module>> roots{model};
  if (auto *pipeline =
          dynamic_cast<nn::parallel::PipelineParallel *>(model.get())) {
    roots = *pipeline->mutable_chunks();
  }
  std::set<const nn::Module *> visited;
  if (step > 0 && optimizer) {
    for (const auto &root : roots) {
      for (const auto &[_, module] : root->NamedModules()) {
        auto *ddp =
            dynamic_cast<nn::parallel::DistributedDataParallel *>(module.get());
        if (!ddp || !visited.insert(ddp).second ||
            ddp->ddp_config().zero_stage < 1) {
          continue;
        }
        const auto *pg = ddp->param_grad_buffers().front()->ddp_pg();
        for (const auto &group : ddp->bucket_groups()) {
          for (size_t i = 0; i < group->buckets().size(); ++i) {
            const auto &bucket = group->buckets()[i];
            std::shared_ptr<Tensor> shard;
            if (ddp->ddp_config().zero_stage >= 2) {
              shard = group->GetLocalGradShardBuffer(i);
            } else {
              // ZeRO-1 retains a full allocation, but only the
              // local reduce-scatter slice is globally reduced.
              const auto &storage = bucket->grad_data();
              CHECK_EQ(storage->NumElements() % dp_, 0);
              const auto elements = storage->NumElements() / dp_;
              const auto local_rank = pg->GetGroupRank(rank_);
              shard = std::make_shared<Tensor>(
                  *storage,
                  local_rank * elements * kDataTypeToSize.at(storage->Dtype()),
                  std::vector<int64_t>{static_cast<int64_t>(elements)});
            }
            auto full = std::make_shared<Tensor>(
                std::vector<int64_t>{
                    static_cast<int64_t>(shard->NumElements() * dp_)},
                shard->Dtype(), shard->GetDevice());
            pg->AllGather(full, shard);
            for (const auto &parameter : bucket->params()) {
              size_t begin = 0, end = 0;
              CHECK(bucket->GetTensorLocInBucket(parameter, begin, end));
              gradients[parameter.get()] = std::make_shared<Tensor>(
                  *full, begin * kDataTypeToSize.at(full->Dtype()),
                  parameter->Dims());
            }
          }
        }
      }
    }
  }
  const auto directory = std::filesystem::path(snapshot_dir_) /
                         ("step_" + std::to_string(step)) /
                         ("rank_" + std::to_string(rank_));
  std::filesystem::create_directories(directory);
  std::ofstream tensors(directory / "tensors.bin", std::ios::binary);
  CHECK(tensors);
  std::ofstream manifest(directory / "manifest.json");
  CHECK(manifest);
  manifest << '{' << context_ << ",\"step\":" << step << ",\"tensors\":[";
  size_t index = 0;
  std::unordered_map<const Tensor *, std::string> aliases;
  auto write = [&](const std::string &name, const std::string &kind,
                   const std::shared_ptr<Tensor> &tensor, bool trainable) {
    if (!tensor) {
      return;
    }
    if (index) {
      manifest << ',';
    }
    ++index;
    const auto offset = static_cast<uint64_t>(tensors.tellp());
    // Cast kernels require a shared-owned source. A temporary Tensor
    // returned by To(Device) cannot call shared_from_this().
    auto host = std::make_shared<Tensor>(tensor->To(Device()));
    host->set_grad(nullptr);
    core::GetDeviceGuardImpl(tensor->GetDevice().type())
        ->SynchronizeDevice(tensor->GetDevice());
    auto cpu = host->To(host->Dtype() == DataType::kBFLOAT16 ||
                                host->Dtype() == DataType::kFLOAT16
                            ? DataType::kFLOAT32
                            : host->Dtype());
    const auto storage_dtype = SaveSnapshotNpy(cpu, tensors);
    const auto length = static_cast<uint64_t>(tensors.tellp()) - offset;
    manifest << "{\"name\":" << JsonString(name)
             << ",\"kind\":" << JsonString(kind)
             << ",\"dtype\":" << JsonString(storage_dtype)
             << ",\"path\":\"tensors.bin\",\"offset\":" << offset
             << ",\"length\":" << length
             << ",\"trainable\":" << (trainable ? "true" : "false")
             << ",\"shape\":[";
    for (size_t i = 0; i < cpu.Dims().size(); ++i) {
      if (i) {
        manifest << ',';
      }
      manifest << cpu.Dims()[i];
    }
    manifest << ']';
    if (kind == "parameter") {
      if (aliases.contains(tensor.get())) {
        manifest << ",\"alias_of\":" << JsonString(aliases.at(tensor.get()));
      } else {
        aliases.emplace(tensor.get(), name);
      }
    }
    if (kind != "optimizer") {
      WriteShard(manifest, name, cpu.Dims());
    }
    manifest << '}';
  };
  std::set<const Tensor *> parameters;
  for (const auto &[_, parameter] : model->NamedParameters()) {
    parameters.insert(parameter.get());
  }
  // StateDict avoids internal PP aliases and has stable logical names.
  std::map<std::string, std::shared_ptr<Tensor>> state;
  for (const auto &[name, tensor] : model->StateDict()) {
    if (parameters.contains(tensor.get())) {
      state.emplace(CanonicalName(name), tensor);
    }
  }
  for (const auto &[name, tensor] : state) {
    write(name, "parameter", tensor, tensor->requires_grad());
    if (step > 0 && optimizer) {
      auto gradient = gradients.contains(tensor.get())
                          ? gradients.at(tensor.get())
                          : tensor->grad();
      write(name, "gradient", gradient, tensor->requires_grad());
    }
  }
  if (optimizer && step > 0) {
    std::map<std::string, std::shared_ptr<Tensor>> states;
    for (const auto &[name, tensor] : optimizer->StateDict()) {
      states.emplace(name, tensor);
    }
    for (const auto &[name, tensor] : states) {
      write("rank_" + std::to_string(rank_) + "." + name, "optimizer", tensor,
            false);
    }
  }
  tensors.flush();
  CHECK(tensors);
  manifest << "]}\n";
  manifest.flush();
  CHECK(manifest);
}

std::string Output::CanonicalName(std::string name) const {
  while (name.starts_with("module.")) {
    name.erase(0, 7);
  }
  const std::string prefix = "transformer.h.";
  if (name.starts_with(prefix)) {
    const auto end = name.find('.', prefix.size());
    const size_t local =
        std::stoul(name.substr(prefix.size(), end - prefix.size()));
    CHECK_LT(local, global_layers_.size());
    name.replace(prefix.size(), end - prefix.size(),
                 std::to_string(global_layers_[local]));
  }
  return name;
}

void Output::WriteShard(std::ostream &out, const std::string &name,
                        const std::vector<int64_t> &shape) const {
  if (tp_ == 1) {
    return;
  }
  int axis = -1;
  std::vector<int64_t> parts;
  const bool column = name.find(".c_attn.") != std::string::npos ||
                      name.find(".c_fc.") != std::string::npos;
  if (column && name.ends_with(".lora_A")) {
    return;
  }
  if (name.find(".c_proj.") != std::string::npos && name.ends_with(".lora_B")) {
    return;
  }
  if (name == "transformer.wte.weight" || name == "lm_head.weight") {
    axis = 0;
  } else if (name.find(".c_attn.") != std::string::npos) {
    axis = 0;
    const auto kv = config_.n_kv_head * config_.n_embd / config_.n_head;
    parts = {config_.n_embd, kv, kv};
  } else if (name.find(".c_fc.") != std::string::npos) {
    axis = 0;
    if (config_.activation_type == nn::MLPType::kSwiGLU) {
      parts = {shape[0] * tp_ / 2, shape[0] * tp_ / 2};
    }
  } else if (name.ends_with(".c_proj.weight") ||
             name.ends_with(".c_proj.lora_A")) {
    axis = 1;
  }
  if (axis < 0) {
    return;
  }
  out << ",\"shard\":{\"axis\":" << axis << ",\"index\":" << tp_rank_
      << ",\"count\":" << tp_;
  if (!parts.empty()) {
    out << ",\"part_sizes\":[";
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i) {
        out << ',';
      }
      out << parts[i];
    }
    out << ']';
  }
  out << '}';
}
}  // namespace infini_train::regression

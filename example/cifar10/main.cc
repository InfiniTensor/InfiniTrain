#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/comm.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/model_parallel_config.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "example/cifar10/checkpoint_io.h"
#include "example/cifar10/dataset.h"
#include "example/cifar10/models.h"
#include "example/common/cli.h"

DEFINE_string(dataset, "", "Path to CIFAR-10 binary directory");
DEFINE_string(model, "resnet18", "Model architecture: resnet18 | mobilenetv2");
DEFINE_double(width_mult, 1.0, "Width multiplier for MobileNetV2");
DEFINE_int32(bs, 128, "Per-rank batch size");
DEFINE_int32(epochs, 20, "Number of training epochs");
DEFINE_double(lr, 0.1, "Initial learning rate");
DEFINE_double(min_lr, 0.0, "Minimum learning rate for cosine schedule");
DEFINE_double(momentum, 0.9, "SGD momentum");
DEFINE_double(weight_decay, 5e-4, "SGD weight decay");
DEFINE_bool(augment, true, "Enable RandomCrop(pad=4) + RandomHorizontalFlip");
DEFINE_int32(seed, 42, "Random seed");
DEFINE_string(device, "cuda", "Device type: cpu | cuda");
DEFINE_int32(nthread_per_process, 1,
             "Number of threads/GPUs per process (DDP)");
DEFINE_bool(gradient_bucketing, true, "Enable DDP gradient bucketing");
DEFINE_string(save_ckpt, "", "Path to save training checkpoint (.bin)");
DEFINE_string(load_ckpt, "", "Path to load training checkpoint (.bin)");
DEFINE_int32(save_every, 0, "Save checkpoint every N epochs (0 = last only)");
DEFINE_bool(eval_only, false, "Run evaluation only");
DEFINE_int32(log_interval, 50, "Log training loss every N steps");
DEFINE_bool(print_shard_indices, false,
            "Print per-rank sample index summary for DDP verification");
DEFINE_int32(benchmark_steps, 0,
             "If > 0, run N synthetic train steps (plus 5 warmup) and report "
             "throughput");

// Alignment verification flags (used by verify_pytorch_alignment.py)
DEFINE_string(align_weights_in, "", "Input initial model weights (.bin)");
DEFINE_string(align_batch_in, "",
              "Input alignment batch file (.bin containing input_0, target_0, "
              "...)");
DEFINE_string(align_dump_dir, "", "Output directory for alignment dumps");
DEFINE_int32(align_steps, 1, "Number of optimization steps in alignment mode");
DEFINE_bool(align_eval_mode, false, "Run alignment forward in eval() mode");

using namespace infini_train;

namespace {

std::shared_ptr<nn::Module> CreateModel(const std::string &model_name,
                                        float width_mult, Device device) {
  if (model_name == "resnet18") {
    return std::make_shared<example::cifar10::ResNet18>(10, device);
  }
  if (model_name == "mobilenetv2") {
    return std::make_shared<example::cifar10::MobileNetV2>(10, width_mult,
                                                           device);
  }
  LOG(FATAL) << "Unsupported model: " << model_name
             << " (expected resnet18 or mobilenetv2)";
  return nullptr;
}

void BroadcastStateDictFromRank0(nn::Module *model, int ddp_rank,
                                 const nn::parallel::ProcessGroup *ddp_pg) {
  if (ddp_pg == nullptr) {
    return;
  }
  auto sd = model->StateDict();
  std::vector<std::string> keys;
  keys.reserve(sd.size());
  for (const auto &[k, _] : sd) {
    keys.push_back(k);
  }
  std::sort(keys.begin(), keys.end());
  for (const auto &k : keys) {
    auto &t = sd.at(k);
    if (ddp_rank != 0) {
      t->Fill(0.0f);
    }
    nn::parallel::comm::AllReduce(t, nn::parallel::comm::ReduceOpType::kSum,
                                  ddp_pg);
  }
}

void SynchronizeDeviceIfAccelerator(Device device) {
  if (!device.IsCPU()) {
    core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
  }
}

void RunAlignmentMode(const nn::parallel::Rank &rank, Device device,
                      int ddp_rank, int ddp_world_size,
                      const nn::parallel::ProcessGroup *ddp_pg) {
  auto raw_model =
      CreateModel(FLAGS_model, static_cast<float>(FLAGS_width_mult), Device());
  if (!FLAGS_align_weights_in.empty()) {
    auto init_sd = example::cifar10::LoadTensorMap(FLAGS_align_weights_in);
    raw_model->LoadStateDict(init_sd);
  }
  raw_model->To(device);
  BroadcastStateDictFromRank0(raw_model.get(), ddp_rank, ddp_pg);

  std::shared_ptr<nn::Module> model = raw_model;
  std::shared_ptr<nn::parallel::DistributedDataParallel> ddp_model = nullptr;
  if (ddp_world_size > 1) {
    nn::parallel::DistributedDataParallelConfig ddp_cfg;
    ddp_cfg.gradient_bucketing_enabled = FLAGS_gradient_bucketing;
    ddp_model = std::make_shared<nn::parallel::DistributedDataParallel>(
        raw_model, rank, ddp_cfg);
    model = ddp_model;
  }

  auto batch_map = example::cifar10::LoadTensorMap(FLAGS_align_batch_in);
  auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();
  loss_fn->To(device);

  auto named_params = raw_model->NamedParameters();
  auto optimizer = std::make_shared<optimizers::SGD>(
      named_params, static_cast<float>(FLAGS_lr),
      static_cast<float>(FLAGS_momentum),
      static_cast<float>(FLAGS_weight_decay));

  if (!FLAGS_load_ckpt.empty()) {
    example::cifar10::LoadTrainingCheckpoint(FLAGS_load_ckpt, raw_model.get(),
                                             optimizer.get());
  }

  if (FLAGS_align_eval_mode) {
    model->Eval();
  } else {
    model->Train();
  }

  std::unordered_map<std::string, std::shared_ptr<Tensor>> dumps;

  for (int step = 0; step < FLAGS_align_steps; ++step) {
    const std::string in_key =
        ddp_world_size > 1 ? std::format("input_{}_rank_{}", step, ddp_rank)
                           : std::format("input_{}", step);
    const std::string tgt_key =
        ddp_world_size > 1 ? std::format("target_{}_rank_{}", step, ddp_rank)
                           : std::format("target_{}", step);
    const std::string resolved_in =
        batch_map.contains(in_key) ? in_key : std::format("input_{}", step);
    const std::string resolved_tgt =
        batch_map.contains(tgt_key) ? tgt_key : std::format("target_{}", step);

    auto x = std::make_shared<Tensor>(batch_map.at(resolved_in)->To(device));
    auto y = std::make_shared<Tensor>(batch_map.at(resolved_tgt)->To(device));

    optimizer->ZeroGrad();
    auto logits = (*model)({x})[0];
    auto loss = loss_fn->Forward({logits, y})[0];

    if (step == 0 && ddp_rank == 0) {
      dumps["logits_0"] = std::make_shared<Tensor>(logits->To(Device()));
    }
    auto loss_cpu = std::make_shared<Tensor>(loss->To(Device()));
    if (ddp_rank == 0) {
      dumps[std::format("loss_{}", step)] = loss_cpu;
    }

    if (!FLAGS_align_eval_mode) {
      loss->Backward();
      if (ddp_model) {
        ddp_model->FinishGradSync();
      }
      if (step == 0 && ddp_rank == 0) {
        for (const auto &[name, param] : raw_model->NamedParameters()) {
          if (param->grad()) {
            dumps["grad." + name] =
                std::make_shared<Tensor>(param->grad()->To(Device()));
          }
        }
      }
      optimizer->Step();
    }
  }

  SynchronizeDeviceIfAccelerator(device);

  if (ddp_rank == 0) {
    if (!FLAGS_save_ckpt.empty()) {
      example::cifar10::SaveTrainingCheckpoint(
          FLAGS_save_ckpt, *raw_model, *optimizer,
          {.epoch = 0,
           .global_step = FLAGS_align_steps,
           .learning_rate = optimizer->learning_rate()});
    }
    for (const auto &[name, tensor] : raw_model->StateDict()) {
      dumps["state." + name] = std::make_shared<Tensor>(tensor->To(Device()));
    }
    if (!FLAGS_align_dump_dir.empty()) {
      const std::string out_path = FLAGS_align_dump_dir + "/align_output.bin";
      example::cifar10::SaveTensorMap(out_path, dumps);
    }
  }
}

void RunBenchmarkMode(const nn::parallel::Rank &rank, Device device,
                      int ddp_rank, int ddp_world_size,
                      const nn::parallel::ProcessGroup *ddp_pg) {
  auto raw_model =
      CreateModel(FLAGS_model, static_cast<float>(FLAGS_width_mult), Device());
  raw_model->To(device);
  BroadcastStateDictFromRank0(raw_model.get(), ddp_rank, ddp_pg);

  std::shared_ptr<nn::Module> model = raw_model;
  std::shared_ptr<nn::parallel::DistributedDataParallel> ddp_model = nullptr;
  if (ddp_world_size > 1) {
    nn::parallel::DistributedDataParallelConfig ddp_cfg;
    ddp_cfg.gradient_bucketing_enabled = FLAGS_gradient_bucketing;
    ddp_model = std::make_shared<nn::parallel::DistributedDataParallel>(
        raw_model, rank, ddp_cfg);
    model = ddp_model;
  }

  auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();
  loss_fn->To(device);
  auto optimizer = std::make_shared<optimizers::SGD>(
      raw_model->NamedParameters(), static_cast<float>(FLAGS_lr),
      static_cast<float>(FLAGS_momentum),
      static_cast<float>(FLAGS_weight_decay));

  auto x = std::make_shared<Tensor>(std::vector<int64_t>{FLAGS_bs, 3, 32, 32},
                                    DataType::kFLOAT32, device);
  x->Fill(0.5f);
  auto y_cpu = std::make_shared<Tensor>(std::vector<int64_t>{FLAGS_bs},
                                        DataType::kINT64);
  for (int i = 0; i < FLAGS_bs; ++i) {
    static_cast<int64_t *>(y_cpu->DataPtr())[i] = i % 10;
  }
  auto y = std::make_shared<Tensor>(y_cpu->To(device));

  model->Train();
  constexpr int kWarmup = 5;
  for (int i = 0; i < kWarmup; ++i) {
    optimizer->ZeroGrad();
    auto logits = (*model)({x})[0];
    auto loss = loss_fn->Forward({logits, y})[0];
    loss->Backward();
    if (ddp_model) {
      ddp_model->FinishGradSync();
    }
    optimizer->Step();
  }
  SynchronizeDeviceIfAccelerator(device);

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < FLAGS_benchmark_steps; ++i) {
    optimizer->ZeroGrad();
    auto logits = (*model)({x})[0];
    auto loss = loss_fn->Forward({logits, y})[0];
    loss->Backward();
    if (ddp_model) {
      ddp_model->FinishGradSync();
    }
    optimizer->Step();
  }
  SynchronizeDeviceIfAccelerator(device);
  const auto t1 = std::chrono::high_resolution_clock::now();

  if (ddp_rank == 0) {
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double total_samples =
        static_cast<double>(FLAGS_benchmark_steps) * FLAGS_bs * ddp_world_size;
    const double ms_per_step = (secs * 1e3) / FLAGS_benchmark_steps;
    const double samples_per_sec = total_samples / secs;
    LOG(ERROR) << std::format(
        "[BENCH] model={} device={} world_size={} bs_per_rank={} "
        "global_bs={} steps={} | {:.2f} ms/step | {:.1f} samples/s",
        FLAGS_model, FLAGS_device, ddp_world_size, FLAGS_bs,
        FLAGS_bs * ddp_world_size, FLAGS_benchmark_steps, ms_per_step,
        samples_per_sec);
  }
}

void Evaluate(const std::shared_ptr<nn::Module> &model,
              const example::cifar10::CIFAR10Dataset &test_dataset,
              const std::shared_ptr<nn::Module> &loss_fn, Device device,
              int batch_size, int epoch) {
  model->Eval();
  autograd::NoGradGuard no_grad;

  const size_t total = test_dataset.Size();
  std::vector<size_t> all_idx(total);
  std::iota(all_idx.begin(), all_idx.end(), 0);

  int64_t correct = 0;
  double loss_sum = 0.0;
  int64_t num_batches = 0;

  for (size_t pos = 0; pos < total; pos += static_cast<size_t>(batch_size)) {
    const size_t end = std::min(pos + static_cast<size_t>(batch_size), total);
    std::vector<size_t> batch_idx(
        all_idx.begin() + static_cast<ptrdiff_t>(pos),
        all_idx.begin() + static_cast<ptrdiff_t>(end));
    auto [imgs_cpu, lbls_cpu] =
        test_dataset.GetBatch(batch_idx, false, nullptr);
    auto imgs = std::make_shared<Tensor>(imgs_cpu->To(device));
    auto lbls = std::make_shared<Tensor>(lbls_cpu->To(device));

    auto logits = (*model)({imgs})[0];
    auto loss = loss_fn->Forward({logits, lbls})[0];

    auto logits_cpu = logits->To(Device());
    auto loss_cpu = loss->To(Device());
    loss_sum += static_cast<const float *>(loss_cpu.DataPtr())[0];
    ++num_batches;

    const int64_t bs = logits_cpu.Dims()[0];
    const float *logit_data = static_cast<const float *>(logits_cpu.DataPtr());
    const int64_t *lbl_data = static_cast<const int64_t *>(lbls_cpu->DataPtr());
    for (int64_t b = 0; b < bs; ++b) {
      const float *row = logit_data + b * 10;
      const int64_t pred = std::max_element(row, row + 10) - row;
      if (pred == lbl_data[b]) {
        ++correct;
      }
    }
  }

  const double acc =
      100.0 * static_cast<double>(correct) / static_cast<double>(total);
  LOG(ERROR) << std::format(
      "[EVAL] epoch {:2d} | test_loss {:.4f} | test_acc {:.2f}% ({}/{})", epoch,
      loss_sum / static_cast<double>(num_batches), acc, correct, total);
}

void TrainWorker(const nn::parallel::Rank &rank) {
  const auto device_type = Device::ParseType(FLAGS_device).value();
  nn::parallel::global::thread_global_rank = rank.GlobalRank();

  const int ddp_world_size = nn::parallel::global::GetDataParallelSize();
  int ddp_rank = 0;
  const nn::parallel::ProcessGroup *ddp_pg = nullptr;
  Device device;

  if (rank.IsParallel()) {
    CHECK(device_type != Device::DeviceType::kCPU)
        << "DDP requires CUDA accelerator";
    device = Device(device_type,
                    nn::parallel::global::GetDeviceIndex(rank.thread_rank()));
    auto *pg_factory =
        nn::parallel::ProcessGroupFactory::Instance(device.type());
    if (ddp_world_size > 1) {
      ddp_pg = pg_factory->GetOrCreate(
          nn::parallel::GetDataParallelProcessGroupName(rank.GlobalRank()),
          nn::parallel::GetDataParallelGroupRanks(rank.GlobalRank()));
      ddp_rank = ddp_pg->GetGroupRank(rank.GlobalRank());
    }
  } else {
    device = Device(device_type, 0);
  }

  core::DeviceGuard guard(device);

  if (!FLAGS_align_batch_in.empty()) {
    RunAlignmentMode(rank, device, ddp_rank, ddp_world_size, ddp_pg);
    return;
  }

  if (FLAGS_benchmark_steps > 0) {
    RunBenchmarkMode(rank, device, ddp_rank, ddp_world_size, ddp_pg);
    return;
  }

  CHECK(!FLAGS_dataset.empty())
      << "--dataset is required for CIFAR-10 training/evaluation";

  example::cifar10::CIFAR10Dataset train_dataset(FLAGS_dataset, true,
                                                 FLAGS_augment);
  example::cifar10::CIFAR10Dataset test_dataset(FLAGS_dataset, false, false);

  auto raw_model =
      CreateModel(FLAGS_model, static_cast<float>(FLAGS_width_mult), Device());
  raw_model->To(device);
  BroadcastStateDictFromRank0(raw_model.get(), ddp_rank, ddp_pg);

  std::shared_ptr<nn::Module> model = raw_model;
  std::shared_ptr<nn::parallel::DistributedDataParallel> ddp_model = nullptr;
  if (ddp_world_size > 1) {
    nn::parallel::DistributedDataParallelConfig ddp_cfg;
    ddp_cfg.gradient_bucketing_enabled = FLAGS_gradient_bucketing;
    ddp_model = std::make_shared<nn::parallel::DistributedDataParallel>(
        raw_model, rank, ddp_cfg);
    model = ddp_model;
  }

  auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();
  loss_fn->To(device);

  auto optimizer = std::make_shared<optimizers::SGD>(
      raw_model->NamedParameters(), static_cast<float>(FLAGS_lr),
      static_cast<float>(FLAGS_momentum),
      static_cast<float>(FLAGS_weight_decay));

  const size_t samples_per_rank =
      train_dataset.Size() / static_cast<size_t>(ddp_world_size);
  const int64_t steps_per_epoch = static_cast<int64_t>(
      ddp_world_size > 1
          ? (samples_per_rank / static_cast<size_t>(FLAGS_bs))
          : ((samples_per_rank + static_cast<size_t>(FLAGS_bs) - 1) /
             static_cast<size_t>(FLAGS_bs)));
  const int64_t total_steps =
      std::max<int64_t>(1, steps_per_epoch * FLAGS_epochs);

  TrainingLRSchedulerConfig sched_cfg;
  sched_cfg.lr = static_cast<float>(FLAGS_lr);
  sched_cfg.min_lr = static_cast<float>(FLAGS_min_lr);
  sched_cfg.lr_decay_style = "cosine";
  sched_cfg.lr_decay_iters = total_steps;
  sched_cfg.lr_warmup_iters = 0;
  auto scheduler = CreateLRScheduler(optimizer, sched_cfg);

  int start_epoch = 0;
  int64_t global_step = 0;
  if (!FLAGS_load_ckpt.empty()) {
    auto meta = example::cifar10::LoadTrainingCheckpoint(
        FLAGS_load_ckpt, raw_model.get(), optimizer.get());
    start_epoch = static_cast<int>(meta.epoch);
    global_step = meta.global_step;
    scheduler->Step(global_step);
    if (ddp_rank == 0) {
      LOG(ERROR) << std::format(
          "Resumed checkpoint {} from epoch={} global_step={} lr={:.5f}",
          FLAGS_load_ckpt, start_epoch, global_step,
          optimizer->learning_rate());
    }
  }

  if (FLAGS_eval_only) {
    if (ddp_rank == 0) {
      Evaluate(raw_model, test_dataset, loss_fn, device, FLAGS_bs, start_epoch);
    }
    return;
  }

  for (int epoch = start_epoch; epoch < FLAGS_epochs; ++epoch) {
    model->Train();
    auto shard_indices = train_dataset.GetShardedEpochIndices(
        epoch, ddp_rank, ddp_world_size, static_cast<uint64_t>(FLAGS_seed),
        /*shuffle=*/true);

    if (FLAGS_print_shard_indices && epoch == start_epoch) {
      LOG(ERROR) << std::format(
          "[SHARD] rank={}/{} epoch={} count={} first5=[{},{},{},{},{}] "
          "last={}",
          ddp_rank, ddp_world_size, epoch, shard_indices.size(),
          shard_indices[0], shard_indices[1], shard_indices[2],
          shard_indices[3], shard_indices[4], shard_indices.back());
    }

    std::mt19937 aug_rng(static_cast<uint32_t>(FLAGS_seed * 10007 +
                                               epoch * 997 + ddp_rank * 131));

    double epoch_loss = 0.0;
    int64_t epoch_steps = 0;
    int64_t epoch_samples = 0;

    SynchronizeDeviceIfAccelerator(device);
    const auto epoch_start = std::chrono::high_resolution_clock::now();

    for (int64_t step = 0; step < steps_per_epoch; ++step) {
      const size_t pos = static_cast<size_t>(step * FLAGS_bs);
      const size_t end =
          std::min(pos + static_cast<size_t>(FLAGS_bs), shard_indices.size());
      std::vector<size_t> batch_idx(
          shard_indices.begin() + static_cast<ptrdiff_t>(pos),
          shard_indices.begin() + static_cast<ptrdiff_t>(end));

      auto [imgs_cpu, lbls_cpu] =
          train_dataset.GetBatch(batch_idx, FLAGS_augment, &aug_rng);
      auto imgs = std::make_shared<Tensor>(imgs_cpu->To(device));
      auto lbls = std::make_shared<Tensor>(lbls_cpu->To(device));

      optimizer->ZeroGrad();
      auto logits = (*model)({imgs})[0];
      auto loss = loss_fn->Forward({logits, lbls})[0];
      loss->Backward();
      if (ddp_model) {
        ddp_model->FinishGradSync();
      }
      optimizer->Step();
      scheduler->Step();
      ++global_step;

      auto loss_cpu = loss->To(Device());
      const float loss_val = static_cast<const float *>(loss_cpu.DataPtr())[0];
      epoch_loss += loss_val;
      ++epoch_steps;
      epoch_samples += static_cast<int64_t>(batch_idx.size()) * ddp_world_size;

      if (ddp_rank == 0 && FLAGS_log_interval > 0 &&
          (step % FLAGS_log_interval == 0)) {
        LOG(ERROR) << std::format(
            "epoch {:2d}/{} step {:3d}/{} | loss {:.4f} | lr {:.5f}", epoch + 1,
            FLAGS_epochs, step + 1, steps_per_epoch, loss_val,
            optimizer->learning_rate());
      }
    }

    SynchronizeDeviceIfAccelerator(device);
    const auto epoch_end = std::chrono::high_resolution_clock::now();
    const double secs =
        std::chrono::duration<double>(epoch_end - epoch_start).count();

    if (ddp_rank == 0) {
      LOG(ERROR) << std::format(
          "[TRAIN] epoch {:2d}/{} | train_loss {:.4f} | lr {:.5f} | {:.2f}s | "
          "{:.1f} samples/s",
          epoch + 1, FLAGS_epochs,
          epoch_loss / static_cast<double>(epoch_steps),
          optimizer->learning_rate(), secs,
          static_cast<double>(epoch_samples) / secs);

      Evaluate(raw_model, test_dataset, loss_fn, device, FLAGS_bs, epoch + 1);

      const bool should_save =
          !FLAGS_save_ckpt.empty() &&
          ((FLAGS_save_every > 0 && (epoch + 1) % FLAGS_save_every == 0) ||
           (epoch + 1 == FLAGS_epochs));
      if (should_save) {
        example::cifar10::SaveTrainingCheckpoint(
            FLAGS_save_ckpt, *raw_model, *optimizer,
            {.epoch = epoch + 1,
             .global_step = global_step,
             .learning_rate = optimizer->learning_rate()});
        LOG(ERROR) << "Saved checkpoint to: " << FLAGS_save_ckpt;
      }
    }
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage(
      "Train ResNet-18 or MobileNetV2 on CIFAR-10 with InfiniTrain.\nUsage: " +
      std::string(argv[0]) +
      " --dataset=<cifar10_bin_dir> --model=resnet18|mobilenetv2 [options]");
  example::ParseModelFlags(&argc, &argv, "example/cifar10/");
  google::InitGoogleLogging(argv[0]);

  nn::parallel::ModelParallelConfig mp_config;
  nn::parallel::global::InitAllEnv(FLAGS_nthread_per_process, mp_config);

  if (FLAGS_nthread_per_process > 1) {
    std::vector<std::thread> threads;
    for (int idx = 0; idx < FLAGS_nthread_per_process; ++idx) {
      nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), idx,
                              nn::parallel::global::GetNprocPerNode(),
                              FLAGS_nthread_per_process);
      threads.emplace_back(TrainWorker, rank);
    }
    for (auto &t : threads) {
      t.join();
    }
  } else {
    nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), 0,
                            nn::parallel::global::GetNprocPerNode(),
                            FLAGS_nthread_per_process);
    TrainWorker(rank);
  }

  gflags::ShutDownCommandLineFlags();
  google::ShutdownGoogleLogging();
  return 0;
}

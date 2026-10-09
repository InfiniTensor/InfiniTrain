#include "infini_train/include/training/training.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_set>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/checkpoint/checkpoint_manager.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/nn/lora/lora_utils.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/utils/global_module_hook_registry.h"
#include "infini_train/include/utils/precision_check_config.h"
#include "infini_train/include/utils/precision_checker.h"
#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif

namespace infini_train::training {
namespace {

void ValidateOptions(const TrainingOptions &options) {
    CHECK(Device::ParseType(options.device).has_value()) << "Unsupported device: " << options.device;
    CHECK(options.dtype == "float32" || options.dtype == "bfloat16") << "Unsupported dtype: " << options.dtype;
    CHECK(options.optimizer == "sgd" || options.optimizer == "adam") << "Unsupported optimizer: " << options.optimizer;
    CHECK_GT(options.batch_size, 0);
    CHECK_GT(options.sequence_length, 0);
    CHECK_GT(options.total_batch_size, 0);
    CHECK_GT(options.nthread_per_process, 0);
    CHECK_GT(options.tensor_parallel, 0);
    CHECK_GT(options.pipeline_parallel, 0);
    CHECK_GT(options.virtual_pipeline_parallel, 0);
    CHECK_GE(options.zero_stage, 0);
    CHECK_LE(options.zero_stage, 3);
    CHECK_EQ(options.save.empty(), options.save_interval == 0) << "--save and --save_interval must be set together";
}

std::shared_ptr<Optimizer> SetupOptimizer(const TrainingOptions &options, const std::shared_ptr<nn::Module> &model,
                                          int ddp_world_size, int ddp_rank, int pp_world_size) {
    using namespace nn::parallel;
    auto optimizer_creator = options.optimizer == "sgd" ? optimizers::SGD::CreateNamed(options.learning_rate)
                                                        : optimizers::Adam::CreateNamed(options.learning_rate);
    std::shared_ptr<Optimizer> optimizer = nullptr;

    std::vector<std::shared_ptr<Tensor>> params_to_optimize;
    if (options.lora_rank > 0) {
        params_to_optimize = nn::lora::GetLoRAParameters(model);
        LOG(INFO) << "Optimizing " << params_to_optimize.size() << " LoRA parameters";
    } else {
        params_to_optimize = model->Parameters();
        LOG(INFO) << "Optimizing " << params_to_optimize.size() << " model parameters";
    }
    std::unordered_set<const Tensor *> params_to_optimize_set;
    params_to_optimize_set.reserve(params_to_optimize.size());
    for (const auto &param : params_to_optimize) { params_to_optimize_set.insert(param.get()); }

    NamedParameterList named_parameters;
    for (const auto &[name, param] : model->NamedParameters()) {
        if (params_to_optimize_set.contains(param.get())) {
            named_parameters.emplace_back(name, param);
        }
    }
    CHECK_EQ(named_parameters.size(), params_to_optimize.size());

    if (options.zero_stage >= 1) {
        auto model_chunks = (pp_world_size > 1)
                              ? *(dynamic_cast<nn::parallel::PipelineParallel *>(model.get())->mutable_chunks())
                              : std::vector<std::shared_ptr<nn::Module>>{model};
        optimizer = std::make_shared<nn::parallel::DistributedOptimizer>(optimizer_creator, named_parameters,
                                                                         model_chunks, ddp_world_size, ddp_rank);
    } else {
        optimizer = optimizer_creator(named_parameters);
    }

    return optimizer;
}

std::shared_ptr<LRScheduler> SetupLRScheduler(const TrainingOptions &options,
                                              const std::shared_ptr<Optimizer> &optimizer) {
    const int64_t lr_decay_iters = options.lr_decay_iters > 0 ? options.lr_decay_iters : options.num_iteration;
    TrainingLRSchedulerConfig sched_config;
    sched_config.lr = static_cast<float>(options.learning_rate);
    sched_config.min_lr = static_cast<float>(options.min_lr);
    sched_config.lr_decay_style = options.lr_decay_style;
    sched_config.lr_decay_iters = lr_decay_iters;
    sched_config.lr_warmup_iters = options.lr_warmup_iters;
    sched_config.lr_warmup_init = static_cast<float>(options.lr_warmup_init);
    return CreateLRScheduler(optimizer, sched_config);
}

void Train(const TrainingOptions &options, const nn::parallel::Rank &rank, const DatasetProvider &dataset_provider,
           const ModelProvider &model_provider, const ForwardStep &forward_step, const AfterStep &after_step) {
    using namespace nn::parallel;

    // select the device
    Device device;
    const auto device_type = Device::ParseType(options.device).value();

    int ddp_world_size = global::GetDataParallelSize();
    int tp_world_size = global::GetTensorParallelSize();
    int sp_world_size = global::GetSequenceParallelEnabled() ? tp_world_size : 1;
    int pp_world_size = global::GetPipelineParallelSize();

    if (options.sequence_parallel) {
        CHECK_EQ(options.sequence_length % tp_world_size, 0)
            << "sequence_length must be divisible by tp_world_size when SP is enabled (pad later if needed).";
    }

    int ddp_rank = 0;
    int tp_rank = 0;
    int pp_rank = 0;

    // Set thread-local global rank
    nn::parallel::global::thread_global_rank = rank.GlobalRank();

    const ProcessGroup *ddp_pg = nullptr;
    const ProcessGroup *tp_pg = nullptr;
    const ProcessGroup *pp_pg = nullptr;

    if (rank.IsParallel()) {
        CHECK(device_type != Device::DeviceType::kCPU) << "Parallel training requires an accelerator backend";
        device = Device(device_type, global::GetDeviceIndex(rank.thread_rank()));
        auto *pg_factory = ProcessGroupFactory::Instance(device.type());

        if (ddp_world_size > 1) {
            ddp_pg = pg_factory->GetOrCreate(GetDataParallelProcessGroupName(rank.GlobalRank()),
                                             GetDataParallelGroupRanks(rank.GlobalRank()));
            ddp_rank = ddp_pg->GetGroupRank(rank.GlobalRank());
        }

        if (tp_world_size > 1) {
            tp_pg = pg_factory->GetOrCreate(GetTensorParallelProcessGroupName(rank.GlobalRank()),
                                            GetTensorParallelGroupRanks(rank.GlobalRank()));
            tp_rank = tp_pg->GetGroupRank(rank.GlobalRank());
            // NOTE(zbl): Reserved for VocabParallelEmbedding
            nn::parallel::tp_rank = tp_rank;
        }

        if (pp_world_size > 1) {
            pp_pg = pg_factory->GetOrCreate(GetPipelineParallelProcessGroupName(rank.GlobalRank()),
                                            GetPipelineParallelGroupRanks(rank.GlobalRank()));
            pp_rank = pp_pg->GetGroupRank(rank.GlobalRank());

            nn::parallel::pp_rank = pp_rank;
        }
    } else {
        device = Device(device_type, 0);
    }

    // calculate gradient accumulation from the desired total batch size and the current run configuration
    const auto tokens_per_fwdbwd = static_cast<uint64_t>(options.batch_size) * options.sequence_length * ddp_world_size;
    CHECK_EQ(options.total_batch_size % tokens_per_fwdbwd, 0);
    const auto grad_accum_steps = options.total_batch_size / tokens_per_fwdbwd;
    if (rank.IsMainRank()) {
        LOG(INFO) << "total desired batch size: " << options.total_batch_size
                  << " => calculated gradient accumulation steps: " << grad_accum_steps;
    }

    auto base_model = model_provider();
    CHECK(base_model) << "model_provider returned a null model";
    const auto model_config = base_model->Config();
    CHECK_LE(options.sequence_length, model_config.block_size);
    CHECK_GT(model_config.GetChunkSize(), 0) << "Every pipeline rank must own a transformer chunk";
    std::shared_ptr<nn::Module> model = base_model;
    ForwardStepFunction rank_forward_step
        = [&](nn::Module &module, const TensorList &inputs, const std::shared_ptr<Tensor> &target) {
              return forward_step(module, inputs, target, model_config);
          };

    model->To(device);

    if (options.synchronize_device) {
        core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
    }

    utils::PrecisionChecker::BuildNameMap(model.get());

    // Apply LoRA using GetLoRAModel (in-place injection)
    bool lora_enabled = options.lora_rank > 0;
    if (lora_enabled) {
        nn::lora::LoRAConfig lora_config{options.lora_rank, static_cast<float>(options.lora_alpha), 0.0f,
                                         nn::lora::ParseLoRATargetModules(options.lora_target_modules)};

        // GetLoRAModel: in-place injection, modifies module tree directly
        model = nn::lora::GetLoRAModel(model, lora_config);

        // Load LoRA weights if specified
        if (!options.lora_load_path.empty()) {
            LOG(INFO) << "Loading LoRA weights from: " << options.lora_load_path;
            nn::lora::LoadLoRAWeights(model, options.lora_load_path);
        }

        // Print LoRA summary
        nn::lora::PrintLoRASummary(model, rank.GlobalRank());
    }

    LOG(INFO) << "Rank " << rank.GlobalRank() << ": Model loaded to device.";

    DataType dtype;
    if (options.dtype == "float32") {
        dtype = DataType::kFLOAT32;
    } else if (options.dtype == "bfloat16") {
        dtype = DataType::kBFLOAT16;
    } else {
        LOG(FATAL) << "Rank " << rank.GlobalRank() << ": Datatype " << options.dtype << " not supported.";
    }

    const auto num_micro_batches = grad_accum_steps;

    if (pp_world_size > 1) {
        // NOTE(dcj): To ensure that the tensor shapes at the pipeline stage boundaries remain correct
        // when sequence parallelism (SP) is enabled, we need to divide by sp_world_size.
        auto shapes = std::vector<std::vector<int64_t>>{
            {options.batch_size, options.sequence_length / sp_world_size, model_config.n_embd}};

        model
            = std::make_shared<nn::parallel::PipelineParallel>(model, pp_world_size, num_micro_batches, shapes, pp_rank,
                                                               device, model_config.GetChunkSize(), rank_forward_step);
        if (ddp_world_size > 1) {
            auto ddp_config = DistributedDataParallelConfig{.zero_stage = options.zero_stage};
            auto *mutable_chunks = dynamic_cast<nn::parallel::PipelineParallel *>(model.get())->mutable_chunks();
            for (int chunk_id = 0; chunk_id < mutable_chunks->size(); ++chunk_id) {
                (*mutable_chunks)[chunk_id]
                    = std::make_shared<DistributedDataParallel>(mutable_chunks->at(chunk_id), rank, ddp_config);
            }
        }
    } else if (ddp_world_size > 1) {
        // NOTE(dcj): Complete all device (.to(device)) and dtype (.to(dtype)) conversions
        // before wrapping the model with DistributedDataParallel (DDP).
        // Otherwise, DDP’s gradient hooks may be lost because new parameter tensors
        // are created during the conversion.

        auto ddp_config = DistributedDataParallelConfig{.zero_stage = options.zero_stage};
        model = std::make_shared<DistributedDataParallel>(model, rank, ddp_config);
    }

    const size_t train_loader_batch_size
        = pp_world_size > 1 ? options.batch_size * num_micro_batches : options.batch_size;
    const auto datasets = dataset_provider(static_cast<size_t>(options.num_iteration)
                                           * (options.total_batch_size / options.sequence_length));
    CHECK(datasets.train) << "dataset_provider must return a training dataset";
    DistributedDataLoader train_loader(datasets.train, train_loader_batch_size, ddp_rank, ddp_world_size);
    std::optional<DistributedDataLoader> val_loader;
    if (datasets.valid) {
        val_loader.emplace(datasets.valid, options.batch_size, ddp_rank, ddp_world_size);
    }

    auto optimizer = SetupOptimizer(options, model, ddp_world_size, ddp_rank, pp_world_size);
    auto scheduler = SetupLRScheduler(options, optimizer);

    auto train_iter = train_loader.begin();
    LOG(INFO) << "Rank " << rank.GlobalRank() << ": start training";

    auto impl = core::GetDeviceGuardImpl(device.type());

    int start_step = 0;
    TrainerState state;
    const auto resume_result = ResumeFromCheckpoint({.resume_root = options.load,
                                                     .rank = rank,
                                                     .model = model,
                                                     .optimizer = options.load_optimizer_state ? optimizer : nullptr,
                                                     .model_config = model_config,
                                                     .state = state,
                                                     .lr_scheduler = scheduler});

    start_step = resume_result.global_step;
    size_t consumed_train_samples = resume_result.consumed_train_samples;

    auto advance_train_iter = [&]() {
        ++train_iter;
        if (train_iter == train_loader.end()) {
            train_iter = train_loader.begin();
        }
    };

    // TODO(jym): Move resume position handling into a Sampler abstraction when available.
    if (consumed_train_samples > 0) {
        const size_t num_skips
            = DataLoaderBatchesToSkip(consumed_train_samples, train_loader_batch_size, ddp_world_size);
        for (size_t i = 0; i < num_skips; ++i) { advance_train_iter(); }
    }

    auto next_train_batch = [&]() {
        auto batch = *train_iter;
        // if we are trying to overfit a single batch, we reset the loader here by commenting out the line below
        // TODO(dcj): support dataloader.reset() later
        advance_train_iter();
        consumed_train_samples += train_loader_batch_size * ddp_world_size;
        return batch;
    };

    auto save_checkpoint = [&](const std::filesystem::path &save_dir, int64_t global_step) {
        SaveCheckpoint({
            .save_dir = save_dir,
            .global_step = global_step,
            .consumed_train_samples = consumed_train_samples,
            .n_layer = model_config.n_layer,
            .n_head = model_config.n_head,
            .n_kv_head = model_config.n_kv_head,
            .n_embd = model_config.n_embd,
            .vocab_size = model_config.vocab_size,
            .ddp_size = ddp_world_size,
            .tp_size = tp_world_size,
            .sp_size = sp_world_size,
            .pp_size = pp_world_size,
            .checkpoint_root_dir = options.save,
            .max_checkpoint_keep = options.max_checkpoint_keep,
            .rank = rank,
            .model = *model,
            .optimizer = options.save_optimizer_state ? optimizer.get() : nullptr,
            .lr_scheduler = scheduler.get(),
        });
    };

    for (int64_t step = start_step; step <= options.num_iteration; ++step) {
        // Reset precision check counters at start of each iteration for file overwrite
        utils::PrecisionChecker::ResetCounters();

        const bool last_step = step == options.num_iteration;

        impl->ResetMemPoolHighWatermarks(device);

        const auto iter_start = std::chrono::high_resolution_clock::now();

        // once in a while evaluate the validation dataset
        if (options.val_loss_every > 0 && (step % options.val_loss_every == 0 || last_step) && val_loader.has_value()) {
            // TODO(dcj): implement this after model.eval() is supported
        }
        // once in a while perform model inference on the master process
        if (options.sample_every > 0 && (step % options.sample_every == 0 || last_step)) {
            // TODO(dcj): implement this after model.eval() is supported
        }

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= num_iterations
        // instead of just < num_iterations (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) {
            break;
        }

#ifdef PROFILE_MODE
        Profiler::Instance().SetTag("Step_" + std::to_string(step));
#endif

        const float current_lr = scheduler ? scheduler->learning_rate() : static_cast<float>(options.learning_rate);
        float lossf = 0.0f;
        if (pp_world_size == 1) {
            // model->Train();
            optimizer->ZeroGrad();

            // if we are trying to overfit a single batch, we reset the loader here
            if (options.overfit_single_batch) {
                // train_loader.Reset();
            }

            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                // enable autocast for the current step
                infini_train::AutocastGuard autocast_guard(device.type(), dtype);

                // (bs, seq_len), (bs, seq_len)
                auto [x, y] = next_train_batch();
                x = std::make_shared<Tensor>(x->To(device));
                y = std::make_shared<Tensor>(y->To(device));

                LOG(INFO) << "Rank " << rank.GlobalRank() << ": start forward";
                // (bs, seq_len, vocab_size)
                auto result = rank_forward_step(*model, {x}, y);
                CHECK(result.loss_func) << "forward_step must return a loss callback";
                LOG(INFO) << "Rank " << rank.GlobalRank() << ": finish model forward, start loss forward";
                auto loss = result.loss_func(result.output);
                // FIXME(jym): verify gradient accumulation precision
                loss = loss / grad_accum_steps;

                // disable autocast for the current step (backward is not under autocast)
                autocast_guard.Disable();

                LOG(INFO) << "Rank " << rank.GlobalRank() << ": finish loss forward";

                LOG(INFO) << "Rank " << rank.GlobalRank() << ": start backward";
                std::unique_ptr<nn::NoSyncGuard> no_sync_guard;
                if (ddp_world_size > 1 && micro_step != grad_accum_steps - 1) {
                    no_sync_guard = model->no_sync();
                }
                loss->Backward();
                // Defer the loss D2H copy until after backward; reading it earlier would synchronize CUDA
                // between forward and backward.
                auto loss_cpu = loss->To(Device());
                lossf += static_cast<const float *>(loss_cpu.DataPtr())[0];
                LOG(INFO) << "Rank " << rank.GlobalRank() << ": finish backward";
            }

            optimizer->Step();
            if (scheduler) {
                scheduler->Step();
            }
        } else {
            auto [x, y] = next_train_batch();
            x = std::make_shared<Tensor>(x->To(device));
            y = std::make_shared<Tensor>(y->To(device));

            lossf = model->TrainStep({x}, {y}, optimizer, nullptr, dtype);
            if (scheduler) {
                scheduler->Step();
            }
        }

        if (ddp_world_size > 1) {
            auto lossf_tensor = std::make_shared<Tensor>(&lossf, std::vector<int64_t>{}, DataType::kFLOAT32, device);
            function::AllReduce(lossf_tensor, function::ReduceOpType::kAvg, ddp_pg);
            lossf = static_cast<const float *>(lossf_tensor->To(Device()).DataPtr())[0];
        }

        const auto iter_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(iter_end - iter_start).count();
        const double tps = options.total_batch_size / (duration_us / 1e6);

        if (rank.IsLastRank()) {
            size_t used_mb = 0, reserved_mb = 0;
            std::tie(used_mb, reserved_mb) = impl->GetMemPoolPeakMB(device);
            LOG(ERROR) << std::format("step {:4d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} tok/s | "
                                      "peak used: {:5d} MB | peak reserved: {:5d} MB, DP={}, TP={}, SP={}, PP={})",
                                      step + 1, options.num_iteration, lossf, current_lr, duration_us / 1e3f, tps,
                                      used_mb, reserved_mb, ddp_world_size, tp_world_size, sp_world_size,
                                      pp_world_size);

            if (after_step) {
                after_step(*model, device, step + 1);
            }
        }

        if (!options.save.empty() && options.save_interval > 0) {
            if ((step + 1) % options.save_interval == 0 || (step + 1) == options.num_iteration) {
                std::filesystem::path step_dir
                    = std::filesystem::path(options.save) / std::format("checkpoint_step_{:06d}", step + 1);
                if (rank.IsParallel()) {
                    step_dir /= std::format("rank_{:06d}", rank.GlobalRank());
                }
                save_checkpoint(step_dir, step + 1);
            }
        }
    }

    // Save LoRA weights if enabled and path specified
    if (lora_enabled && !options.lora_save_path.empty()) {
        LOG(INFO) << "Saving LoRA weights to: " << options.lora_save_path;
        nn::lora::SaveLoRAWeights(model, options.lora_save_path);
    }

#ifdef PROFILE_MODE
    Profiler::Instance().Report(options.profile_name + ".report", Profiler::SortBy::DeviceTimePercentage);
    Profiler::Instance().PrintRecords(options.profile_name + ".records.log");
#endif

    if (options.synchronize_device) {
        impl->SynchronizeDevice(device);
    }
}

} // namespace

void Pretrain(const TrainingOptions &options, const DatasetProvider &dataset_provider,
              const ModelProvider &model_provider, const ForwardStep &forward_step, const AfterStep &after_step) {
    ValidateOptions(options);
    CHECK(dataset_provider);
    CHECK(model_provider);
    CHECK(forward_step);
    auto precision_config = utils::PrecisionCheckConfig::Parse(options.precision_check);
    nn::parallel::global::InitAllEnv(options.nthread_per_process, options.tensor_parallel, options.sequence_parallel,
                                     options.pipeline_parallel, options.virtual_pipeline_parallel);
    utils::PrecisionCheckEnv::Instance().Init(precision_config);
    LOG(INFO) << nn::parallel::global::ProcessGroupOverview();

    auto run_rank = [&](int thread_rank) {
        nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), thread_rank,
                                nn::parallel::global::GetNprocPerNode(), options.nthread_per_process);
        Train(options, rank, dataset_provider, model_provider, forward_step, after_step);
    };
    if (options.nthread_per_process > 1) {
        std::vector<std::thread> threads;
        for (int idx = 0; idx < options.nthread_per_process; ++idx) { threads.emplace_back(run_rank, idx); }
        for (auto &thread : threads) { thread.join(); }
    } else {
        run_rank(0);
    }
}

} // namespace infini_train::training

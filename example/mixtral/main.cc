#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
// Out-of-tree builds inject the selected provider declaration without adding
// a vendor dependency to the upstream example.
#ifdef INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_HEADER
#include INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_HEADER
#endif

#include "infini_train/include/autocast.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"
#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/mixtral/checkpoint_loader.h"
#include "example/mixtral/config.h"

// I/O
DEFINE_string(input_bin, "", "input .bin to train on");
DEFINE_string(llmc_filepath, "", "llmc model file path to load from");
DEFINE_string(model_config, std::string(mixtral::kTinyModelPreset), "Model preset (tiny/mixtral-8x7b)");
// token layout for each step of the optimization
DEFINE_uint32(micro_batch_size, 4, "batch size, in units of #batch dimensions");
DEFINE_uint32(global_batch_size, 4, "total desired batch size, in units of #batch dimensions");
DEFINE_uint32(sequence_length, 64, "sequence length");
// workload (number of steps)
DEFINE_uint32(num_iteration, 10, "number of iterations to run");
// optimization
DEFINE_double(learning_rate, 1e-4, "Peak learning rate.");
DEFINE_int32(zero_stage, 0, "ZeRO stage (0/1/2); 0 disables DistributedOptimizer");
// memory management
DEFINE_string(device, "cpu", "Training device.");
// parallel
DEFINE_int32(nthread_per_process, 1,
             "Number of threads to use for each process. "
             "When set > 1, enables data parallelism on the specified accelerator devices.");
DEFINE_uint32(tensor_parallel, 1, "Tensor Parallel world size");
DEFINE_uint32(pipeline_parallel, 1, "Pipeline Parallel world size, specified the number of PP stages.");
DEFINE_uint32(virtual_pipeline_parallel, 1, "Number of chunks in PP stage.");
DEFINE_uint32(expert_parallel, 1, "Expert Parallel world size");
DEFINE_uint32(expert_tensor_parallel, 1, "Expert Tensor Parallel world size (currently must be 1)");
// precision
DEFINE_string(dtype, "float32", "precision used in training (float32/bfloat16)");
// logging
DEFINE_uint32(log_interval, 1, "Print train loss every N steps. 0 disables step loss logging.");
DEFINE_bool(print_timing, false, "Print training-loop elapsed time and token throughput.");

using namespace infini_train;

namespace {

constexpr char kDtypeFP32[] = "float32";
constexpr char kDtypeBF16[] = "bfloat16";

void ValidateRuntimeFlags(const nn::TransformerConfig &config) {
    CHECK(!FLAGS_input_bin.empty()) << "Mixtral training requires --input_bin";
    CHECK_GT(FLAGS_micro_batch_size, 0);
    CHECK_GT(FLAGS_global_batch_size, 0);
    CHECK_GT(FLAGS_sequence_length, 0);
    CHECK_LE(FLAGS_sequence_length, config.block_size) << "sequence_length must be <= model max positions (block_size)";
    CHECK(config.moe_config.has_value());
    CHECK_EQ(config.moe_config->num_experts % FLAGS_expert_parallel, 0)
        << "The number of Mixtral experts must be divisible by expert_parallel";
    if (FLAGS_expert_parallel > 1 && nn::parallel::global::GetExpertDataParallelSize() > 1) {
        CHECK_EQ(FLAGS_zero_stage, 1)
            << "The initial EP+EDP training path requires --zero_stage=1 so every expert replica participates "
               "in gradient synchronization, including replicas with no routed tokens";
    }

    const uint64_t batch_per_micro_step
        = static_cast<uint64_t>(FLAGS_micro_batch_size) * nn::parallel::global::GetDataParallelSize();
    CHECK_EQ(FLAGS_global_batch_size % batch_per_micro_step, 0)
        << "global_batch_size must be divisible by micro_batch_size * data_parallel_size";
}

} // namespace

DEFINE_validator(device, [](const char *, const std::string &value) { return Device::ParseType(value).has_value(); });
DEFINE_validator(zero_stage, [](const char *, int32_t value) { return value >= 0 && value <= 2; });
DEFINE_validator(model_config, [](const char *, const std::string &value) {
    return value == mixtral::kTinyModelPreset || value == mixtral::kMixtral8x7BModelPreset;
});

void Train(const nn::parallel::Rank &rank) {
    using namespace nn::parallel;
    const auto device_type = Device::ParseType(FLAGS_device).value();

    global::thread_global_rank = rank.GlobalRank();

    const int ddp_world_size = global::GetDataParallelSize();
    const int pp_world_size = global::GetPipelineParallelSize();
    int ddp_rank = 0;
    int pp_rank = 0;
    Device device;

    if (rank.IsParallel()) {
        device = Device(device_type, global::GetDeviceIndex(rank.thread_rank()));
        auto *pg_factory = ProcessGroupFactory::Instance(device.type());
        if (ddp_world_size > 1) {
            const auto *ddp_pg = pg_factory->GetOrCreate(GetDataParallelProcessGroupName(rank.GlobalRank()),
                                                         GetDataParallelGroupRanks(rank.GlobalRank()));
            CHECK_NOTNULL(ddp_pg);
            ddp_rank = ddp_pg->GetGroupRank(rank.GlobalRank());
        }
        if (pp_world_size > 1) {
            const auto *pp_pg = pg_factory->GetOrCreate(GetPipelineParallelProcessGroupName(rank.GlobalRank()),
                                                        GetPipelineParallelGroupRanks(rank.GlobalRank()));
            CHECK_NOTNULL(pp_pg);
            pp_rank = pp_pg->GetGroupRank(rank.GlobalRank());
            nn::parallel::pp_rank = pp_rank;
        }
        nn::parallel::tp_rank = 0;
    } else {
        device = Device(device_type, 0);
    }

    nn::TransformerConfig model_config = mixtral::ConfigForPreset(FLAGS_model_config);
    mixtral::SanitizeMixtralConfig(model_config);
    ValidateRuntimeFlags(model_config);

    std::shared_ptr<nn::Module> model;
    if (!FLAGS_llmc_filepath.empty()) {
        model = mixtral::LoadFromLLMC(FLAGS_llmc_filepath, model_config);
    } else {
        model = std::make_shared<nn::TransformerModel>(model_config);
    }
    model->To(device);

    const uint32_t grad_accum_steps
        = FLAGS_global_batch_size / (FLAGS_micro_batch_size * static_cast<uint32_t>(ddp_world_size));
    if (rank.IsMainRank()) {
        LOG(INFO) << "global batch size: " << FLAGS_global_batch_size << ", data parallel size: " << ddp_world_size
                  << ", gradient accumulation steps: " << grad_accum_steps;
    }

    const uint32_t loader_batch_size
        = pp_world_size > 1 ? FLAGS_micro_batch_size * grad_accum_steps : FLAGS_micro_batch_size;
    DistributedDataLoader train_loader(std::make_shared<TinyShakespeareDataset>(FLAGS_input_bin, FLAGS_sequence_length),
                                       loader_batch_size, ddp_rank, ddp_world_size);
    auto train_iter = train_loader.begin();

    auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();
    loss_fn->To(device);
    DataType dtype;
    if (FLAGS_dtype == kDtypeFP32) {
        dtype = DataType::kFLOAT32;
    } else if (FLAGS_dtype == kDtypeBF16) {
        dtype = DataType::kBFLOAT16;
    } else {
        LOG(FATAL) << "Rank " << rank.GlobalRank() << ": Datatype " << FLAGS_dtype << " not supported.";
    }

    if (pp_world_size > 1) {
        const std::vector<std::vector<int64_t>> recv_shapes{
            {FLAGS_micro_batch_size, FLAGS_sequence_length, model_config.n_embd}};
        model = std::make_shared<PipelineParallel>(model, pp_world_size, grad_accum_steps, recv_shapes, pp_rank, device,
                                                   model_config.GetChunkSize());
        if (ddp_world_size > 1) {
            auto ddp_config = DistributedDataParallelConfig{};
            ddp_config.overlap_grad_reduce = false;
            ddp_config.zero_stage = FLAGS_zero_stage;
            auto *mutable_chunks = dynamic_cast<PipelineParallel *>(model.get())->mutable_chunks();
            for (size_t chunk_id = 0; chunk_id < mutable_chunks->size(); ++chunk_id) {
                mutable_chunks->at(chunk_id)
                    = std::make_shared<DistributedDataParallel>(mutable_chunks->at(chunk_id), rank, ddp_config);
            }
        }
    } else if (ddp_world_size > 1) {
        auto ddp_config = DistributedDataParallelConfig{};
        ddp_config.overlap_grad_reduce = false;
        ddp_config.zero_stage = FLAGS_zero_stage;
        model = std::make_shared<DistributedDataParallel>(model, rank, ddp_config);
    }

    auto optimizer_creator = optimizers::Adam::CreateNamed(static_cast<float>(FLAGS_learning_rate));
    const auto params_to_optimize = model->NamedParameters();
    std::shared_ptr<Optimizer> optimizer;
    if (FLAGS_zero_stage >= 1) {
        auto model_chunks = pp_world_size > 1 ? *dynamic_cast<PipelineParallel *>(model.get())->mutable_chunks()
                                              : std::vector<std::shared_ptr<nn::Module>>{model};
        optimizer = std::make_shared<DistributedOptimizer>(optimizer_creator, params_to_optimize, model_chunks,
                                                           ddp_world_size, ddp_rank);
    } else {
        optimizer = optimizer_creator(params_to_optimize);
    }
    auto device_impl = core::GetDeviceGuardImpl(device.type());
    std::vector<double> step_duration_ms;
    step_duration_ms.reserve(FLAGS_num_iteration);
    const double tokens_per_step = static_cast<double>(FLAGS_global_batch_size) * FLAGS_sequence_length;

    LOG(INFO) << "Rank " << rank.GlobalRank() << ": start " << FLAGS_model_config << " training on "
              << device.ToString();
    for (uint32_t step = 0; step < FLAGS_num_iteration; ++step) {
#ifdef PROFILE_MODE
        Profiler::Instance().SetTag("Step_" + std::to_string(step));
#endif
        device_impl->ResetMemPoolHighWatermarks(device);
        device_impl->SynchronizeDevice(device);
        const auto step_start_time = std::chrono::steady_clock::now();

        float lossf = 0.0f;
        if (pp_world_size > 1) {
            if (train_iter == train_loader.end()) {
                train_iter = train_loader.begin();
            }
            auto [x_cpu, y_cpu] = *train_iter;
            ++train_iter;
            auto x = std::make_shared<Tensor>(x_cpu->To(device));
            auto y = std::make_shared<Tensor>(y_cpu->To(device));
            lossf = model->TrainStep({x}, {y}, optimizer, loss_fn, dtype);
        } else {
            optimizer->ZeroGrad();
            for (uint32_t micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                AutocastGuard autocast_guard(device.type(), dtype);
                if (train_iter == train_loader.end()) {
                    train_iter = train_loader.begin();
                }
                auto [x_cpu, y_cpu] = *train_iter;
                ++train_iter;
                auto x = std::make_shared<Tensor>(x_cpu->To(device));
                auto y = std::make_shared<Tensor>(y_cpu->To(device));
                auto logits = (*model)({x})[0];
                auto loss = (*loss_fn)({logits, y})[0] / static_cast<float>(grad_accum_steps);
                autocast_guard.Disable();
                loss->Backward();
                // Defer the loss D2H copy until after backward; reading it earlier would synchronize the device
                // between forward and backward.
                auto loss_cpu = loss->To(Device());
                lossf += static_cast<const float *>(loss_cpu.DataPtr())[0];
            }
            nn::parallel::FinalizeModelGrads({model});
            optimizer->Step();
        }

        device_impl->SynchronizeDevice(device);
        const auto step_end_time = std::chrono::steady_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(step_end_time - step_start_time).count();
        step_duration_ms.push_back(duration_ms);

        if (rank.IsLastRank() && FLAGS_log_interval > 0
            && ((step + 1) % FLAGS_log_interval == 0 || step + 1 == FLAGS_num_iteration)) {
            size_t used_mb = 0;
            size_t reserved_mb = 0;
            std::tie(used_mb, reserved_mb) = device_impl->GetMemPoolPeakMB(device);
            std::cout << std::format(
                "step {:4d}/{} | train loss {:.6f} | norm -1.0000 | lr {:.2e} | ({:.2f} ms | {:.0f} tok/s | "
                "peak used: {:5d} MB | peak reserved: {:5d} MB)",
                step + 1, FLAGS_num_iteration, lossf, FLAGS_learning_rate, duration_ms,
                tokens_per_step / (duration_ms / 1e3), used_mb, reserved_mb)
                      << std::endl;
        }
    }

    if (rank.IsLastRank() && FLAGS_print_timing && !step_duration_ms.empty()) {
        double duration_sum_ms = 0.0;
        for (size_t idx = step_duration_ms.size() > 1 ? 1 : 0; idx < step_duration_ms.size(); ++idx) {
            duration_sum_ms += step_duration_ms[idx];
        }
        const size_t averaged_steps
            = step_duration_ms.size() > 1 ? step_duration_ms.size() - 1 : step_duration_ms.size();
        std::cout << std::format("final {} iters avg: {:.3f}ms", averaged_steps, duration_sum_ms / averaged_steps)
                  << std::endl;
    }

#ifdef PROFILE_MODE
    if (rank.IsLastRank()) {
        Profiler::Instance().Report("mixtral.report", Profiler::SortBy::DeviceTimePercentage);
        Profiler::Instance().PrintRecords("mixtral.records.log");
    }
#endif
}

int main(int argc, char *argv[]) {
    // Register provider metadata and implementations before gflags validates
    // --device. The device runtime initializes lazily on first DeviceGuard use.
#ifdef INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_REGISTRAR
    INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_REGISTRAR();
#endif
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    CHECK_GT(FLAGS_nthread_per_process, 0);
    CHECK_EQ(FLAGS_tensor_parallel, 1) << "The Mixtral EP launcher currently requires --tensor_parallel=1";
    CHECK_EQ(FLAGS_expert_tensor_parallel, 1)
        << "The Mixtral EP launcher currently requires --expert_tensor_parallel=1";
    CHECK_GT(FLAGS_pipeline_parallel, 0);
    CHECK_GT(FLAGS_virtual_pipeline_parallel, 0);
    CHECK_GT(FLAGS_expert_parallel, 0);

    nn::parallel::global::InitAllEnv(FLAGS_nthread_per_process, FLAGS_tensor_parallel,
                                     /*sequence_parallel_enabled=*/false, FLAGS_pipeline_parallel,
                                     FLAGS_virtual_pipeline_parallel, FLAGS_expert_parallel,
                                     FLAGS_expert_tensor_parallel);

    LOG(INFO) << nn::parallel::global::ProcessGroupOverview(nn::parallel::global::GetDenseRankGenerator(),
                                                            nn::parallel::global::GetExpertRankGenerator());
    // Use one worker thread for each logical rank owned by this process.
    if (FLAGS_nthread_per_process > 1) {
        std::vector<std::thread> threads;
        for (int idx = 0; idx < FLAGS_nthread_per_process; ++idx) {
            nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), idx,
                                    nn::parallel::global::GetNprocPerNode(), FLAGS_nthread_per_process);
            threads.emplace_back(Train, rank);
        }

        for (auto &thread : threads) { thread.join(); }
    } else {
        nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), /*thread_rank=*/0,
                                nn::parallel::global::GetNprocPerNode(), /*thread_size=*/1);
        Train(rank);
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}

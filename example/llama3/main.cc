#include <cstdlib>
#include <format>
#include <memory>
#include <optional>
#include <unordered_set>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/optimizer.h"
#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/utils/global_module_hook_registry.h"
#include "infini_train/include/utils/precision_check_config.h"
#include "infini_train/include/utils/precision_checker.h"

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "example/llama3/net.h"

// I/O
DEFINE_string(input_bin, "", "input .bin to train on");
DEFINE_string(input_val_bin, "", "input .bin to eval validation loss on");
DEFINE_string(tokenizer_bin, "", "input .bin to tokenizer");
// model bin file is downloaded and processed using the script at
// https://github.com/karpathy/llm.c/blob/master/train_llama3.py
DEFINE_string(llmc_filepath, "", "llmc model file path to load from");
DEFINE_string(model, "llama3", "meta-llama/Meta-Llama-3.1-8B");
// token layout for each step of the optimization
DEFINE_uint32(batch_size, 4, "batch size, in units of #batch dimensions");
DEFINE_uint32(sequence_length, 64, "sequence length");
DEFINE_uint32(total_batch_size, 256, "total desired batch size, in units of #tokens");
// workload (number of steps)
DEFINE_uint32(num_iteration, 10, "number of iterations to run");
DEFINE_uint32(freq_generate_txt, 10, "frequency of text generation");
DEFINE_uint32(text_length, 64, "the length of the generated text");
// optimization
DEFINE_double(learning_rate, 1e-5, "learning rate warmup iterations");
DEFINE_bool(use_distributed_optimizer, false, "Whether to enable DistributedOptimizer(only take effects when DP>1)");
// evaluation
DEFINE_uint32(val_loss_every, 0, "every how many steps to evaluate val loss?");
DEFINE_uint32(sample_every, 0, "how often to sample from the model?");
// debugging
DEFINE_bool(overfit_single_batch, true, "overfit just one batch of data");
// memory management
DEFINE_string(device, "cuda", "device type (cpu/cuda), useless if using parallel training mode");
// parallel
DEFINE_int32(
    nthread_per_process, 1,
    "Number of threads to use for each process. "
    "When set > 1, enables data parallelism with device=cuda on the specified number of visible CUDA devices.");
DEFINE_uint32(tensor_parallel, 1, "Tensor Parallel world size");
DEFINE_bool(sequence_parallel, false, "Whether to enable Sequence Parallel");
DEFINE_uint32(pipeline_parallel, 1, "Pipeline Parallel world size, specified the number of PP stages.");
DEFINE_uint32(virtual_pipeline_parallel, 1, "Number of chunks in PP stage.");
// precision
DEFINE_string(dtype, "float32", "precision used in training (float32/bfloat16)");
// precision check
DEFINE_string(
    precision_check, "",
    "precision check config: level=N,format=simple|table,output_md5=true|false,output_path=PATH,baseline=PATH");

using namespace infini_train;

namespace {
// validation
const std::unordered_set<std::string> kSupportedModels = {"llama3"};
constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";
constexpr char kDtypeFP32[] = "float32";
constexpr char kDtypeBF16[] = "bfloat16";
} // namespace

DEFINE_validator(model, [](const char *, const std::string &value) { return kSupportedModels.contains(value); });
DEFINE_validator(device,
                 [](const char *, const std::string &value) { return value == kDeviceCPU || value == kDeviceCUDA; });

void Train(const nn::parallel::Rank &rank) {
    using namespace nn::parallel;

    // select the device
    const Device *device;

    int ddp_world_size = global::GetDataParallelSize();
    int tp_world_size = global::GetTensorParallelSize();
    int sp_world_size = global::GetSequenceParallelEnabled() ? tp_world_size : 1;
    int pp_world_size = global::GetPipelineParallelSize();

    if (FLAGS_sequence_parallel) {
        CHECK_EQ(FLAGS_sequence_length % tp_world_size, 0)
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
        device = DeviceManager::Instance()->GetDevice(DeviceType::kCUDA, rank.thread_rank());

        if (ddp_world_size > 1) {
            ddp_pg = ProcessGroupFactory::Instance()->GetOrCreate(GetDataParallelProcessGroupName(rank.GlobalRank()),
                                                                  GetDataParallelGroupRanks(rank.GlobalRank()));
            ddp_rank = ddp_pg->GetGroupRank(rank.GlobalRank());
        }

        if (tp_world_size > 1) {
            tp_pg = ProcessGroupFactory::Instance()->GetOrCreate(GetTensorParallelProcessGroupName(rank.GlobalRank()),
                                                                 GetTensorParallelGroupRanks(rank.GlobalRank()));
            tp_rank = tp_pg->GetGroupRank(rank.GlobalRank());
            // NOTE(zbl): Reserved for VocabParallelEmbedding
            nn::parallel::tp_rank = tp_rank;
        }

        if (pp_world_size > 1) {
            pp_pg = ProcessGroupFactory::Instance()->GetOrCreate(GetPipelineParallelProcessGroupName(rank.GlobalRank()),
                                                                 GetPipelineParallelGroupRanks(rank.GlobalRank()));
            pp_rank = pp_pg->GetGroupRank(rank.GlobalRank());

            nn::parallel::pp_rank = pp_rank;
        }
    } else {
        device = FLAGS_device == kDeviceCPU ? DeviceManager::Instance()->GetDefaultDevice()
                                            : DeviceManager::Instance()->GetDevice(DeviceType::kCUDA, 0);
    }

    // calculate gradient accumulation from the desired total batch size and the current run configuration
    const auto tokens_per_fwdbwd = FLAGS_batch_size * FLAGS_sequence_length * ddp_world_size;
    CHECK_EQ(FLAGS_total_batch_size % tokens_per_fwdbwd, 0);
    const auto grad_accum_steps = FLAGS_total_batch_size / tokens_per_fwdbwd;
    if (rank.IsMainRank()) {
        LOG(INFO) << "total desired batch size: " << FLAGS_total_batch_size
                  << " => calculated gradient accumulation steps: " << grad_accum_steps;
    }

    // rng / reproducibility
    // ManualSeed(42);

    LLaMA3Config model_config = LLaMA3Config();
    std::shared_ptr<nn::Module> model = nullptr;
    if (!FLAGS_llmc_filepath.empty()) {
        model = LLaMA3::FromLLMC(FLAGS_llmc_filepath);
    } else {
        model = std::make_shared<LLaMA3>(model_config);
    }

    model->To(device);

    LOG(INFO) << "Rank " << rank.GlobalRank() << ": Model loaded to device.";

    DataType dtype;
    if (FLAGS_dtype == kDtypeFP32) {
        dtype = DataType::kFLOAT32;
    } else if (FLAGS_dtype == kDtypeBF16) {
        dtype = DataType::kBFLOAT16;
    } else {
        LOG(FATAL) << "Rank " << rank.GlobalRank() << ": Datatype " << FLAGS_dtype << " not supported.";
    }

    auto num_micro_batches = FLAGS_total_batch_size / (FLAGS_batch_size * FLAGS_sequence_length * ddp_world_size);

    if (pp_world_size > 1) {
        // NOTE(dcj): To ensure that the tensor shapes at the pipeline stage boundaries remain correct
        // when sequence parallelism (SP) is enabled, we need to divide by sp_world_size.
        auto shapes = std::vector<std::vector<int64_t>>{
            {FLAGS_batch_size, FLAGS_sequence_length / sp_world_size, model_config.n_embd}};

        model = std::make_shared<nn::parallel::PipelineParallel>(
            model, pp_world_size, num_micro_batches, shapes, pp_rank, rank.thread_rank(),
            std::dynamic_pointer_cast<LLaMA3>(model)->GetChunkSize());
        if (ddp_world_size > 1) {
            auto ddp_config
                = DistributedDataParallelConfig{.use_distributed_optimizer = FLAGS_use_distributed_optimizer};
            auto *mutable_chunks = dynamic_cast<nn::parallel::PipelineParallel *>(model.get())->mutable_chunks();
            for (int chunk_id = 0; chunk_id < mutable_chunks->size(); ++chunk_id) {
                (*mutable_chunks)[chunk_id] = std::make_shared<DistributedDataParallel>(mutable_chunks->at(chunk_id),
                                                                                        rank.thread_rank(), ddp_config);
            }
        }
    } else if (ddp_world_size > 1) {
        // NOTE(dcj): Complete all device (.to(device)) and dtype (.to(dtype)) conversions
        // before wrapping the model with DistributedDataParallel (DDP).
        // Otherwise, DDP’s gradient hooks may be lost because new parameter tensors
        // are created during the conversion.

        auto ddp_config = DistributedDataParallelConfig{.use_distributed_optimizer = FLAGS_use_distributed_optimizer};
        model = std::make_shared<DistributedDataParallel>(model, rank.thread_rank(), ddp_config);
    }

    DistributedDataLoader train_loader(std::make_shared<TinyShakespeareDataset>(FLAGS_input_bin, FLAGS_sequence_length),
                                       pp_world_size > 1 ? FLAGS_batch_size * num_micro_batches : FLAGS_batch_size,
                                       ddp_rank, ddp_world_size);

    std::optional<DistributedDataLoader> val_loader = std::nullopt;
    if (!FLAGS_input_val_bin.empty()) {
        val_loader = DistributedDataLoader(
            std::make_shared<TinyShakespeareDataset>(FLAGS_input_val_bin, FLAGS_sequence_length), FLAGS_batch_size,
            ddp_rank, ddp_world_size);
    }

    //
    // main training loop
    //
    std::unique_ptr<Tokenizer> tokenizer = nullptr;
    if (!FLAGS_tokenizer_bin.empty()) {
        tokenizer = std::make_unique<Tokenizer>(FLAGS_tokenizer_bin);
    }

    // TODO(dcj): support more complex optimizer later
    // auto optimizer = optimizers::Adam(model->Parameters(), FLAGS_learning_rate);
    auto optimizer_creator = optimizers::Adam::Create(FLAGS_learning_rate);
    std::shared_ptr<Optimizer> optimizer = nullptr;

    if (FLAGS_use_distributed_optimizer) {
        auto model_chunks = (pp_world_size > 1)
                              ? *(dynamic_cast<nn::parallel::PipelineParallel *>(model.get())->mutable_chunks())
                              : std::vector<std::shared_ptr<nn::Module>>{model};
        optimizer = std::make_shared<nn::parallel::DistributedOptimizer>(optimizer_creator, model->Parameters(),
                                                                         model_chunks, ddp_world_size, ddp_rank);
    } else {
        optimizer = optimizer_creator(model->Parameters());
    }

    auto train_iter = train_loader.begin();
    std::shared_ptr<nn::Module> loss_fn
        = (tp_world_size > 1) ? std::static_pointer_cast<nn::Module>(std::make_shared<VocabParallelCrossEntropyLoss>())
                              : std::static_pointer_cast<nn::Module>(std::make_shared<nn::CrossEntropyLoss>());
    loss_fn->To(device);
    LOG(INFO) << "Rank " << rank.GlobalRank() << ": start training";

    auto cuda_device = device->IsCUDA() ? dynamic_cast<const CudaDevice *>(device) : nullptr;

    for (int step = 0; step < FLAGS_num_iteration + 1; ++step) {
        // Reset precision check counters at start of each iteration for file overwrite
        utils::PrecisionChecker::ResetCounters();

        const bool last_step = step == FLAGS_num_iteration;

        if (cuda_device) {
            cuda_device->ResetMemPoolHighWatermarks();
        }

        const auto iter_start = std::chrono::high_resolution_clock::now();

        // once in a while evaluate the validation dataset
        if (FLAGS_val_loss_every > 0 && (step % FLAGS_val_loss_every == 0 || last_step) && val_loader.has_value()) {
            // TODO(dcj): implement this after model.eval() is supported
        }
        // once in a while perform model inference on the master process
        if (FLAGS_sample_every > 0 && (step % FLAGS_sample_every == 0 || last_step)) {
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

        float lossf = 0.0f;
        if (pp_world_size == 1) {
            // model->Train();
            optimizer->ZeroGrad();

            // if we are trying to overfit a single batch, we reset the loader here
            if (FLAGS_overfit_single_batch) {
                // train_loader.Reset();
            }

            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                // enable autocast for the current step
                infini_train::AutocastGuard autocast_guard(device->Type(), dtype);

                // (bs, seq_len), (bs, seq_len)
                auto [x, y] = *train_iter;
                // if we are trying to overfit a single batch, we reset the loader here by commenting out the line below
                // TODO(dcj): support dataloader.reset() later
                ++train_iter;
                x = std::make_shared<Tensor>(x->To(device));
                y = std::make_shared<Tensor>(y->To(device));

                LOG(INFO) << "Rank " << rank.GlobalRank() << ": start forward";
                // (bs, seq_len, vocab_size)
                auto logits = (*model)({x, y})[0];
                LOG(INFO) << "Rank " << rank.GlobalRank() << ": finish model forward, start loss forward";
                auto loss = (*loss_fn)({logits, y})[0];
                // FIXME(jym): verify gradient accumulation precision
                loss = loss / grad_accum_steps;

                // disable autocast for the current step (backward is not under autocast)
                autocast_guard.Disable();

                LOG(INFO) << "Rank " << rank.GlobalRank() << ": finish loss forward";

                auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
                lossf += static_cast<const float *>(loss_cpu.DataPtr())[0];
                LOG(INFO) << "Rank " << rank.GlobalRank() << ": start backward";
                loss->Backward();
                LOG(INFO) << "Rank " << rank.GlobalRank() << ": finish backward";
            }

            optimizer->Step();
        } else {
            auto [x, y] = *train_iter;
            // if we are trying to overfit a single batch, we reset the loader here by commenting out the line below
            // TODO(dcj): support dataloader.reset() later
            ++train_iter;
            x = std::make_shared<Tensor>(x->To(device));
            y = std::make_shared<Tensor>(y->To(device));

            lossf = model->TrainStep({x}, {y}, optimizer, loss_fn, dtype);
        }

        if (ddp_world_size > 1) {
            auto lossf_tensor = std::make_shared<Tensor>(&lossf, std::vector<int64_t>{}, DataType::kFLOAT32, device);
            function::AllReduce(lossf_tensor, function::ReduceOpType::kAvg, ddp_pg);
            lossf = static_cast<const float *>(
                lossf_tensor->To(DeviceManager::Instance()->GetDefaultDevice()).DataPtr())[0];
        }

        const auto iter_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(iter_end - iter_start).count();
        const double tps = FLAGS_total_batch_size / (duration_us / 1e6);

        if (rank.IsLastRank()) {
            size_t used_mb = 0, reserved_mb = 0;
            if (cuda_device) {
                std::tie(used_mb, reserved_mb) = cuda_device->GetMemPoolPeakMB();
            }

            LOG(ERROR) << std::format("step {:4d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} tok/s | "
                                      "peak used: {:5d} MB | peak reserved: {:5d} MB, DP={}, TP={}, SP={}, PP={})",
                                      step + 1, FLAGS_num_iteration, lossf, FLAGS_learning_rate, duration_us / 1e3f,
                                      tps, used_mb, reserved_mb, ddp_world_size, tp_world_size, sp_world_size,
                                      pp_world_size);

            if ((step + 1) % FLAGS_freq_generate_txt == 0) {
                // FIXME(jym): to support PP
                if (tokenizer) {
                    CHECK_EQ(pp_world_size, 1);
                    tokenizer->GenerateText(*model, FLAGS_batch_size, FLAGS_sequence_length, FLAGS_text_length, device);
                }
            }
        }
    }
#ifdef PROFILE_MODE
    Profiler::Instance().Report("llama3.report", Profiler::SortBy::DeviceTimePercentage);
    Profiler::Instance().PrintRecords("llama3.records.log");
#endif
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    auto precision_config = utils::PrecisionCheckConfig::Parse(FLAGS_precision_check);
    nn::parallel::global::InitAllEnv(FLAGS_nthread_per_process, FLAGS_tensor_parallel, FLAGS_sequence_parallel,
                                     FLAGS_pipeline_parallel, FLAGS_virtual_pipeline_parallel);
    utils::PrecisionCheckEnv::Instance().Init(precision_config);

    LOG(INFO) << nn::parallel::global::ProcessGroupOverview();

    // NOTE(dcj): currently we only support single process
    if (FLAGS_nthread_per_process > 1) {
        std::vector<std::thread> threads;
        for (int idx = 0; idx < FLAGS_nthread_per_process; ++idx) {
            nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), idx,
                                    nn::parallel::global::GetNprocPerNode(), FLAGS_nthread_per_process);
            threads.emplace_back(Train, rank);
        }

        for (auto &thread : threads) { thread.join(); }
    } else {
        Train({0, 0, 1, 1});
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}

#include <chrono>
#include <cstdlib>
#include <format>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel_functional.h"
#include "infini_train/include/optimizer.h"
#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "example/common/utils.h"
#include "example/gpt2/net.h"

// I/O
DEFINE_string(input_bin, "", "input .bin to train on");
DEFINE_string(input_val_bin, "", "input .bin to eval validation loss on");
DEFINE_string(tokenizer_bin, "", "input .bin to tokenizer");
// model bin file is downloaded and processed using the script at
// https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
DEFINE_string(llmc_filepath, "", "llmc model file path to load from");
DEFINE_string(model, "gpt2", "gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48");
// token layout for each step of the optimization
DEFINE_uint32(batch_size, 4, "batch size, in units of #batch dimensions");
DEFINE_uint32(sequence_length, 64, "sequence length");
DEFINE_uint32(total_batch_size, 256, "total desired batch size, in units of #tokens");
// workload (number of steps)
DEFINE_uint32(num_iteration, 10, "number of iterations to run");
DEFINE_uint32(freq_generate_txt, 10, "frequency of text generation");
DEFINE_uint32(text_length, 64, "the length of the generated text");
// optimization
DEFINE_double(learning_rate, 1e-4, "learning rate warmup iterations");
// evaluation
DEFINE_uint32(val_loss_every, 0, "every how many steps to evaluate val loss?");
DEFINE_uint32(sample_every, 0, "how often to sample from the model?");
// debugging
DEFINE_bool(overfit_single_batch, true, "overfit just one batch of data");
// flash_attention
DEFINE_bool(flash, false, "Enable FlashAttention (true/false), default=false");
// memory management
DEFINE_string(device, "cuda", "device type (cpu/cuda), useless if data_parallel=true");
// parallel
DEFINE_int32(
    data_parallel, 1,
    "Number of GPUs to use for data parallel training. "
    "When set > 1, enables data parallelism with device=cuda on the specified number of visible CUDA devices.");
// precision
DEFINE_string(dtype, "float32", "precision used in training (float32/bfloat16)");

using namespace infini_train;

namespace {
// validation
const std::unordered_set<std::string> kSupportedModels
    = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"};
constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";
constexpr char kDtypeFP32[] = "float32";
constexpr char kDtypeBF16[] = "bfloat16";

//
const std::unordered_map<std::string, GPT2Config> kModelToConfigs = {
    {"d12", {.block_size = 1024, .vocab_size = 50257, .n_layer = 12, .n_head = 12, .n_embd = 768}},
    {"d24", {.block_size = 1024, .vocab_size = 50257, .n_layer = 24, .n_head = 16, .n_embd = 1024}},
    {"d36", {.block_size = 1024, .vocab_size = 50257, .n_layer = 36, .n_head = 20, .n_embd = 1280}},
    {"d48", {.block_size = 1024, .vocab_size = 50257, .n_layer = 48, .n_head = 25, .n_embd = 1600}},
};
const std::unordered_map<std::string, GPT2::ModelType> kStrToModelType = {
    {"gpt2", GPT2::ModelType::kGPT2},
    {"gpt2-medium", GPT2::ModelType::kGPT2Medium},
    {"gpt2-large", GPT2::ModelType::kGPT2Large},
    {"gpt2-xl", GPT2::ModelType::kGPT2XL},
};
} // namespace

DEFINE_validator(model, [](const char *, const std::string &value) { return kSupportedModels.contains(value); });
DEFINE_validator(device,
                 [](const char *, const std::string &value) { return value == kDeviceCPU || value == kDeviceCUDA; });

void Train(const nn::parallel::DistributedDataParallel::Rank &rank) {
    // select the device
    const Device *device;
    if (rank.IsDDP()) {
        device = DeviceManager::Instance()->GetDevice(DeviceType::kCUDA, rank.thread_rank());
    } else {
        device = FLAGS_device == kDeviceCPU ? DeviceManager::Instance()->GetDefaultDevice()
                                            : DeviceManager::Instance()->GetDevice(DeviceType::kCUDA, 0);
    }

    // calculate gradient accumulation from the desired total batch size and the current run configuration
    const auto tokens_per_fwdbwd = FLAGS_batch_size * FLAGS_sequence_length * rank.WorldSize();
    CHECK_EQ(FLAGS_total_batch_size % tokens_per_fwdbwd, 0);
    const auto grad_accum_steps = FLAGS_total_batch_size / tokens_per_fwdbwd;
    LOG(INFO) << "total desired batch size: " << FLAGS_total_batch_size
              << " => calculated gradient accumulation steps: " << grad_accum_steps;

    // rng / reproducibility
    // ManualSeed(42);

    // init the model, either from scratch or from OpenAI pretrained checkpoint
    GPT2Config model_config;
    std::shared_ptr<nn::Module> model = nullptr;
    if (!FLAGS_llmc_filepath.empty()) {
        model = GPT2::FromLLMC(FLAGS_llmc_filepath);
    } else if (kModelToConfigs.count(FLAGS_model)) {
        model_config = kModelToConfigs.at(FLAGS_model);
        model = std::make_shared<GPT2>(model_config);
    } else {
        model = GPT2::FromPretrained(kStrToModelType.at(FLAGS_model));
    }

    model->To(device);

    // select the data type
    // TODO(lzm): change to solely rely on the weight file info for determining the dtype when autocast is supported
    DataType dtype;
    if (FLAGS_dtype == kDtypeFP32) {
        dtype = DataType::kFLOAT32;
    } else if (FLAGS_dtype == kDtypeBF16) {
        dtype = DataType::kBFLOAT16;
        model->To(dtype);
    } else {
        LOG(FATAL) << "Rank " << rank.thread_rank() << ": Datatype " << FLAGS_dtype << " not supported.";
    }

    DistributedDataLoader train_loader(std::make_shared<TinyShakespeareDataset>(FLAGS_input_bin, FLAGS_sequence_length),
                                       FLAGS_batch_size, rank.thread_rank(), rank.WorldSize());
    std::optional<DistributedDataLoader> val_loader = std::nullopt;
    if (!FLAGS_input_val_bin.empty()) {
        val_loader = DistributedDataLoader(
            std::make_shared<TinyShakespeareDataset>(FLAGS_input_val_bin, FLAGS_sequence_length), FLAGS_batch_size,
            rank.thread_rank(), rank.WorldSize());
    }

    //
    // main training loop
    //

    std::unique_ptr<Tokenizer> tokenizer = nullptr;
    if (!FLAGS_tokenizer_bin.empty()) {
        tokenizer = std::make_unique<Tokenizer>(FLAGS_tokenizer_bin);
    }

    // TODO(dcj): support more complex optimizer later
    auto optimizer = optimizers::SGD(model->Parameters(), FLAGS_learning_rate);

    auto train_iter = train_loader.begin();
    auto loss_fn = nn::CrossEntropyLoss();
    loss_fn.To(device);
    LOG(INFO) << "Rank " << rank.thread_rank() << ": start training";

    for (int step = 0; step < FLAGS_num_iteration + 1; ++step) {
        const bool last_step = step == FLAGS_num_iteration;

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

        // model->Train();
        optimizer.ZeroGrad();
        // if we are trying to overfit a single batch, we reset the loader here
        if (FLAGS_overfit_single_batch) {
            // train_loader.Reset();
        }
        float lossf = 0.0f;
#ifdef PROFILE_MODE
        Profiler::Instance().SetTag("Step_" + std::to_string(step));
#endif
        for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            // (bs, seq_len), (bs, seq_len)
            auto [x, y] = *train_iter;
            // if we are trying to overfit a single batch, we reset the loader here by commenting out the line below
            // TODO(dcj): support dataloader.reset() later
            ++train_iter;
            x = std::make_shared<Tensor>(x->To(device));
            y = std::make_shared<Tensor>(y->To(device));
            LOG(INFO) << "Rank " << rank.thread_rank() << ": start forward";
            // (bs, seq_len, vocab_size)
            auto logits = model->Forward({x, y})[0];
            LOG(INFO) << "Rank " << rank.thread_rank() << ": finish model forward, start loss forward";
            auto loss = loss_fn.Forward({logits, y})[0];
            loss = loss / grad_accum_steps;
            LOG(INFO) << "Rank " << rank.thread_rank() << ": finish loss forward";
            if (rank.IsDDP()) {
                nn::parallel::function::AllReduce(loss, nn::parallel::function::ReduceOpType::kAvg);
            }
            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
            if (FLAGS_dtype == kDtypeFP32) {
                lossf += static_cast<const float *>(loss_cpu.DataPtr())[0];
            } else if (FLAGS_dtype == kDtypeBF16) {
                lossf += ConvertBF16ToFloat(loss_cpu.DataPtr());
            }
            LOG(INFO) << "Rank " << rank.thread_rank() << ": start backward";
            loss->Backward();
            if (rank.IsDDP()) {
                for (auto param : model->Parameters()) {
                    nn::parallel::function::AllReduce(param->grad(), nn::parallel::function::ReduceOpType::kAvg);
                }
            }
            LOG(INFO) << "Rank " << rank.thread_rank() << ": finish backward";
        }
        optimizer.Step();

        const auto iter_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(iter_end - iter_start).count();
        const double tps = FLAGS_total_batch_size / (duration_us / 1e6);

        if (rank.IsMainRank()) {
            LOG(ERROR) << std::format("step {:4d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} tok/s)",
                                      step + 1, FLAGS_num_iteration, lossf, FLAGS_learning_rate, duration_us / 1e3f,
                                      tps);

            if ((step + 1) % FLAGS_freq_generate_txt == 0) {
                if (!tokenizer) {
                    continue;
                }
                tokenizer->GenerateText(*model, FLAGS_batch_size, FLAGS_sequence_length, FLAGS_text_length, device);
            }
        }
    }
#ifdef PROFILE_MODE
    Profiler::Instance().Report("gpt2.report", Profiler::SortBy::DeviceTimePercentage);
    Profiler::Instance().PrintRecords("gpt2.records.log");
#endif
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // NOTE(dcj): currently we only support single process
    if (FLAGS_data_parallel > 1) {
        std::vector<std::thread> threads;
        for (int idx = 0; idx < FLAGS_data_parallel; ++idx) {
            nn::parallel::DistributedDataParallel::Rank rank(0, idx, 1, FLAGS_data_parallel);
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

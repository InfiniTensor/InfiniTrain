#include <cstdlib>
#include <format>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/data_parallel.h"
#include "infini_train/include/optimizer.h"
#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif

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
// evaluation
DEFINE_uint32(val_loss_every, 0, "every how many steps to evaluate val loss?");
DEFINE_uint32(sample_every, 0, "how often to sample from the model?");
// debugging
DEFINE_bool(overfit_single_batch, true, "overfit just one batch of data");
// memory management
DEFINE_string(device, "cuda", "device type (cpu/cuda), useless if data_parallel=true");
// parallel
DEFINE_bool(
    data_parallel, false,
    "use data parallelism or not, will always use device=cuda and use all cuda visible devices when set to true");
// precision
DEFINE_string(dtype, "float32", "precision used in training (float32/bfloat16)");

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

float DecodeBF16(void *ptr) {
    uint16_t *raw_data = reinterpret_cast<uint16_t *>(ptr);
    uint32_t f32_bits = static_cast<uint32_t>(raw_data[0]) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // select the device
    const auto *device = DeviceManager::Instance()->GetDevice(
        FLAGS_data_parallel || FLAGS_device == kDeviceCUDA ? DeviceType::kCUDA : DeviceType::kCPU);
    // const Device *cpu_device = DeviceManager::Instance()->GetDefaultDevice();

    // calculate gradient accumulation from the desired total batch size and the current run configuration
    const auto tokens_per_fwdbwd = FLAGS_batch_size * FLAGS_sequence_length;
    CHECK_EQ(FLAGS_total_batch_size % tokens_per_fwdbwd, 0);
    const auto grad_accum_steps = FLAGS_total_batch_size / tokens_per_fwdbwd;
    LOG(INFO) << "total desired batch size: " << FLAGS_total_batch_size
              << " => calculated gradient accumulation steps: " << grad_accum_steps;

    // rng / reproducibility
    // ManualSeed(42);

    // init the model, either from scratch or from OpenAI pretrained checkpoint
    LLaMA3Config model_config = LLaMA3Config();
    std::shared_ptr<nn::Module> model = nullptr;
    if (!FLAGS_llmc_filepath.empty()) {
        model = LLaMA3::FromLLMC(FLAGS_llmc_filepath);
    } else {
        model = std::make_shared<LLaMA3>(model_config);
    }

    if (FLAGS_data_parallel) {
        model = std::make_shared<nn::parallel::DataParallel>(model);
    } else {
        model->To(device);
    }
    LOG(INFO) << "Model loaded to device.";

    DataType dtype;
    if (FLAGS_dtype == kDtypeFP32) {
        dtype = DataType::kFLOAT32;
    } else if (FLAGS_dtype == kDtypeBF16) {
        // TODO(zbl): Use autocast instead of manually casting the whole model params to bf16
        dtype = DataType::kBFLOAT16;
        model->To(dtype);
    } else {
        LOG(FATAL) << "Datatype " << FLAGS_dtype << " not supported.";
    }

    DataLoader train_loader(std::make_shared<TinyShakespeareDataset>(FLAGS_input_bin, FLAGS_sequence_length),
                            FLAGS_batch_size);
    std::optional<DataLoader> val_loader = std::nullopt;
    if (!FLAGS_input_val_bin.empty()) {
        val_loader = DataLoader(std::make_shared<TinyShakespeareDataset>(FLAGS_input_val_bin, FLAGS_sequence_length),
                                FLAGS_batch_size);
    }

    //
    // main training loop
    //
    std::unique_ptr<Tokenizer> tokenizer = nullptr;
    if (!FLAGS_tokenizer_bin.empty()) {
        tokenizer = std::make_unique<Tokenizer>(FLAGS_tokenizer_bin);
    }

    // TODO(dcj): support more complex optimizer later
    auto optimizer = optimizers::Adam(model->Parameters(), FLAGS_learning_rate);

    auto train_iter = train_loader.begin();
    auto loss_fn = nn::CrossEntropyLoss();
    loss_fn.To(device);
    LOG(INFO) << "start training";

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
            LOG(INFO) << "start forward";
            // (bs, seq_len, vocab_size)
            auto logits = model->Forward({x, y})[0];
            LOG(INFO) << "finish model forward, start loss forward";
            auto loss = loss_fn.Forward({logits, y})[0];
            loss = loss / grad_accum_steps;
            LOG(INFO) << "finish loss forward";
            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
            if (FLAGS_dtype == kDtypeFP32) {
                lossf += static_cast<const float *>(loss_cpu.DataPtr())[0];
            } else if (FLAGS_dtype == kDtypeBF16) {
                lossf += DecodeBF16(loss_cpu.DataPtr());
            }
            LOG(INFO) << "start backward";
            loss->Backward();
            LOG(INFO) << "finish backward";
        }
        optimizer.Step();

        const auto iter_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(iter_end - iter_start).count();
        const double tps = FLAGS_total_batch_size / (duration_us / 1e6);

        LOG(ERROR) << std::format("step {:4d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} tok/s)",
                                  step + 1, FLAGS_num_iteration, lossf, FLAGS_learning_rate, duration_us / 1e3f, tps);

        if ((step + 1) % FLAGS_freq_generate_txt == 0) {
            if (!tokenizer) {
                continue;
            }
            tokenizer->GenerateText(*model, FLAGS_batch_size, FLAGS_sequence_length, FLAGS_text_length, device);
        }
    }
#ifdef PROFILE_MODE
    Profiler::Instance().Report("llama3.report", Profiler::SortBy::DeviceTimePercentage);
    Profiler::Instance().PrintRecords("llama3.records.log");
#endif

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}

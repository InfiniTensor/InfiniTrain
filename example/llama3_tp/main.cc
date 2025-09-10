#include <cstdlib>
#include <format>
#include <memory>
#include <optional>
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
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel_functional.h"
#include "infini_train/include/optimizer.h"
#ifdef PROFILE_MODE
#include "infini_train/include/profiler.h"
#endif

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "example/common/utils.h"
#include "example/llama3_tp/net.h"

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
DEFINE_uint32(tensor_parallel, 0, "Tensor Parallel world size (0=use all visible CUDA devices)");
// precision
DEFINE_string(dtype, "float32", "precision used in training (float32/bfloat16)");

using namespace infini_train;
namespace nn = infini_train::nn;
namespace tp = infini_train::nn::parallel;

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

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // calculate gradient accumulation from the desired total batch size and the current run configuration
    const auto tokens_per_fwdbwd = FLAGS_batch_size * FLAGS_sequence_length;
    CHECK_EQ(FLAGS_total_batch_size % tokens_per_fwdbwd, 0);
    const auto grad_accum_steps = FLAGS_total_batch_size / tokens_per_fwdbwd;
    LOG(INFO) << "total desired batch size: " << FLAGS_total_batch_size
              << " => calculated gradient accumulation steps: " << grad_accum_steps;

    std::vector<const Device *> cuda_devs = DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA);
    CHECK(!cuda_devs.empty()) << "No CUDA devices found.";

    // FIXME(zbl): DeviceManager will create comms for all visible devices
    const uint32_t world_size = (FLAGS_tensor_parallel == 0)
                                  ? static_cast<uint32_t>(cuda_devs.size())
                                  : std::min<uint32_t>(FLAGS_tensor_parallel, static_cast<uint32_t>(cuda_devs.size()));
    CHECK_GT(world_size, 0u) << "Invalid tp_world_size";
    CHECK_LE(world_size, cuda_devs.size()) << "tp_world_size should be less than device count";
    LOG(INFO) << "TP world size = " << world_size;

    auto build_model = [&](tp::TensorParallelGroup tp_group) -> std::shared_ptr<TensorParallelLLaMA3> {
        std::shared_ptr<TensorParallelLLaMA3> model = nullptr;
        if (!FLAGS_llmc_filepath.empty()) {
            model = TensorParallelLLaMA3::FromLLMC(FLAGS_llmc_filepath, tp_group);
        } else {
            TPLLaMA3Config model_config = TPLLaMA3Config();
            model_config.tp_group = tp_group;
            model = std::make_shared<TensorParallelLLaMA3>(model_config);
        }
        return model;
    };

    // Each rank has its own local model/optimizer/loss_fn
    std::vector<std::shared_ptr<TensorParallelLLaMA3>> models(world_size);
    std::vector<std::unique_ptr<Optimizer>> opts(world_size);
    std::vector<std::shared_ptr<nn::parallel::VocabParallelCrossEntropyLoss>> loss_modules(world_size);
    std::vector<tp::TensorParallelGroup> tp_groups(world_size);

    DataType dtype = (FLAGS_dtype == std::string("bfloat16")) ? DataType::kBFLOAT16 : DataType::kFLOAT32;

    for (uint32_t r = 0; r < world_size; ++r) {
        tp_groups[r] = tp::TensorParallelGroup();
        tp_groups[r].devices.assign(cuda_devs.begin(), cuda_devs.begin() + world_size);
        tp_groups[r].rank = r;

        models[r] = build_model(tp_groups[r]);
        models[r]->To(tp_groups[r].devices[r]);
        if (dtype == DataType::kBFLOAT16) {
            models[r]->To(dtype);
        }

        opts[r] = std::make_unique<optimizers::Adam>(models[r]->Parameters(), FLAGS_learning_rate);

        // TODO(zbl): get original vocab_size dynamically
        loss_modules[r] = std::make_shared<nn::parallel::VocabParallelCrossEntropyLoss>(tp_groups[r]);
        loss_modules[r]->To(tp_groups[r].devices[r]);
    }

    DataLoader train_loader(std::make_shared<TinyShakespeareDataset>(FLAGS_input_bin, FLAGS_sequence_length),
                            FLAGS_batch_size);
    std::optional<DataLoader> val_loader = std::nullopt;
    if (!FLAGS_input_val_bin.empty()) {
        val_loader = DataLoader(std::make_shared<TinyShakespeareDataset>(FLAGS_input_val_bin, FLAGS_sequence_length),
                                FLAGS_batch_size);
    }
    auto train_iter = train_loader.begin();

    // TODO(zbl): check tokenizer
    std::unique_ptr<Tokenizer> tokenizer = nullptr;
    if (!FLAGS_tokenizer_bin.empty()) {
        tokenizer = std::make_unique<Tokenizer>(FLAGS_tokenizer_bin);
    }

    LOG(INFO) << "Start training LLaMA3 with Tensor Parallel";
    auto *cpu = DeviceManager::Instance()->GetDefaultDevice();

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
        // Perform ZeroGrad for each rank
        for (uint32_t r = 0; r < world_size; ++r) { opts[r]->ZeroGrad(); }
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
            ++train_iter;

            // copy input data to each rank
            std::vector<std::shared_ptr<Tensor>> x_rank(world_size), y_rank(world_size);
            for (uint32_t r = 0; r < world_size; ++r) {
                x_rank[r] = std::make_shared<Tensor>(x->To(tp_groups[r].devices[r]));
                y_rank[r] = std::make_shared<Tensor>(y->To(tp_groups[r].devices[r]));
            }

            // Perform forward on each rank
            std::vector<std::shared_ptr<Tensor>> local_logits(world_size);
            std::vector<float> local_losses(world_size, 0.0f);
            {
                std::vector<std::thread> threads;
                threads.reserve(world_size);
                for (uint32_t r = 0; r < world_size; ++r) {
                    threads.emplace_back([&, r]() {
                        dynamic_cast<const CudaDevice *>(tp_groups[r].devices[r])->SetDevice();
                        local_logits[r] = models[r]->Forward({x_rank[r]})[0];

                        auto local_loss = loss_modules[r]->Forward({local_logits[r], y_rank[r]})[0];
                        local_loss = local_loss / static_cast<float>(grad_accum_steps);
                        auto loss_cpu = local_loss->To(cpu);

                        if (dtype == DataType::kFLOAT32) {
                            local_losses[r] = static_cast<const float *>(loss_cpu.DataPtr())[0];
                        } else {
                            local_losses[r] = ConvertBF16ToFloat(loss_cpu.DataPtr());
                        }

                        local_loss->Backward();
                    });
                }
                for (auto &th : threads) { th.join(); }
            }

            // NOTE(zbl): losses on each rank should be the same,
            //            since AllReduce has been called in VocabParallelCrossEntropyLoss
            lossf += local_losses[0];
        }

        // update gradient for each rank
        {
            std::vector<std::thread> threads;
            threads.reserve(world_size);
            for (uint32_t r = 0; r < world_size; ++r) {
                threads.emplace_back([&, r]() { opts[r]->Step(); });
            }
            for (auto &th : threads) { th.join(); }
        }

        const auto iter_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(iter_end - iter_start).count();
        const double tps = FLAGS_total_batch_size / (duration_us / 1e6);

        LOG(ERROR) << std::format("step {:4d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} tok/s)",
                                  step + 1, FLAGS_num_iteration, lossf, FLAGS_learning_rate, duration_us / 1e3f, tps);

        if ((step + 1) % FLAGS_freq_generate_txt == 0 && tokenizer) {
            LOG(FATAL) << "[TP] text generation skipped in this example.";
        }
    }
#ifdef PROFILE_MODE
    Profiler::Instance().Report("llama3_tp.report", Profiler::SortBy::DeviceTimePercentage);
    Profiler::Instance().PrintRecords("llama3_tp.records.log");
#endif

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
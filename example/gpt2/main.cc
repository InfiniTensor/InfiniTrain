#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"

#include "example/gpt2/dataset.h"
#include "example/gpt2/net.h"

// I/O
DEFINE_string(input_bin, "", "input .bin to train on");
DEFINE_string(input_val_bin, "", "input .bin to eval validation loss on");
DEFINE_string(model, "gpt2", "gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48");
// token layout for each step of the optimization
DEFINE_uint32(batch_size, 4, "batch size, in units of #batch dimensions");
DEFINE_uint32(sequence_length, 64, "sequence length");
DEFINE_uint32(total_batch_size, 256, "total desired batch size, in units of #tokens");
// workload (number of steps)
DEFINE_uint32(num_iteration, 10, "number of iterations to run");
// optimization
DEFINE_double(learning_rate, 1e-4, "learning rate warmup iterations");
// evaluation
DEFINE_uint32(val_loss_every, 0, "every how many steps to evaluate val loss?");
DEFINE_uint32(sample_every, 0, "how often to sample from the model?");
// debugging
DEFINE_bool(overfit_single_batch, true, "overfit just one batch of data");
// memory management
DEFINE_string(device, "cuda", "device type (cpu/cuda)");

using namespace infini_train;

namespace {
// validation
const std::unordered_set<std::string> kSupportedModels
    = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"};
constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";

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

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // select the device
    Device device;
    if (FLAGS_device == kDeviceCPU) {
        device = Device(DeviceType::kCPU, 0);
    } else {
        device = Device(DeviceType::kCUDA, 0);
    }

    // calculate gradient accumulation from the desired total batch size and the current run configuration
    const auto tokens_per_fwdbwd = FLAGS_batch_size * FLAGS_sequence_length;
    CHECK_EQ(FLAGS_total_batch_size % tokens_per_fwdbwd, 0);
    const auto grad_accum_steps = FLAGS_total_batch_size / tokens_per_fwdbwd;
    LOG(INFO) << "total desired batch size: " << FLAGS_total_batch_size
              << " => calculated gradient accumulation steps: " << grad_accum_steps;

    // rng / reproducibility
    // ManualSeed(42);

    // init the model, either from scratch or from OpenAI pretrained checkpoint
    GPT2Config model_config;
    std::unique_ptr<GPT2> model = nullptr;
    if (kModelToConfigs.count(FLAGS_model)) {
        model_config = kModelToConfigs.at(FLAGS_model);
        model = std::make_unique<GPT2>(model_config);
    } else {
        model = GPT2::FromPretrained(kStrToModelType.at(FLAGS_model));
    }
    model->To(device);

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

    // TODO(dcj): support more complex optimizer later
    auto optimizer = optimizers::SGD(model->Parameters(), FLAGS_learning_rate);

    auto train_iter = train_loader.begin();
    auto loss_fn = nn::CrossEntropyLoss();
    loss_fn.To(device);

    for (int step = 0; step < FLAGS_num_iteration; ++step) {
        const bool last_step = step == FLAGS_num_iteration - 1;
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
        for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            auto [x, y] = *train_iter;
            ++train_iter;
            x = std::make_shared<Tensor>(x->To(device));
            y = std::make_shared<Tensor>(y->To(device));
            auto logits = model->Forward({x, y})[0];
            auto loss = loss_fn.Forward({logits, y})[0];
            auto loss_cpu = loss->To(Device());
            lossf += static_cast<const float *>(loss_cpu.DataPtr())[0] / grad_accum_steps;
            loss->Backward();
        }
        optimizer.Step();
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}

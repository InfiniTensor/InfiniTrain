#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/tiny_mixtral/checkpoint_loader.h"
#include "example/tiny_mixtral/config.h"
#include "infini_train/include/autocast.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

DEFINE_string(input_bin, "", "input .bin to train on");
DEFINE_uint32(micro_batch_size, 4, "micro batch size per training step");
DEFINE_uint32(global_batch_size, 4, "global batch size across gradient accumulation and data parallelism");
DEFINE_uint32(sequence_length, 64, "sequence length");
DEFINE_uint32(num_iteration, 10, "number of training iterations");
DEFINE_double(learning_rate, 1e-4, "SGD learning rate");
DEFINE_string(llmc_filepath, "",
              "optional PyTorch-generated tiny Mixtral LLMC model file path to load before training");
DEFINE_string(device, "cpu", "Training device: cpu or cuda.");
DEFINE_string(dtype, "float32", "precision used in training (float32/bfloat16)");
DEFINE_uint32(log_interval, 1, "Print train loss every N steps. 0 disables step loss logging.");
DEFINE_bool(print_timing, false, "Print training-loop elapsed time and token throughput.");

namespace {

using infini_train::Device;
using infini_train::Tensor;

constexpr char kDtypeFP32[] = "float32";
constexpr char kDtypeBF16[] = "bfloat16";

void ValidateRuntimeFlags(const infini_train::nn::TransformerConfig &config) {
    CHECK(!FLAGS_input_bin.empty()) << "tiny Mixtral training requires --input_bin";
    CHECK_GT(FLAGS_micro_batch_size, 0);
    CHECK_GT(FLAGS_global_batch_size, 0);
    CHECK_EQ(FLAGS_global_batch_size % FLAGS_micro_batch_size, 0)
        << "global_batch_size must be divisible by micro_batch_size";
    CHECK_GT(FLAGS_sequence_length, 0);
    CHECK_LE(FLAGS_sequence_length, config.block_size) << "sequence_length must be <= model max positions (block_size)";
}

} // namespace

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    infini_train::nn::parallel::global::InitAllEnv(
        /*nthread_per_process=*/1,
        /*tensor_parallel_size=*/1,
        /*sequence_parallel_enabled=*/false,
        /*pipeline_parallel_size=*/1,
        /*virtual_pipeline_parallel_size=*/1);

    infini_train::nn::TransformerConfig model_config = tiny_mixtral::TinyMixtralConfig();
    tiny_mixtral::SanitizeTinyMixtralConfig(model_config);
    std::shared_ptr<infini_train::nn::TransformerModel> model = nullptr;
    if (!FLAGS_llmc_filepath.empty()) {
        model = tiny_mixtral::LoadFromLLMC(FLAGS_llmc_filepath, model_config);
    } else {
        model = std::make_shared<infini_train::nn::TransformerModel>(model_config);
    }
    ValidateRuntimeFlags(model_config);

    Device train_device;
    if (FLAGS_device == "cuda") {
        train_device = Device(Device::DeviceType::kCUDA, 0);
        model->To(train_device);
    } else {
        CHECK_EQ(FLAGS_device, "cpu") << "Unsupported training device: " << FLAGS_device;
        train_device = Device();
    }

    infini_train::DistributedDataLoader train_loader(
        std::make_shared<TinyShakespeareDataset>(FLAGS_input_bin, FLAGS_sequence_length), FLAGS_micro_batch_size,
        /*ddp_rank=*/0, /*ddp_world_size=*/1);
    auto train_iter = train_loader.begin();

    infini_train::DataType dtype;
    if (FLAGS_dtype == kDtypeFP32) {
        dtype = infini_train::DataType::kFLOAT32;
    } else if (FLAGS_dtype == kDtypeBF16) {
        dtype = infini_train::DataType::kBFLOAT16;
    } else {
        LOG(FATAL) << "Datatype " << FLAGS_dtype << " not supported.";
    }

    auto loss_fn = std::make_shared<infini_train::nn::CrossEntropyLoss>();
    auto optimizer
        = infini_train::optimizers::SGD::Create(static_cast<float>(FLAGS_learning_rate))(model->Parameters());

    auto device_impl = infini_train::core::GetDeviceGuardImpl(train_device.type());
    std::vector<double> step_duration_ms;
    step_duration_ms.reserve(FLAGS_num_iteration);
    const uint32_t grad_accum_steps = FLAGS_global_batch_size / FLAGS_micro_batch_size;
    const double tokens_per_step = static_cast<double>(FLAGS_global_batch_size) * FLAGS_sequence_length;
    for (uint32_t step = 0; step < FLAGS_num_iteration; ++step) {
        device_impl->SynchronizeDevice(train_device);
        const auto step_start_time = std::chrono::steady_clock::now();

        optimizer->ZeroGrad();
        float lossf = 0.0f;
        for (uint32_t micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            infini_train::AutocastGuard autocast_guard(train_device.type(), dtype);
            if (train_iter == train_loader.end()) {
                train_iter = train_loader.begin();
            }
            auto [x_cpu, y_cpu] = *train_iter;
            ++train_iter;
            auto x = std::make_shared<Tensor>(x_cpu->To(train_device));
            auto y = std::make_shared<Tensor>(y_cpu->To(train_device));
            auto logits = (*model)({x})[0];
            auto loss = (*loss_fn)({logits, y})[0];
            auto loss_cpu = loss->To(Device());
            lossf += static_cast<const float *>(loss_cpu.DataPtr())[0] / grad_accum_steps;
            loss = loss / static_cast<float>(grad_accum_steps);
            autocast_guard.Disable();
            loss->Backward();
        }
        optimizer->Step();

        device_impl->SynchronizeDevice(train_device);
        const auto step_end_time = std::chrono::steady_clock::now();
        const double duration_ms = std::chrono::duration<double, std::milli>(step_end_time - step_start_time).count();
        step_duration_ms.push_back(duration_ms);

        if (FLAGS_log_interval > 0 && ((step + 1) % FLAGS_log_interval == 0 || step + 1 == FLAGS_num_iteration)) {
            std::cout << std::format(
                "step {:4d}/{} | train loss {:.6f} | norm -1.0000 | lr {:.2e} | ({:.2f} ms | {:.0f} tok/s)", step + 1,
                FLAGS_num_iteration, lossf, FLAGS_learning_rate, duration_ms, tokens_per_step / (duration_ms / 1e3))
                      << std::endl;
        }
    }
    if (!step_duration_ms.empty()) {
        double duration_sum_ms = 0.0;
        for (size_t idx = step_duration_ms.size() > 1 ? 1 : 0; idx < step_duration_ms.size(); ++idx) {
            duration_sum_ms += step_duration_ms[idx];
        }
        const size_t averaged_steps
            = step_duration_ms.size() > 1 ? step_duration_ms.size() - 1 : step_duration_ms.size();
        std::cout << std::format("final {} iters avg: {:.3f}ms", averaged_steps, duration_sum_ms / averaged_steps)
                  << std::endl;
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}

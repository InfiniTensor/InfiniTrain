#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#ifdef USE_OMP
#include <omp.h>
#endif
#include "example/mnist/dataset.h"
#include "example/mnist/distributed.h"
#include "example/mnist/training.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/optimizer.h"

DEFINE_string(dataset, "", "Directory containing uncompressed MNIST IDX files");
DEFINE_int32(bs, 64, "Batch size");
DEFINE_int32(num_epoch, 3, "Number of complete training epochs");
DEFINE_double(lr, 0.05, "SGD learning rate, without momentum or weight decay");
DEFINE_string(device, "cpu", "Device: cpu or cuda");
DEFINE_uint32(seed, 42, "Initialization and per-epoch shuffle seed");
DEFINE_bool(shuffle, true, "Shuffle only the training split each epoch");
DEFINE_int32(threads, 4, "OpenMP thread count");
DEFINE_int32(log_interval, 100, "Steps between training progress records");
DEFINE_string(output_dir, "runs/mnist", "New run directory for config, metrics and checkpoint");
DEFINE_bool(eval_only, false, "Evaluate --checkpoint without training");
DEFINE_string(checkpoint, "", "Checkpoint directory, used only with --eval_only");
DEFINE_bool(ddp, false, "Single-node NCCL DDP; launch one process per GPU with infini_run");
DEFINE_bool(ddp_buckets, true, "Use the framework DDP gradient buckets (false: per-parameter all-reduce)");

using namespace infini_train;
namespace {
using Clock = std::chrono::steady_clock;
void WriteRecord(std::ofstream &out, const char *split, int epoch, int64_t step, const mnist::Metrics &m,
                 double seconds) {
    out << std::setprecision(12) << "{\"split\":\"" << split << "\",\"epoch\":" << epoch << ",\"step\":" << step
        << ",\"samples\":" << m.samples << ",\"correct\":" << m.correct << ",\"loss\":" << m.Loss()
        << ",\"accuracy\":" << m.Accuracy() << ",\"seconds\":" << seconds << "}\n";
    out.flush();
    CHECK(out.good()) << "Failed to write metrics";
    std::cout << std::fixed << std::setprecision(6) << split << " epoch=" << epoch << " step=" << step
              << " samples=" << m.samples << " loss=" << m.Loss() << " accuracy=" << m.Accuracy()
              << " seconds=" << seconds << std::endl;
}
} // namespace
int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    CHECK(!FLAGS_dataset.empty()) << "--dataset is required";
    CHECK(FLAGS_device == "cpu" || FLAGS_device == "cuda");
    CHECK_GT(FLAGS_bs, 0);
    CHECK_GT(FLAGS_threads, 0);
    CHECK_GT(FLAGS_log_interval, 0);
    CHECK_GT(FLAGS_num_epoch, 0);
    CHECK(std::isfinite(FLAGS_lr) && FLAGS_lr > 0);
    CHECK(FLAGS_eval_only == !FLAGS_checkpoint.empty()) << "Use --checkpoint together with --eval_only";
#ifndef USE_CUDA
    CHECK_EQ(FLAGS_device, "cpu") << "Rebuild with USE_CUDA=ON";
#endif
#ifdef USE_OMP
    omp_set_dynamic(0);
    omp_set_num_threads(FLAGS_threads);
#endif
    const mnist::DistributedContext distributed(FLAGS_ddp, FLAGS_device == "cuda");
    const auto device = distributed.device;
    const bool main_rank = distributed.rank == 0;
    const std::filesystem::path output_dir(FLAGS_output_dir);
    CHECK(!FLAGS_output_dir.empty());
    if (main_rank) {
        CHECK(!std::filesystem::exists(output_dir) || std::filesystem::is_empty(output_dir))
            << "Choose a new --output_dir to preserve previous results";
        std::filesystem::create_directories(output_dir);
    }
    distributed.Barrier();
    const auto rank_dir = FLAGS_ddp ? output_dir / ("rank" + std::to_string(distributed.rank)) : output_dir;
    std::filesystem::create_directories(rank_dir);
    std::ofstream config(rank_dir / "config.txt");
    CHECK(config.is_open());
    config << "model=Conv2d(1,16,3)-ReLU-Conv2d(16,32,3)-ReLU-Flatten-Linear(18432,10)\n"
           << "dtype=float32\nnormalization=pixel/255\noptimizer=SGD\nmomentum=0\nweight_decay=0\n"
           << "dataset=" << std::filesystem::absolute(FLAGS_dataset) << "\ndevice=" << FLAGS_device
           << "\nbatch_size=" << FLAGS_bs << "\nepochs=" << FLAGS_num_epoch << "\nlearning_rate=" << FLAGS_lr
           << "\nseed=" << FLAGS_seed << "\nshuffle=" << FLAGS_shuffle << "\nthreads=" << FLAGS_threads
           << "\neval_only=" << FLAGS_eval_only << "\ncheckpoint=" << FLAGS_checkpoint << "\nddp=" << FLAGS_ddp
           << "\nworld_size=" << distributed.world_size << "\nrank=" << distributed.rank
           << "\nlocal_device=" << static_cast<int>(device.index())
           << "\nglobal_batch_size=" << static_cast<int64_t>(FLAGS_bs) * distributed.world_size
           << "\nddp_buckets=" << FLAGS_ddp_buckets << '\n';
    config.close();
    std::ofstream metrics;
    if (main_rank) {
        metrics.open(output_dir / "metrics.jsonl");
        CHECK(metrics.is_open());
    }
    auto net = std::make_shared<MNIST>();
    mnist::Initialize(*net, FLAGS_seed);
    TrainerState state;
    if (FLAGS_eval_only) {
        Checkpoint::Load(FLAGS_checkpoint, *net, nullptr, state, nullptr);
    }
    net->To(device);
    auto model = distributed.Wrap(net, FLAGS_ddp_buckets);
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);
    CHECK_GT(test_dataset->Size(), 0);
    auto test_shard = std::make_shared<mnist::ShuffledDataset>(test_dataset, distributed.rank, distributed.world_size);
    DataLoader test_loader(test_shard, FLAGS_bs);
    auto evaluate = [&](int epoch) {
        const auto start = Clock::now();
        const auto result = distributed.Sum(mnist::Evaluate(*net, test_loader, device));
        CHECK_EQ(result.samples, test_dataset->Size());
        if (main_rank) {
            WriteRecord(metrics, "test", epoch, state.global_step, result,
                        std::chrono::duration<double>(Clock::now() - start).count());
        }
    };
    evaluate(0);
    if (!FLAGS_eval_only) {
        auto train_data = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
        CHECK_GT(train_data->Size(), 0);
        auto sampler
            = std::make_shared<mnist::ShuffledDataset>(train_data, distributed.rank, distributed.world_size, true);
        CHECK_GT(sampler->Size(), 0) << "Training dataset must have at least world_size samples";
        DataLoader train_loader(sampler, FLAGS_bs);
        config.open(rank_dir / "config.txt", std::ios::app);
        config << "train_samples=" << train_data->Size() << "\ntest_samples=" << test_dataset->Size()
               << "\nlocal_train_samples=" << sampler->Size()
               << "\ndropped_train_samples=" << train_data->Size() % distributed.world_size << '\n';
        config.close();
        nn::CrossEntropyLoss loss_fn;
        optimizers::SGD optimizer(net->Parameters(), FLAGS_lr);
        for (int epoch = 1; epoch <= FLAGS_num_epoch; ++epoch) {
            sampler->Reset(FLAGS_seed + static_cast<uint32_t>(epoch), FLAGS_shuffle);
            mnist::Metrics train_metrics;
            const auto start = Clock::now();
            for (const auto &[image, label] : train_loader) {
                auto x = std::make_shared<Tensor>(image->To(device));
                auto target = std::make_shared<Tensor>(label->To(device));
                optimizer.ZeroGrad();
                auto logits = (*model)({x})[0];
                auto loss = loss_fn({logits, target})[0];
                loss->Backward();
                optimizer.Step();
                mnist::Accumulate(train_metrics, logits, loss, label);
                ++state.global_step;
                state.consumed_train_samples += image->Dims()[0] * distributed.world_size;
                if (state.global_step % FLAGS_log_interval == 0) {
                    const auto global_metrics = distributed.Sum(train_metrics);
                    if (main_rank) {
                        WriteRecord(metrics, "train_progress", epoch, state.global_step, global_metrics,
                                    std::chrono::duration<double>(Clock::now() - start).count());
                    }
                }
            }
            CHECK_EQ(train_metrics.samples, sampler->Size());
            const auto global_metrics = distributed.Sum(train_metrics);
            CHECK_EQ(global_metrics.samples, train_data->Size() / distributed.world_size * distributed.world_size);
            if (main_rank) {
                WriteRecord(metrics, "train", epoch, state.global_step, global_metrics,
                            std::chrono::duration<double>(Clock::now() - start).count());
            }
            evaluate(epoch);
        }
        optimizer.ZeroGrad();
        distributed.VerifyReplicas(*net);
        if (FLAGS_ddp) {
            std::ofstream replica_check(rank_dir / "replica_check.txt");
            replica_check << "All 189130 parameters finite and bitwise equal to rank 0\n";
            CHECK(replica_check.good());
        }
        if (main_rank) {
            net->To(Device());
            core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
            Checkpoint::Save(output_dir / "checkpoint", *net, nullptr, state, nullptr);
        }
    }
    distributed.Barrier();
    if (main_rank) {
        std::cout << "Results: " << std::filesystem::absolute(output_dir) << std::endl;
    }
    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}

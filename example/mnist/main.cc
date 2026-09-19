#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>


#include <algorithm>
#include <cstdint>
#include <thread>



#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"

#include "example/mnist/dataset.h"
#include "example/mnist/net.h"



#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"

DEFINE_string(dataset, "", "mnist dataset path");
DEFINE_int32(bs, 64, "batch size");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");
DEFINE_string(device, "cpu", "device type (cpu/cuda)");



DEFINE_string(model, "mlp", "model type (mlp/cnn)");
DEFINE_string(optimizer, "sgd", "optimizer type (sgd/adam)");


DEFINE_int32(nthread_per_process, 1, "Number of threads to use for each process. >1 enables DDP on visible CUDA devices.");


using namespace infini_train;

namespace {
constexpr int kNumItersOfOutputDuration = 10;
constexpr int kNumClasses = 10;

constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";



constexpr char kModelMLP[] = "mlp";
constexpr char kModelCNN[] = "cnn";
constexpr char kOptimizerSGD[] = "sgd";
constexpr char kOptimizerAdam[] = "adam";

}; // namespace

DEFINE_validator(device,
                 [](const char *, const std::string &value) { return value == kDeviceCPU || value == kDeviceCUDA; });

DEFINE_validator(model, [](const char *, const std::string &value) { return value == kModelMLP || value == kModelCNN; });
DEFINE_validator(optimizer, [](const char *, const std::string &value) {
    return value == kOptimizerSGD || value == kOptimizerAdam;
});


DEFINE_validator(nthread_per_process, [](const char *, int32_t value) {
    return value >= 1;
});

// 重构： 把原来main()里面的训练逻辑整体搬进Train(rank), 入口结构对齐 example/gpt2/main.cc
void Train(const nn::parallel::Rank &rank) {
    using namespace nn::parallel;

    // 设置thread-local global rank, DDP/ProcessGroup 依赖它定位当前线程归属
    global::thread_global_rank = rank.GlobalRank();

    const int ddp_world_size = global::GetDataParallelSize();
    int ddp_rank = 0;

    Device device;
    if (rank.IsParallel()) {
        // 并行模式下忽略 --device, 按照 thread_rank选择CUDA卡
        device = Device(Device::DeviceType::kCUDA, global::GetDeviceIndex(rank.thread_rank()));

        auto *pg_factory = ProcessGroupFactory::Instance(device.type());
        if (ddp_world_size > 1) {
            const auto *ddp_pg = pg_factory->GetOrCreate(GetDataParallelProcessGroupName(rank.GlobalRank()), GetDataParallelGroupRanks(rank.GlobalRank()));
            ddp_rank = ddp_pg->GetGroupRank(rank.GlobalRank());
        }

    } else {
        // 保留单卡和CPU的选择
        device = FLAGS_device == kDeviceCPU ? Device() : Device(Device::DeviceType::kCUDA, 0);
    }
    const Device cpu_device = Device();


    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);

    // 分线程保障
    const size_t batch_size = static_cast<size_t>(FLAGS_bs);
    const size_t train_num_batches = (train_dataset->Size() + batch_size - 1) / batch_size;
    if (ddp_world_size > 1) {
        CHECK_EQ(train_num_batches % static_cast<size_t>(ddp_world_size), 0)
            << "MNIST train batches must be divisible by ddp_world_size. "
            << "train_size=" << train_dataset->Size() << ", bs=" << FLAGS_bs
            << ", batches=" << train_num_batches << ", ddp_world_size=" << ddp_world_size
            << ". For 4 GPUs use --bs 60; for 2 GPUs --bs 64 is OK.";
    }

    // DDP用DistributedDataLoader做rank分片；而单卡保持DataLoader
    std::shared_ptr<DataLoader> train_dataloader;
    if (ddp_world_size > 1) {
        train_dataloader = std::make_shared<DistributedDataLoader>(
            train_dataset, batch_size, static_cast<size_t>(ddp_rank), static_cast<size_t>(ddp_world_size)
        );

    } else {
        train_dataloader = std::make_shared<DataLoader>(train_dataset, batch_size);
    }

    // 测试集仍然用DataLoader;评测在main rank做
    DataLoader test_dataloader(test_dataset, batch_size);

    //原模型选择逻辑
    std::shared_ptr<nn::Module> network;
    if (FLAGS_model == kModelMLP) {
        network = std::make_shared<MNIST>();
    } else {
        network = std::make_shared<MNISTCNN>();
    }
    network->To(device);


    // 先To(device) 再DDP，最后optimizer
    // DDP构造会基于当前参数注册backward hook; 如果之后再换参数/搬设备，hook有可能失效
    std::shared_ptr<nn::Module> eval_module = network;
    if (ddp_world_size > 1) {
        DistributedDataParallelConfig ddp_config{.zero_stage = 0};
        // zero_stage=0: 纯DDP梯度AllReduce， 不用ZeRO
        network = std::make_shared<DistributedDataParallel>(network, rank, ddp_config);

        // 评测时 unwrap到内部module，避免eval forward触发DDP reducer的PrepareForBackward
        eval_module = std::dynamic_pointer_cast<DistributedDataParallel>(network)->module();
        CHECK(eval_module) << "Failed to unwrap DDP module for evaluation.";
    }

    if (rank.IsMainRank()) {
        LOG(ERROR) << "model: " << FLAGS_model << ", ddp_world_size: " << ddp_world_size
                    << ", ddp_rank: " << ddp_rank;
    }

    auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();
    loss_fn->To(device);

    // 原有的optimizer选择逻辑；注意必须再DDP包装后创建，以保证拿到的是包装后的参数列表
    std::shared_ptr<Optimizer> optimizer;
    if (FLAGS_optimizer == kOptimizerSGD) {
        optimizer = std::make_shared<optimizers::SGD>(network->Parameters(), FLAGS_lr);
    } else {
        optimizer = std::make_shared<optimizers::Adam>(network->Parameters(), FLAGS_lr);
    }

    if (rank.IsMainRank()) {
        LOG(ERROR) << "optimizer: " << FLAGS_optimizer << ", lr: " << FLAGS_lr;
    }


    // DDP： 1.只在main rank评测，避免多rank重复输出； 2.评测基于 eval_module (DDP时为内部 module), 训练仍走 network (DDP wrapper)
    auto evaluate = [&](int epoch_id) {
        if (!rank.IsMainRank()) {
            return;
        }

        //框架无 no_grad，评测前把参数临时换成Detach副本，避免 eval 建图污染依赖计数
        const auto named_params = eval_module->NamedParameters();
        for (const auto &[name, param] : named_params) {
            const auto dot = name.rfind('.');
            *eval_module->mutable_module(name.substr(0, dot))->mutable_parameter(name.substr(dot + 1)) = param->Detach();
        }

        std::vector<float> test_losses;
        int correct = 0;
        int total = 0;
        for (const auto &[image, label] : test_dataloader) {
            auto new_image = std::make_shared<Tensor>(image->To(device));
            auto new_label = std::make_shared<Tensor>(label->To(device));


            auto label_cpu = new_label->To(cpu_device);
            auto outputs = eval_module->Forward({new_image});
            auto output_cpu = outputs[0]->To(cpu_device);
            auto loss = loss_fn->Forward({outputs[0], new_label});
            auto loss_cpu = loss[0]->To(cpu_device);

            const int cur_batch_size = output_cpu.Dims()[0];
            for (int batch_idx = 0; batch_idx < cur_batch_size; ++batch_idx) {
                auto label_index = reinterpret_cast<uint8_t *>(label_cpu.DataPtr())[batch_idx];
                const auto *output_values = static_cast<float *>(output_cpu.DataPtr()) + batch_idx * kNumClasses;
                const int output_index = std::max_element(output_values, output_values + kNumClasses) - output_values;

                if (output_index == label_index) {
                    ++correct;
                }
            }

            total += cur_batch_size;
            test_losses.push_back(static_cast<float *>(loss_cpu.DataPtr())[0]);
        }

        // 在评测结束后恢复原始参数对象
        for (const auto &[name, param] : named_params) {
            const auto dot = name.rfind('.');
            *eval_module->mutable_module(name.substr(0, dot))->mutable_parameter(name.substr(dot + 1)) = param;
        }

        const auto avg_loss = std::accumulate(test_losses.begin(), test_losses.end(), 0.0) / test_losses.size();
        LOG(ERROR) << std::format("epoch {:2d} | test | Total: {}, Correct: {}, Accuracy: {:.4f}, AverageLoss: {:.6f}", epoch_id, total, correct, static_cast<float>(correct) / total, avg_loss);
    };


    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        int train_idx = 0;
        float total_loss = 0.0;

        const auto epoch_start = std::chrono::high_resolution_clock::now();

        for (const auto &[image, label] : *train_dataloader) {
            auto new_image = std::make_shared<Tensor>(image->To(device));
            auto new_label = std::make_shared<Tensor>(label->To(device));

            auto outputs = network->Forward({new_image});
            optimizer->ZeroGrad();

            auto loss = loss_fn->Forward({outputs[0], new_label});
            loss[0]->Backward();

            // loss的D2H拷贝放在backward之后，避免前反向之间多一次同步
            auto loss_cpu = loss[0]->To(cpu_device);
            const float current_loss = static_cast<float *>(loss_cpu.DataPtr())[0];
            total_loss += current_loss;

            // 只有main rank打印训练日志; DDP下把进度换算成全局样本数
            if (rank.IsMainRank() && train_idx % kNumItersOfOutputDuration == 0) {
                LOG(ERROR) << "epoch: " << epoch << ", ["
                            << train_idx * FLAGS_bs * static_cast<int64_t>(ddp_world_size) << "/"
                            << train_dataset->Size() << "] "
                            << " loss: " << current_loss;
            }

            optimizer->Step();
            train_idx += 1;
        }

        const auto epoch_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(epoch_end - epoch_start).count();

        // 只有main rank打印 epoch汇总； samples/s 按全局训练集大小/ rank0 epoch 时长估算
        if (rank.IsMainRank()) {
            LOG(ERROR) << std::format(
                "epoch {:2d}/{} | train loss {:.6f} | lr {:.2e} | ddp {} | ({:.2f} ms | {:.0f} samples/s)", epoch,
                FLAGS_num_epoch - 1, total_loss / train_idx, FLAGS_lr, ddp_world_size, duration_us / 1e3f,
                train_dataset->Size() / (duration_us / 1e6f)
            );
        }

        evaluate(epoch);
    }
}


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // DDP仅支持CUDA； CPU多线程 DP 不在本任务范围内
    if (FLAGS_nthread_per_process > 1 && FLAGS_device != kDeviceCUDA) {
        LOG(FATAL) << "DDP mode requires CUDA devices.Please use --device cuda with --nthread_per_process > 1.";
    }

    // 对齐gpt2入口. MNIST只启用DP，而不用TP/SP/PP
    nn::parallel::global::InitAllEnv(FLAGS_nthread_per_process, 1, false, 1, 1);
    LOG(INFO) << nn::parallel::global::ProcessGroupOverview();

    // nthread_per_process>1时， 每个线程绑定一个rank, 执行一个Train(rank)
    if (FLAGS_nthread_per_process > 1) {
        std::vector<std::thread> threads;
        for (int idx = 0; idx < FLAGS_nthread_per_process; ++idx) {
            nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), idx, nn::parallel::global::GetNprocPerNode(), FLAGS_nthread_per_process);
            threads.emplace_back(Train, rank);
        }

        for (auto &thread : threads) { thread.join(); }
    } else {
        nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), 0, nn::parallel::global::GetNprocPerNode(), FLAGS_nthread_per_process);
        Train(rank);
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}




// int main(int argc, char *argv[]) {
//     gflags::ParseCommandLineFlags(&argc, &argv, true);
//     google::InitGoogleLogging(argv[0]);

//     auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
//     DataLoader train_dataloader(train_dataset, FLAGS_bs);

//     // TODO(dcj): Add sampler & eval dataloader later.
//     auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);
//     DataLoader test_dataloader(test_dataset, FLAGS_bs);

//     // auto network = MNIST();
//     //////// auto network = std::make_shared<MNIST>();

//     // 改动加入： mlp 和CNN
//     std::shared_ptr<nn::Module> network;
//     if (FLAGS_model == kModelMLP) {
//         network = std::make_shared<MNIST>();
//     } else {
//         network = std::make_shared<MNISTCNN>();
//     }
//     LOG(ERROR) << "model: " << FLAGS_model;
//     /////////////////////////////

//     Device device = FLAGS_device == kDeviceCPU ? Device() : Device(Device::DeviceType::kCUDA, 0);
//     Device cpu_device = Device();
//     // network.To(device);
//     network->To(device);

//     // auto loss_fn = nn::CrossEntropyLoss();
//     // loss_fn.To(device);
//     auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();
//     loss_fn->To(device);
//     //auto optimizer = optimizers::SGD(network.Parameters(), FLAGS_lr);
//     ////////////////auto optimizer = optimizers::SGD(network->Parameters(), FLAGS_lr);
//     std::shared_ptr<Optimizer> optimizer;
//     if (FLAGS_optimizer == kOptimizerSGD) {
//         optimizer = std::make_shared<optimizers::SGD>(network->Parameters(), FLAGS_lr);
//     } else {
//         optimizer = std::make_shared<optimizers::Adam>(network->Parameters(), FLAGS_lr);
//     }
//     LOG(ERROR) << "optimizer: " << FLAGS_optimizer << ", lr: " << FLAGS_lr;
//     //////////////////////////////////





//     // // ===== overfit single batch test =====
//     // {
//     //     auto first = *train_dataloader.begin();
//     //     for (int step = 0; step < 300; ++step) {
//     //         auto new_image = std::make_shared<Tensor>(first.first->To(device));
//     //         auto new_label = std::make_shared<Tensor>(first.second->To(device));
//     //         auto outputs = network->Forward({new_image});
//     //         optimizer.ZeroGrad();
//     //         auto loss = loss_fn->Forward({outputs[0], new_label});
//     //         loss[0]->Backward();
//     //         auto loss_cpu = loss[0]->To(cpu_device);
//     //         if (step % 20 == 0) {
//     //             LOG(ERROR) << "[overfit] step " << step
//     //                        << " loss: " << static_cast<float *>(loss_cpu.DataPtr())[0];
//     //         }
//     //         optimizer.Step();
//     //     }
//     // }
//     // // =======================================
//     // 增加:每个epoch结束后调用一次测试集评测
//     auto evaluate = [&](int epoch_id) {

//         const auto named_params = network->NamedParameters();
//         for (const auto &[name, param] : named_params) {
//             const auto dot = name.rfind('.');
//             *network->mutable_module(name.substr(0, dot))->mutable_parameter(name.substr(dot + 1)) = param->Detach();
//         }

//         std::vector<float> test_losses;
//         int correct = 0;
//         int total = 0;
//         for (const auto &[image, label] : test_dataloader) {
//             auto new_image = std::make_shared<Tensor>(image->To(device));
//             auto new_label = std::make_shared<Tensor>(label->To(device));

//             auto label_cpu = label->To(cpu_device);
//             auto outputs = network->Forward({new_image});
//             auto output_cpu = outputs[0]->To(cpu_device);
//             auto loss = loss_fn->Forward({outputs[0], new_label});
//             auto loss_cpu = loss[0]->To(cpu_device);

//             const int batch_size = output_cpu.Dims()[0];
//             for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
//                 auto label_index = reinterpret_cast<uint8_t *>(label_cpu.DataPtr())[batch_idx];
//                 const auto *output_values = static_cast<float *>(output_cpu.DataPtr()) + batch_idx * kNumClasses;
//                 const int output_index = std::max_element(output_values, output_values + kNumClasses) - output_values;
//                 if (output_index == label_index) {
//                     ++correct;
//                 }
//             }

//             total += batch_size;
//             test_losses.push_back(static_cast<float *>(loss_cpu.DataPtr())[0]);
//         }


//         for (const auto &[name, param] : named_params) {
//             const auto dot = name.rfind('.');
//             *network->mutable_module(name.substr(0, dot))->mutable_parameter(name.substr(dot + 1)) = param;
//         }

//         const auto avg_loss = std::accumulate(test_losses.begin(), test_losses.end(), 0.0) / test_losses.size();
//         LOG(ERROR) << std::format("epoch {:2d} | test | Total: {}, Correct: {}, Accuracy: {:.4f}, AverageLoss: {:.6f}", epoch_id, total, correct, static_cast<float>(correct) / total, avg_loss);
//     };
//     //////////////////////////////



//     for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
//         int train_idx = 0;
//         float total_loss = 0.0;

//         const auto epoch_start = std::chrono::high_resolution_clock::now();

//         for (const auto &[image, label] : train_dataloader) {
//             auto new_image = std::make_shared<Tensor>(image->To(device));
//             auto new_label = std::make_shared<Tensor>(label->To(device));

//             // // ===== debug pairing =====
//             // if (train_idx == 0 || train_idx == 1) {
//             //     auto lc = new_label->To(cpu_device);
//             //     auto ic = new_image->To(cpu_device);
//             //     const auto *lp = static_cast<const uint8_t *>(lc.DataPtr());
//             //     const float *ip = static_cast<const float *>(ic.DataPtr());
//             //     for (int i = 0; i < 16; ++i) {
//             //         double s = 0.0;
//             //         for (int j = 0; j < 784; ++j) { s += ip[i * 784 + j]; }
//             //         LOG(ERROR) << "[debug] batch " << train_idx << " sample " << i
//             //                    << " | label = " << static_cast<int>(lp[i])
//             //                    << " | image sum = " << s;
//             //     }
//             // }
//             // // ===========================

//             // auto outputs = network.Forward({new_image});
//             auto outputs = network->Forward({new_image});
//             //optimizer.ZeroGrad();
//             // 变成指针了
//             optimizer->ZeroGrad();

//             // auto loss = loss_fn.Forward({outputs[0], new_label});
//             auto loss = loss_fn->Forward({outputs[0], new_label});
//             loss[0]->Backward();

//             // // ===== debug start =====
//             // const auto params = network->Parameters();   // 关键：先把临时 vector 接住
//             // for (size_t pi = 0; pi < params.size(); ++pi) {
//             //     const auto &p = params[pi];              // 现在引用有效
//             //     const auto g = p->grad();
//             //     if (!g) {
//             //         LOG(ERROR) << "[debug] param " << pi << " grad is NULL";
//             //         continue;
//             //     }
//             //     auto gc = g->To(cpu_device);
//             //     const float *gd = static_cast<const float *>(gc.DataPtr());
//             //     double sum = 0.0;
//             //     for (size_t j = 0; j < g->NumElements(); ++j) { sum += gd[j] * gd[j]; }
//             //     LOG(ERROR) << "[debug] param " << pi << " grad norm = " << std::sqrt(sum);
//             // }
//             // auto pb = network->Parameters()[0]->To(cpu_device);
//             // LOG(ERROR) << "[debug] w0[0] before step: " << static_cast<float *>(pb.DataPtr())[0];
//             // // ===== debug end =====

//             // Defer the loss D2H copy until after backward; reading it earlier would synchronize CUDA
//             // between forward and backward.
//             auto loss_cpu = loss[0]->To(cpu_device);
//             float current_loss = static_cast<float *>(loss_cpu.DataPtr())[0];
//             total_loss += current_loss;
//             if (train_idx % kNumItersOfOutputDuration == 0) {
//                 LOG(ERROR) << "epoch: " << epoch << ", [" << train_idx * FLAGS_bs << "/" << train_dataset->Size()
//                            << "] "
//                            << " loss: " << current_loss;
//             }

//             //optimizer.Step();
//             // 变成指针
//             optimizer->Step();
//             train_idx += 1;
//         }

//         const auto epoch_end = std::chrono::high_resolution_clock::now();
//         const double duration_us = std::chrono::duration<double, std::micro>(epoch_end - epoch_start).count();

//         LOG(ERROR) << std::format("epoch {:2d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} samples/s)",
//                                   epoch, FLAGS_num_epoch - 1, total_loss / train_idx, FLAGS_lr, duration_us / 1e3f,
//                                   train_dataset->Size() / (duration_us / 1e6));

//         // 每个epoch结束评测一次
//         evaluate(epoch);
//     }

//     // // TODO(dcj): Add no_grad() context manager later.
//     // std::vector<float> test_losses;
//     // int correct = 0;
//     // int total = 0;
//     // for (const auto &[image, label] : test_dataloader) {
//     //     auto new_image = std::make_shared<Tensor>(image->To(device));
//     //     auto new_label = std::make_shared<Tensor>(label->To(device));

//     //     auto label_cpu = label->To(cpu_device);
//     //     // auto outputs = network.Forward({new_image});
//     //     auto outputs = network->Forward({new_image});
//     //     auto output_cpu = outputs[0]->To(cpu_device);
//     //     // auto loss = loss_fn.Forward({outputs[0], new_label});
//     //     auto loss = loss_fn->Forward({outputs[0], new_label});
//     //     auto loss_cpu = loss[0]->To(cpu_device);

//     //     const int batch_size = output_cpu.Dims()[0];
//     //     for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
//     //         auto label_index = reinterpret_cast<uint8_t *>(label_cpu.DataPtr())[batch_idx];
//     //         const auto *output_values = static_cast<float *>(output_cpu.DataPtr()) + batch_idx * kNumClasses;
//     //         const int output_index = std::max_element(output_values, output_values + kNumClasses) - output_values;
//     //         if (output_index == label_index) {
//     //             ++correct;
//     //         }
//     //     }
//     //     total += batch_size;
//     //     test_losses.push_back(static_cast<float *>(loss_cpu.DataPtr())[0]);
//     // }
//     // const auto avg_loss = std::accumulate(test_losses.begin(), test_losses.end(), 0.0) / test_losses.size();
//     // LOG(ERROR) << "Total: " << total << ", Correct: " << correct
//     //            << ", Accuracy: " << static_cast<float>(correct) / total << ", AverageLoss: " << avg_loss;

//     gflags::ShutDownCommandLineFlags();
//     google::ShutdownGoogleLogging();

//     return 0;
// }

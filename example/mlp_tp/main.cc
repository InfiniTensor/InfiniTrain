#include <atomic>
#include <chrono>
#include <format>
#include <memory>
#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/optimizer.h"

#include "example/mlp_tp/net.h" 

using namespace infini_train;
namespace nn = infini_train::nn;
namespace tp = infini_train::nn::parallel;

// ===== CLI flags =====
DEFINE_uint32(tp_world_size, 0, "Tensor Parallel world size (default: use all visible CUDA devices)");
DEFINE_uint32(batch_size, 8, "batch size");
DEFINE_uint32(seq_len, 128, "sequence length");
DEFINE_uint32(steps, 50, "training steps");
DEFINE_double(lr, 1e-3, "learning rate");
DEFINE_uint32(n_embd, 768, "embedding dim (input/output dim of MLP)");
DEFINE_uint32(hidden_dim, 3072, "hidden dim (usually 4 * n_embd)");

// ===== helpers =====
static std::shared_ptr<Tensor> RandnLike(const std::vector<int64_t>& shape, DataType dt, const Device* dev) {
    auto t = std::make_shared<Tensor>(shape, dt, dev);
    // 如果你有 RandomNormal/Uniform 的 API，替换这里：
    t->Fill<float>(0.01f); // 占位：给一个非零初值，方便梯度流动
    return t;
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // --- 设备与 TP 组 ---
    auto* dm = DeviceManager::Instance();
    std::vector<const Device*> cuda_devs = dm->GetAllAvailableDevices(DeviceType::kCUDA);
    CHECK(!cuda_devs.empty()) << "No CUDA devices found.";

    const uint32_t world_size = (FLAGS_tp_world_size == 0)
        ? static_cast<uint32_t>(cuda_devs.size())
        : std::min<uint32_t>(FLAGS_tp_world_size, cuda_devs.size());
    CHECK_GT(world_size, 0u);

    tp::TensorParallelGroup tp_group;
    tp_group.devices.assign(cuda_devs.begin(), cuda_devs.begin() + world_size);

    LOG(INFO) << "TP world size = " << world_size;

    // --- 构建每个 rank 的模型与优化器 ---
    std::vector<std::shared_ptr<TensorParallelMLP>> models(world_size);
    std::vector<std::unique_ptr<Optimizer>> opts(world_size);

    for (uint32_t r = 0; r < world_size; ++r) {
        models[r] = std::make_shared<TensorParallelMLP>(
            /*n_embd=*/FLAGS_n_embd,
            /*hidden_dim=*/FLAGS_hidden_dim,
            /*tp_group=*/tp_group,
            /*fc_bias=*/true,
            /*proj_bias=*/true);
        models[r]->To(tp_group.devices[r]);

        // 每 rank 一个优化器
        opts[r] = std::make_unique<optimizers::SGD>(models[r]->Parameters(), FLAGS_lr);
    }

    // --- loss 函数与 dtype ---
    auto* cpu = dm->GetDefaultDevice();
    const DataType dtype = DataType::kFLOAT32;
    auto mse = nn::CrossEntropyLoss();
    mse.To(tp_group.devices[0]); // 我们在 rank0 上算 loss

    // --- 训练循环 ---
    for (uint32_t step = 0; step < FLAGS_steps; ++step) {
        const auto t0 = std::chrono::high_resolution_clock::now();

        // 准备一个共享输入（复制到每个 rank 的 device）
        const std::vector<int64_t> x_shape = {static_cast<int64_t>(FLAGS_batch_size),
                                              static_cast<int64_t>(FLAGS_seq_len),
                                              static_cast<int64_t>(FLAGS_n_embd)};
        auto x_host = RandnLike(x_shape, dtype, cpu);

        // 每个 rank 的输入副本
        std::vector<std::shared_ptr<Tensor>> x_rank(world_size);
        for (uint32_t r = 0; r < world_size; ++r) {
            x_rank[r] = std::make_shared<Tensor>(x_host->To(tp_group.devices[r]));
        }

        // 前向输出缓存（每 rank 一份）
        std::vector<std::shared_ptr<Tensor>> y_rank(world_size);

        // 优化器清零
        for (uint32_t r = 0; r < world_size; ++r) opts[r]->ZeroGrad();

        // --- 多线程前向 ---
        std::vector<std::thread> threads;
        threads.reserve(world_size);
        for (uint32_t r = 0; r < world_size; ++r) {
            threads.emplace_back([&, r]() {
                // 每个 rank 的前向：x -> TP-MLP
                y_rank[r] = models[r]->Forward({x_rank[r]})[0];
            });
        }
        for (auto& th : threads) th.join();

        // 由于 RowParallelLinear(reduce_output=true)，理论上各 rank 的输出一致
        // 在 rank0 上计算 MSE 到一个零目标，发起反向
        auto y0 = y_rank[0];
        auto target0 = std::make_shared<Tensor>(y0->Dims(), y0->Dtype(), y0->GetDevice());
        target0->Fill<float>(0.0f);

        auto loss = mse.Forward({y0, target0})[0];
        // 你可以加上 /grad_accum_steps 的缩放，这里演示单步
        loss->Backward();

        // 所有 rank 的参数 step
        for (uint32_t r = 0; r < world_size; ++r) {
            opts[r]->Step();
        }

        // 取 loss 数值
        float lossf = 0.0f;
        auto loss_cpu = loss->To(cpu);
        lossf = static_cast<const float*>(loss_cpu.DataPtr())[0];

        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LOG(INFO) << std::format("step {:4d}/{} | loss {:.6f} | lr {:.2e} | {:.2f} ms",
                                 step + 1, FLAGS_steps, lossf, FLAGS_lr, ms);
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}

# 小模型训练支持 — 项目报告（分支 `work/small-model-cnn-impl`）

本报告对应 PR「【训练营】小模型训练支持」：功能代码 + CMake 集成 + 单元测试 + MNIST CNN 演示。
PyTorch 对齐脚本见 `scripts/mnist_parity_torch.py`，C++ dump 端见 `tests/example/mnist_parity.cc`。

---

## 一、设计与实现

### 1.1 整体方案

目标是在 InfiniTrain 上跑通小模型（MNIST CNN）端到端训练，同时保证数学正确性可审计。
方案是第三版取舍：采用 PR #219 的语义设计（数据形状语义、DDP metric 设计）+ PR #220 的 kernel 与验证方法（kernel 实现、验证 discipline），去掉两边共有的 demo-level special case。

整体训练链路：

```text
MNISTDataset（单样本 [1,28,28]）
  → DataLoader Stack（shape-preserving，batch 后 [B,1,28,28]）
  → MnistCnn 前向（Conv→ReLU→Conv→ReLU→Flatten→Linear）
  → CrossEntropyLoss（mean）
  → Backward（autograd selective-save）
  → SGD 更新（lr=0.01）
  → 可选 DDP（Broadcast → wrap，训练只同步梯度；eval 分片 + epoch 级 AllReduce）
```

Oracle 策略（双独立 oracle，不依赖 CUDA direct 中间实现）：

```text
            PyTorch
               │
               ▼
CPU direct ─────── CUDA GEMM
   │                   │
   └── finite diff ────┘
```

即 CPU numerical gradcheck（tiny tensor，dX/dW/db）+ PyTorch 对标（logits / loss / grad / 单步更新 / 多步趋势）。

### 1.2 CNN 相关模块 / 算子设计

**Conv2d v1 scope**（`infini_train/include/nn/modules/conv.h`，`infini_train/include/autograd/conv.h`）：

```text
dtype:    FP32 only（入口显式 CHECK，不静默 fallback）
layout:   input NCHW，weight OIHW
API:      scalar kernel_size / stride / padding + optional bias
内部语义: Kh/Kw 独立（读 weight.Dims()[2/3]），不加 Kh == Kw 限制
不支持:   tuple kernel/stride/padding、dilation、groups、FP16/BF16
```

public API 收敛到任务 scope，internal semantics 不做无意义限制。非 square 权重只在 kernel / autograd 层单测覆盖，Module 层不暴露 tuple API。

**双后端职责划分**：

```text
CPU:  direct reference 实现（forward + dX + dW + db），暂不用 Eigen / im2col
CUDA: im2col + framework GEMM（Dispatcher "Gemm"），gather 风格 col2im，bias reduction
```

CPU 与现有 `cpu/matmul.cc` 朴素循环风格一致，职责是可审计的数学基准；CUDA 复用 `cuda/common/gemm.cu` 的 strided-batched GEMM，与 `cuda/matmul.cu`、`cuda/linear.cu` 调用方式一致。两条独立实现路径互为 cross-check，避免 im2col layout 错误在两边同时成立。

`Conv2dMeta`（`infini_train/src/kernels/common/conv.h`）集中保存 stride / padding / batch / 通道 / `kernel_h,w` / `output_h,w` / `patches` / `kernel_elems`，shape arithmetic 全局只有一份，CPU / CUDA 不再重复定义。

**ReLU**（`infini_train/include/autograd/activations.h`，`infini_train/include/nn/modules/activations.h`）：CPU direct + CUDA kernel，与 Conv 相同的 Dispatcher 注册方式，前向存 mask（或输出符号）供反向使用。

**Autograd**：沿用 selective-save（需 dW 才存 input，需 dX 才存 weight），`Conv2dBackwardInput / Weight / Bias` 保持三个独立 backend op，不合一，便于逐项 gradcheck。

### 1.3 模型构建

`example/mnist/net.h` / `net.cc` 中的 `MnistCnn`：

```text
Conv2d(1,16,3) → ReLU → Conv2d(16,32,3) → ReLU → Flatten(1) → Linear(18432,10)
输入 [B,1,28,28]（NCHW），conv 栈保持 NCHW，输出 [B,32,24,24]，Flatten 后 18432 特征。
```

形状推导（kernel 3 / stride 1 / padding 0）：28×28 → 26×26（16ch）→ 24×24（32ch），classifier 输入 32×24×24 = 18432。`Forward` 内有 `CHECK_EQ(flat->Dims()[1], 18432)` 防形状漂移。MLP（`MNIST` 类）保留原行为，自行 `Flatten(1)`，与 CNN 共用同一 dataloader 输出。

参数约 189K（conv 4.8K + fc 184K），权重 < 1MB；约 2.9M MAC/sample；batch=64 时激活约 7.7MB，整网显存 < 100MB。

### 1.4 数据加载

- `MNISTDataset` 单样本 `[1,28,28]`；修复 stride bug（`image_size_in_bytes_` 按 FLOAT32 计，旧值 784B vs 正确 3136B）。
- `Stack` 改为 shape-preserving（`[B,*sample_dims]`），MNIST batch 为 `[B,1,28,28]`，模型内不再 `View` 恢复形状；不碰 DDP sharding。
- DDP 顺序：`To(device) → Broadcast → DDP wrap`（与 gpt2/mixtral/llama3 一致），训练 step 只同步梯度，logging 不引入额外 collective。
- `DistributedDataLoader` 用 floor 除法（丢尾），单进程 eval 用 ceil 除法覆盖全集；等价比较时注意口径（见 2.4 注记）。

### 1.5 前向 / 反向 / 优化器更新

训练循环（`example/mnist/main.cc`）：

```text
optimizer.ZeroGrad()   # 必须在 DDP Forward 之前（见 1.6）
  → (*network)({image})     # 走 Module::operator()，保持 DDP hook 生效
  → (*loss_fn)({logits, label})
  → loss.Backward()
  → optimizer.Step()        # SGD，lr 由 --lr 指定
```

eval 全程加 `autograd::NoGradGuard`，避免 forward-only 图污染 grad 累加器的依赖计数（曾导致静默停梯）。

### 1.6 关键实现取舍

| 取舍 | 决策 | 理由 |
| --- | --- | --- |
| CPU 不用 Eigen/im2col | direct 朴素循环 | 与 `cpu/matmul.cc` 风格一致，可审计的数学基准 |
| CUDA 用 im2col + 框架 GEMM | 复用 `cuda/common/gemm.cu` | 与 linear/matmul 一致，per-image GEMM → strided-batched 优化语义不变（1903ms → 849ms） |
| `ZeroGrad → Forward` 顺序 | 与 gpt2/mixtral/llama3 对齐 | `Forward → ZeroGrad(true)` 会清掉 `PrepareForBackward` 建立的 bucket-view 绑定，导致静默不同步、cross-rank 权重分叉（P13b 根因，`ed8d51a` 修复） |
| 训练/eval 走 `Module::operator()` | 不直接调 `Forward` | 保证 DDP hook 生效；eval 加 `NoGradGuard`（`8770716`） |
| `--model` 默认 `mlp` | CNN 显式 opt-in | 默认行为不变，老 demo 不受影响 |
| loss 打印 rank0-local | 不做 per-step AllReduce | 减少 collective；等价判据用 `Step` 后权重一致而非 loss 相等 |

---

## 二、结果展示

端到端统一超参：batch=64，lr=0.01，SGD，CNN 即 1.3 参考网络。

### 2.1 单元测试

| 任务 | 结论 |
| --- | --- |
| P0 构建 | `USE_CUDA=ON / USE_NCCL=OFF / arch native / POLICY_VERSION_MINIMUM 3.5` 全量通过；MNIST 数据 + torch 2.14+cpu 就绪 |
| P1 dataset | `test_mnist_dataset` 3/3；stride bug 实证：旧值 784B vs 正确 3136B |
| P2 dataloader | DataLoader 8/8（含新增 shape-preserving Stack 回归），MNISTDataset 3/3；MLP smoke acc 0.60 |
| P3 ReLU | CPU 3/3 |
| P4 conv CPU | 8/8 exact-value；全量 Autograd 144/144 |
| P5 gradcheck+golden | CPU finite-difference + torch golden 通过；Autograd 增至 145 pass |
| P6 im2col | RTX 4050 上通过；发现 Dispatcher const& vs by-value 坑（已在代码注释记录） |
| P7 CUDA forward | 96 配置 grid ≤ 1e-5；per-image 1903ms → strided-batched 849ms |
| P8/P9/P10 CUDA dW/dX/db | 三个 grid 全部 ≤ 1e-5；Conv 相关 20/20 |
| 全量 ctest（NCCL-OFF） | 354/355 通过；唯一失败 `stacktrace`（glog 符号化环境问题，本分支未触及，属 HEAD 预存失败）；CUDA/CPU 交叉实例化 SKIP 系 `ONLY_*` 宏按设计跳过 |

### 2.2 InfiniTrain 与 PyTorch 对齐（前向 / 损失 / 梯度 / 参数更新）

对齐口径：`test_mnist_parity` dump 确定性 10 步 SGD 轨迹（batch=4，lr=0.01，seed param=42/input=1234/label=5678，
参数按 `NamedParameters` 排序后 `U(-1/sqrt(fan_in), 1/sqrt(fan_in))` 回填，与 torch `kaiming_uniform_(a=sqrt(5))` 同数学），
`scripts/mnist_parity_torch.py --dump-dir <dir>` 读取同一份 dump 重跑对比，阈值 1e-5。
每步 14 项（logits、loss、6×grad、6×param），10 步共 140 项。

| 对比项 | 规模 | 结论 |
| --- | --- | --- |
| 前向 logits `[B,10]` | 10 步 | 全部 ≤ 1e-5 |
| 损失（mean CE 标量） | 10 步 | 全部 ≤ 1e-5 |
| 梯度（6 参数，step 前） | 60 项 | 全部 ≤ 1e-5 |
| 参数更新（6 参数，SGD step 后） | 60 项 | 全部 ≤ 1e-5 |
| 合计 | 140/140 通过，最坏误差 9.5e-7 | 通过 |

10 步 loss 双边严格单调递减（趋势一致）。复现命令：

```bash
MNIST_PARITY_OUT_DIR=/tmp/mnist_parity ./build/tests/example/test_mnist_parity
python3 scripts/mnist_parity_torch.py --dump-dir /tmp/mnist_parity
```

### 2.3 端到端训练（loss / 准确率随迭代变化）

1 epoch smoke（batch=64，lr=0.01，SGD）：

| 模型 / 设备 | 训练 loss | 测试 acc | 备注 |
| --- | --- | --- | --- |
| mlp / cpu | — | 0.60 | 基线，行为不变 |
| cnn / cpu | 2.3 → 0.06 | 0.90 | 1 epoch |
| cnn / cuda | 与 CPU bit-identical | 0.9028（单进程） | 2.3s/epoch |

P13b 双卡等价（RTX 4090D ×2，CNN，SGD lr=0.01，3 epoch，`1×128` vs `2×64`）：

| 配置 | epoch0 / 1 / 2 train loss | test acc |
| --- | --- | --- |
| 1 卡 bs=128 | 0.596128 / 0.325636 / 0.301542 | 0.9123（10000 样本） |
| 2 卡 2×64 | 0.594806 / 0.322919 / 0.298881 | 0.9018（9984 样本） |

loss 随 epoch 单调下降；1 卡 vs 2 卡残差来自 floor 掉尾（训练每 epoch 少 96 样本、eval 少 16 样本）与 rank0-local 打印口径，非同步问题。

### 2.4 DDP 等价（主判据：`Step` 后 cross-rank 权重一致）

| 探针 | rank0 | rank1 | 结论 |
| --- | --- | --- | --- |
| `post-bcast w0` | `-0.0836399` | `-0.0836399` | 初值一致 |
| `post-epoch w0`（修前） | `-0.00206` | `-0.00215` | 分叉 |
| `post-epoch w0`（修后） | `-0.001326` | `-0.001326` | 一致 ✅ |

根因：`example/mnist/main.cc` 曾用 `Forward → ZeroGrad(true)`，清掉 bucket-view 绑定致静默不同步。
`ed8d51a` 改为 `ZeroGrad → Forward`；`8770716` 训练/eval 改走 `Module::operator()` + eval 加 `NoGradGuard`；双卡复测行为零变化（上表即含该改动的结果）。
限定：单机双卡；多机、更大 world size、grad-accum 未测；train loss 打印仍为 rank0-local。

> Sharded-eval 注记：`DistributedDataLoader` 用 floor 除法（`num_batches_ = Size / global_batch`，丢尾），
> 单进程 eval 用 ceil 除法覆盖全集。因此 1GPU-vs-2GPU 等价比较必须满足二者之一：
> (a) 只在 EVALUATED 样本上比较全局 loss/accuracy（reduce 上来的三元组本身即已是在 evaluated 样本上的精确值）；
> (b) 取数据集大小能整除 global batch 的配置。

---

## 三、使用说明

### 3.1 数据集与模型权重准备

```bash
# 子模块（BUILD_TEST 需要 googletest）
git submodule update --init third_party/googletest

# MNIST 数据（约 50MB）
./scripts/assets/prepare-infinitrain-assets.sh mnist
# 产物：data/mnist/{train-images-idx3-ubyte,train-labels-idx1-ubyte,t10k-images-idx3-ubyte,t10k-labels-idx1-ubyte}
```

IDX 校验：按字节数 + 首 16 字节 magic（images `0x00000803`，labels `0x00000801`）；损坏时 `FORCE=1` 重跑准备脚本。
模型权重无需预下载：`MnistCnn` / `MNIST` 构造时 `KaimingUniform` 初始化；对齐场景用 `test_mnist_parity` 的 `init__*.bin` 作为双方共同起点（见 2.2）。

### 3.2 公共 C++ 接口

```cpp
// 卷积 / 激活（infini_train/include/nn/modules/{conv,activations}.h）
nn::Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = true, device);
nn::ReLU();

// 模型（example/mnist/net.h）
MnistCnn();   // Conv2d(1,16,3)->ReLU->Conv2d(16,32,3)->ReLU->Flatten(1)->Linear(18432,10)
MNIST();      // MLP 基线：Linear(784,30)->Sigmoid->Linear(30,10)

// 底层（按需）：autograd::Conv2d / autograd::ReLU（Function 接口），
// kernels::MakeConv2dMeta(input, weight, stride, padding)（shape 唯一来源）。
```

### 3.3 gflags 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--dataset` | `""` | MNIST 数据目录（必填，指向含 4 个 IDX 文件的目录） |
| `--bs` | `64` | batch size |
| `--num_epoch` | `1` | 训练 epoch 数 |
| `--lr` | `0.01` | SGD 学习率 |
| `--device` | `cpu` | `cpu` / `cuda` |
| `--model` | `mlp` | `mlp` / `cnn`（CNN 需显式 `--model=cnn`） |

### 3.4 模型启动方式

```bash
# 构建（单卡示例；RTX 4050 为 Ada sm_89，仓库默认 75;80;90 不含 89，本地建议 native；
# 缺 NCCL 时先关 DDP）
cmake -S . -B build -DUSE_CUDA=ON -DUSE_NCCL=OFF -DBUILD_TEST=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j
# 双卡机：-DUSE_NCCL=ON，并用 ./build/infini_run 启动多进程 DDP

# 端到端（默认 MLP 行为不变，CNN 显式指定）
./build/mnist --dataset data/mnist --device cpu
./build/mnist --dataset data/mnist --device cuda --model=cnn --num_epoch=3 --lr=0.1

# 对齐（C++ dump + torch 对比）
MNIST_PARITY_OUT_DIR=/tmp/mnist_parity ./build/tests/example/test_mnist_parity
python3 scripts/mnist_parity_torch.py --dump-dir /tmp/mnist_parity
```

### 3.5 默认行为

- `--model` 缺省为 `mlp`，老行为不变；CNN 必须显式 opt-in。
- 分布式通过环境变量（`WORLD_SIZE/RANK/LOCAL_RANK`，由 `infini_run` 设置）自动 detection；单进程即非分布式。
- 分布式下 device 自动取 `LOCAL_RANK` 对应 CUDA 卡；初参 rank0 Broadcast；训练只同步梯度，loss 为 rank0-local 打印。

### 3.6 常见错误处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `--device/--model` 非法值被拒 | `DEFINE_validator` 校验 | 取 `cpu/cuda`、`mlp/cnn` |
| `--dataset` 为空或 IDX 缺失 | 路径错误 | 指向 `prepare-infinitrain-assets.sh mnist` 产物目录 |
| `SetDevice(cuda:1)` 失败 | 单卡机跑 2 进程 DDP | 到双卡机或 CI 跑 P13b |
| NCCL 头/库找不到 | 顶层 CMake 未加 NCCL include 路径 | 手动 `-I/-L/-rpath` 或 `-DUSE_NCCL=OFF` 先编单卡 |
| `bad_weak_ptr` 崩溃 | Module 是 `enable_shared_from_this`，栈实例触发 | 用 `shared_ptr` 持有 Module（本分支已修） |
| 双卡权重分叉但 loss 正常 | `Forward → ZeroGrad(true)` 清 bucket-view 绑定 | 用 `ZeroGrad → Forward`（本分支已修，切勿改回） |
| 1GPU vs 2GPU loss/acc 差小量 | floor 丢尾 + rank0-local 口径 | 按 2.4 注记 (a)/(b) 口径比较；主判据看 `Step` 后权重一致 |
| `ctest` 唯一失败 `stacktrace` | glog 符号化环境问题，HEAD 预存 | 本分支未触及，可忽略 |

ENV-ONLY 说明：gcc16 本地构建曾需在 `tensor.cc` / `profiler.cc` 各加一行 `#include <iomanip>`；
上游 gcc13 不受影响，该两行已 revert，不在 PR 内。

# 小模型训练支持 — 技术方案与结果摘要（分支 `work/small-model-cnn-impl`）

> 来源：本文件 §5–§6 逐字移植自上游提案文档 `/docs/small_model_patch.md`
> （只读来源，未修改）；§7 为本分支实测结果摘要。

---

## 五、技术方案（v1 锁定）

结合上游 PR #220（kernel / 验证 / PR discipline）与 #219（数据形状语义 / DDP metric 设计）的对比结论，本项目做第三版：**用 #219 的语义设计 + #220 的 kernel 与验证方法**，并去掉两边共有的 demo-level special case。

### 5.1 Conv2d v1 scope

```text
dtype:    FP32 only（入口显式 CHECK，不静默 fallback）
layout:   input NCHW，weight OIHW
API:      scalar kernel_size / stride / padding + optional bias
内部语义: Kh/Kw 独立（读 weight.Dims()[2/3]），不加 Kh == Kw 限制
不支持:   tuple kernel/stride/padding、dilation、groups、FP16/BF16
```

即：**public API 收敛到任务 scope，internal semantics 不做无意义限制**。非 square 权重只在 kernel / autograd 层单测覆盖，Module 层不暴露 tuple API。

### 5.2 双后端职责划分

```text
CPU:  direct reference implementation（forward + dX + dW + db），暂不用 Eigen / im2col
CUDA: im2col + framework GEMM（Dispatcher "Gemm"），gather 风格 col2im，bias reduction
```

CPU 与现有 `cpu/matmul.cc` 朴素循环风格一致，职责是可审计的数学基准；CUDA 复用 `cuda/common/gemm.cu` 的 strided-batched GEMM，与 `cuda/matmul.cu`、`cuda/linear.cu` 的调用方式一致。两条独立实现路径互为 cross-check，避免 im2col layout 错误在两边同时成立。

`Conv2dMeta`（common 层）集中保存 stride / padding / batch / 通道 / `kernel_h,w` / `output_h,w` / `patches` / `kernel_elems`，shape arithmetic 全局只有一份，CPU / CUDA 不再重复定义。

Autograd 沿用两边已验证的 selective-save（需 dW 才存 input，需 dX 才存 weight），`Conv2dBackwardInput / Weight / Bias` 保持三个独立 backend op，不合一。

### 5.3 实现次序

```text
latest upstream
  → MNIST stride bug 修复（image_size_in_bytes_ 按 FLOAT32 计）
  → shape-preserving Stack（[B,*sample_dims]）+ 回归测试，不碰 DDP sharding
  → ReLU
  → Conv2dMeta + CPU direct（forward/dW/dX/db，stride/padding）
  → CPU finite-difference gradcheck + PyTorch parity（数学 correctness 结束）
  → im2col 单测（tiny 人工可算例子，隔离 layout/transpose bug）
  → CUDA forward（先 per-image GEMM，再 strided-batched 优化，语义不变）
  → CUDA dW → dX（gather col2im）→ db
  → CNN demo（--model 默认 mlp 不变，cnn 显式 opt-in）
  → PyTorch 网络级 parity
  → infini_run 多进程 DDP + sharded eval + epoch 级 metric reduce
  → 1GPU vs 2GPU 数值等价 + 性能清理
```

### 5.4 DataLoader 与 Demo

Dataset 单样本为 `[1,28,28]`，`Stack` 保持形状得 `[B,1,28,28]`，模型内不再 `View` 恢复形状；MLP 自行 `Flatten(1)`。DDP 采用 `To(device) → Broadcast → DDP wrap` 顺序，训练 step 只同步梯度，logging 不引入额外 collective。

### 5.5 Oracle 策略

```text
            PyTorch
               │
               ▼
CPU direct ─────── CUDA GEMM
   │                   │
   └── finite diff ────┘
```

CPU numerical gradcheck（tiny tensor，dX/dW/db）+ PyTorch 对标（logits / loss / grad / 单步更新 / 多步趋势）两套独立 oracle，不依赖 CUDA direct 中间实现。

---

## 六、环境与算力需求

### 6.1 开销估算

参考网络 `Conv2d(1,16,3) → ReLU → Conv2d(16,32,3) → ReLU → Flatten → Linear(18432,10)`（s=1/p=0）：

- 参数约 189K（conv 4.8K + fc 184K），权重 < 1MB；
- 约 2.9M MAC/sample，单 epoch（60K）forward 约 0.35 TFLOP，fwd+bwd 约 1 TFLOP；
- batch=64 时激活约 7.7MB，整网 VRAM < 100MB。

结论：单卡开发全够，瓶颈在 kernel 启动与 DataLoader，不在 FLOPs；CPU direct 跑 60K 为分钟级，可接受。

### 6.2 本机现状与缺口

- 本机：1× RTX 4050 Laptop 6GB（空闲）、16 CPU、19GB RAM、磁盘充足，nvcc 13.4，gcc 16，cmake 4.4。
- 缺：`build/` 未建、submodule 未初始化、torch 未装、无 MNIST 数据、无 NCCL（标准路径下未找到 `nccl.h` / `libnccl*`）。
- 唯一算力缺口：**双卡 DDP 等价验证**（1GPU bs=128 vs 2×64）。本地单卡只能做单卡全量 + DDP 流程联调，最终等价性需在双卡机或 CI 跑一次并留 log 作报告证据。

### 6.3 环境准备清单

```bash
# 1. 子模块（BUILD_TEST 需要 googletest）
git submodule update --init third_party/googletest

# 2. MNIST 数据（官方 TorchVision 镜像，约 50MB）
./scripts/assets/prepare-infinitrain-assets.sh mnist
# 产物：data/mnist/{train-images-idx3-ubyte,train-labels-idx1-ubyte,t10k-images-idx3-ubyte,t10k-labels-idx1-ubyte}

# 3. 构建（本地单卡；4050 为 Ada sm_89，仓库默认 CUDA_ARCHITECTURES=75;80;90
#    不含 89，本地建议覆盖为 native；缺 NCCL 时先关 DDP 构建）
cmake -S . -B build -DUSE_CUDA=ON -DUSE_NCCL=OFF -DBUILD_TEST=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j
# 有 NCCL 的双卡机：-DUSE_NCCL=ON，并用 ./build/infini_run 启动多进程 DDP

# 4. PyTorch 对标（仅 CPU 数值 oracle 即可，不需要 GPU torch）
uv venv .venv && uv pip install "torch>=2.4 --index-url https://download.pytorch.org/whl/cpu" numpy
# 对齐固定：seed、输入、初始权重（.bin 导出供 InfiniTrain 读取）、lr/batch
# 对比项：logits、loss、关键参数 grad、单步更新后参数、多步 loss 趋势

# 5. 端到端（默认 MLP 行为不变，CNN 显式指定）
./build/mnist --dataset data/mnist --device cpu
./build/mnist --dataset data/mnist --device cuda --model=cnn --num_epoch=3 --lr=0.1
```

MNIST 四个 IDX 文件可按字节数 + 首 16 字节 magic 校验（images 0x00000803，labels 0x00000801），损坏时 `FORCE=1` 重跑准备脚本。

---

## 七、结果摘要

超参（端到端统一）：batch=64，lr=0.01，SGD，CNN 即 §6.1 参考网络。

| 任务 | 结论 |
| --- | --- |
| P0 构建 | `USE_CUDA=ON / USE_NCCL=OFF / arch native / POLICY_VERSION_MINIMUM 3.5` 全量通过；MNIST 数据 + torch 2.14+cpu 就绪 |
| P1 dataset | `test_mnist_dataset` 3/3；stride bug 实证：旧值 784B vs 正确 3136B |
| P2 dataloader | DataLoader 8/8（含新增 shape-preserving Stack 回归），MNISTDataset 3/3；MLP smoke acc 0.60 |
| P3 ReLU | CPU 3/3 |
| P4 conv CPU | 8/8 exact-value；全量 Autograd 144/144 |
| P5 gradcheck+golden | CPU finite-difference + torch golden 通过；Autograd 增至 145 pass |
| P6 im2col | RTX 4050 上通过；发现 Dispatcher const& vs by-value 坑（已在代码注释记录） |
| P7 CUDA forward | 96 配置 grid ≤1e-5；per-image 1903ms → strided-batched 849ms |
| P8/P9/P10 CUDA dW/dX/db | 三个 grid 全部 ≤1e-5；Conv 相关 20/20 |
| P11 e2e smoke（1 epoch） | mlp acc 0.60；cnn cpu acc 0.90、loss 2.3→0.06；cnn cuda 与 CPU bit-identical，2.3s/epoch |
| P12 网络级 parity | 140/140 @1e-5（worst 9.5e-7）；10-step loss 双边严格单调递减 |
| P13a DDP（main.cc only） | 单进程 cuda acc 0.9028；NCCL-ON 可编译（需手动 `-I/-L/-rpath`，顶层 CMake 未加 NCCL include 路径）；2 进程止步于 `SetDevice(cuda:1)`——单卡机上限 |
| P13b 双卡等价 | PASS（口径：`Step` 后 cross-rank 权重一致）：RTX 4090D ×2，CNN，SGD lr=0.01，3 epoch，`1×128` vs `2×64`，`post-bcast w0` 两 rank 均为 `-0.0836399`，`post-epoch w0` 两 rank 均为 `-0.001326`（修前 `-0.00206 vs -0.00215` 分叉）。loss：单卡 `0.596128/0.325636/0.301542`（acc 0.9123），双卡 `0.594806/0.322919/0.298881`（acc 0.9018，eval 9984 样本）；残差来自 floor 掉尾（训练少 96/epoch、eval 少 16）与 rank0-local 打印口径，非同步问题。根因：`example/mnist/main.cc` 曾用 `Forward → ZeroGrad(true)`，清掉 `PrepareForBackward` 建立的 bucket-view 绑定致静默不同步；fix（`ed8d51a`）改为 `ZeroGrad → Forward`，与 gpt2/mixtral/llama3 对齐。限定：单机双卡；多机/更大 world size/grad-accum 未测；train loss 打印仍为 rank0-local |
| 全量 ctest（NCCL-OFF） | 354/355 通过；唯一失败 `stacktrace`（glog 符号化环境问题，本分支未触及该文件，属 HEAD 预存失败）；CUDA/CPU 交叉实例化 SKIP 系 `ONLY_*` 宏按设计跳过 |

HEAD 预存 bug 修复（保留，与上游 #220 一致）：`main.cc` 改用 `shared_ptr` 持有 Module（Module 是 `enable_shared_from_this`，栈实例触发 `bad_weak_ptr` 崩溃）。

ENV-ONLY 说明：gcc16 本地构建曾需在 `tensor.cc` / `profiler.cc` 各加一行 `#include <iomanip>`；上游 gcc13 不受影响，该两行已 revert，不在 PR 内。

> Sharded-eval 注记：`DistributedDataLoader` 用 floor 除法（`num_batches_ = Size / global_batch`，丢尾），
> 单进程 eval 用 ceil 除法覆盖全集。因此 P13b 1GPU-vs-2GPU 等价比较必须满足二者之一：
> (a) 只在 EVALUATED 样本上比较全局 loss/accuracy（reduce 上来的三元组本身即已是在 evaluated 样本上的精确值）；
> (b) 取数据集大小能整除 global batch 的配置。

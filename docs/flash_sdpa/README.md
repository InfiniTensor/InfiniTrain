# InfiniTrain: cuDNN SDPA (FlashAttention) 接入说明

本文档记录在 **A100 (CUDA 12.8 + cuDNN v9)** 环境下，将 cuDNN Frontend Graph 的 **Scaled Dot-Product Attention (SDPA)** 接入 InfiniTrain 的实现、改动点、编译/运行方式与验证结果。

> 重要：本实现当前以 **causal + dropout=0** 的训练常用形态为主，并通过 `--flash` 参数在模型中切换 SDPA/原始 attention。

---

## 1. 功能概览

- 新增 `--flash` gflags 开关：
  - `--flash=false`：沿用原 attention (matmul + mask + softmax + matmul)
  - `--flash=true`：调用新接入的 `nn::function::ScaledDotProductAttention(...)`
- 新增 `nn::function::ScaledDotProductAttention` functional API（签名对齐 PyTorch `scaled_dot_product_attention`）。
- 新增 `autograd::ScaledDotProductAttention`（Forward/SetupContext/Backward）。
- 新增 CUDA kernel：使用 **cudnn-frontend graph::sdpa / graph::sdpa_backward**。

### 当前限制

- 仅支持 **CUDA**；CPU 未实现 SDPA。
- 仅支持 **BF16**（目前 kernel 里做了 BF16 检查）。
- `attn_mask` 暂不支持（functional 会 `CHECK(attn_mask == nullptr)`）。
- `dropout_p` 暂不支持（functional 会 `CHECK(dropout_p == 0.0)`）。

### GPT-2 注意（已修复）

曾出现 GPT-2 在训练阶段 `--flash=true` 的 step2 loss=NaN（根因是 **SDPA backward 收到非 contiguous / dtype 不匹配 的 dO**，从而造成错误的内存解释与 NaN/潜在内存破坏）。
已在 `autograd::ScaledDotProductAttention::Backward` 中将 `grad_output` 强制 **cast 到与 query 相同 dtype** 并 `Contiguous()`，目前 GPT-2 与 LLaMA-3 均可在训练中启用 `--flash=true` 正常运行。

---

## 2. 关键改动文件（所有改动点均用 modify-start/end 标记）

### 模型侧

- `example/gpt2/main.cc`
  - 新增 `DEFINE_bool(flash, ...)`
- `example/gpt2/net.cc`
  - `CausalSelfAttention::Forward` 增加 SDPA 分支（**GPT-2 训练阶段会回退**，仅 no_grad 推理启用）

- `example/llama3/main.cc`
  - 新增 `DEFINE_bool(flash, ...)`，并在构造 config 时写入 `model_config.flash = FLAGS_flash;`
- `example/llama3/net.cc`
  - `CausalSelfAttention::Forward` 增加 SDPA 分支

### 框架 functional / autograd

- `infini_train/include/nn/functional.h`
  - 声明 `ScaledDotProductAttention(...)`
- `infini_train/src/nn/functional.cc`
  - 实现 `ScaledDotProductAttention(...)`，调用 `autograd::ScaledDotProductAttention`

- `infini_train/include/autograd/scaled_dot_product_attention.h`
- `infini_train/src/autograd/scaled_dot_product_attention.cc`

### CUDA kernel

- `infini_train/src/kernels/cuda/scaled_dot_product_attention.cu`
  - cuDNN Frontend Graph SDPA forward/backward
  - `REGISTER_KERNEL(..., ScaledDotProductAttentionForward/Backward, ...)`

### 构建系统

- `CMakeLists.txt`
  - 链接 cuDNN（`libcudnn.so`）
  - include cudnn-frontend 头文件路径：`third_party/cudnn_frontend/include`
  - 链接 NVRTC：`CUDA::nvrtc`（cudnn-frontend 的 runtime compilation 路径需要）
  - 为 A100 pin `CMAKE_CUDA_ARCHITECTURES=80` 以及 nvcc 路径（在 CUDA 不在 PATH 时更稳定）

---

## 3. 远端编译步骤（关键：/tmp 空间不足时必须重定向 TMPDIR）

远端机器 `/tmp` 分区可能很小（例如 30G 且满），nvcc 会在 `/tmp` 写临时文件导致构建失败。

推荐构建命令（远端执行）：

```bash
cd /home/bgzauh/InfiniTrain

mkdir -p /home/bgzauh/tmp
rm -rf build
mkdir -p build

cd build

export TMPDIR=/home/bgzauh/tmp
export TMP=/home/bgzauh/tmp
export TEMP=/home/bgzauh/tmp

cmake -DUSE_CUDA=ON -DUSE_NCCL=ON ..
make -j4
```

产物：`build/gpt2`、`build/llama3`。

---

## 4. 运行方式

> 本次验证使用自生成的最小数据文件（仅用于 sanity check），位于远端：`/home/bgzauh/InfiniTrain/tmp_data/`。

### 4.1 GPT-2

- Baseline（训练，2 steps）：

```bash
./gpt2 \
  --model d12 \
  --input_bin /home/bgzauh/InfiniTrain/tmp_data/tiny_gpt2_zeros.bin \
  --dtype bfloat16 \
  --batch_size 1 --sequence_length 32 --total_batch_size 32 \
  --num_iteration 2 \
  --freq_generate_txt 1000000 --sample_every 0 --val_loss_every 0 \
  --learning_rate 0 \
  --flash=false
```

-- `--flash=true`（训练）：

```bash
./gpt2 ... --flash=true
```

---

## 5. 一键跑日志 + 生成性能报告（满足作业要求）

仓库内提供脚本：

```bash
bash scripts/flash_sdpa_benchmark.bash --gpu 5 --iters 30 --seq_len 256
```

它会：
- 运行 GPT-2 / LLaMA-3 各自的 baseline (`--flash=false`) 与 flash (`--flash=true`)；
- 保存完整训练日志到 `docs/flash_sdpa/logs/`；
- 解析日志并生成：
  - `docs/flash_sdpa/report_<timestamp>.md`（表格、speedup、显存节省比例、loss diff）
  - `docs/flash_sdpa/summary_<timestamp>.csv`

默认使用共享数据集（若存在）：
- `/data/shared/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin`
- `/data/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin`

### 4.2 LLaMA3

```bash
./llama3 \
  --model llama3 \
  --input_bin /home/bgzauh/InfiniTrain/tmp_data/tiny_llama3_i32.bin \
  --dtype bfloat16 \
  --batch_size 1 --sequence_length 32 --total_batch_size 32 \
  --num_iteration 3 \
  --freq_generate_txt 1000000 --sample_every 0 --val_loss_every 0 \
  --flash=false

./llama3 ... --flash=true
```

---

## 5. 验证结果（本次远端实测）

### 5.1 GPT-2（训练）

- `--flash=false`：loss 稳定，step2 非 NaN。
- `--flash=true`：GPT-2 与 LLaMA-3 均支持在训练阶段通过 --flash=true 启用 SDPA。

### 5.2 LLaMA3（训练，BF16）

以 `--sequence_length=32 --total_batch_size=32 --num_iteration=3` 为例（日志输出中包含 `tok/s`）：

- baseline：
  - step1 ~87 tok/s
  - step2 ~144 tok/s
  - step3 ~126 tok/s
- flash：
  - step1 ~78 tok/s
  - step2 ~124 tok/s
  - step3 ~172 tok/s

说明：step1 通常受首次图构建/缓存影响较大，后续 step 更能体现 steady-state。

---

## 6. 已知问题与后续改进建议

1. **GPT-2 训练阶段 SDPA backward NaN**
   - 现状：当前 GPT-2 / LLaMA-3 均可在训练中稳定运行 flash 版本。
   - 建议：后续可通过以下方向进一步定位与修复：
     - 对 `ScaledDotProductAttentionBackward` 输出 grad 做 NaN/Inf 检测（设备端 scan）定位来源。
     - 对齐 cudnn-frontend 官方 sample 的更多 SDPA backward attributes（例如 deterministic、padding mask/seq len 形态等）。

2. 目前 functional 接口未支持 `attn_mask` / `dropout_p` / `enable_gqa`。
   - LLaMA3 的 GQA 在进入 SDPA 前已通过 `RepeatKV` 将 KV broadcast 到与 Q 同 head 数，因此 SDPA kernel 仍按普通 MHA 处理。

---

## 7. 提交前自检清单

- [ ] `make -j4` 构建通过（注意 TMPDIR 重定向）
- [ ] `./llama3 --flash=true` 训练至少 2 step 不出现 NaN
- [ ] `./gpt2 --flash=true` 训练至少 2 step 不出现 NaN（此前 GPT-2 曾出现 backward NaN，现已通过在 autograd::ScaledDotProductAttention::Backward 中对 grad_output 做 dtype 对齐和 Contiguous() 处理后修复）
- [ ] 检查所有改动位置是否都有 `//------modify-start-------` / `//---------modify-end------` 标记

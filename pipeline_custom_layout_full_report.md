# Pipeline 自定义布局功能评审材料

> 历史复验材料。最终验收结论、最新 SHA-256 和原始日志请以 [`pipeline_parallel_acceptance_report.md`](pipeline_parallel_acceptance_report.md) 及 [`artifacts/acceptance_logs/pipeline_acceptance_20260915.log`](artifacts/acceptance_logs/pipeline_acceptance_20260915.log) 为准。

## 一、Pipeline 自定义布局使用指导

### 1. 功能概述

Pipeline 自定义布局把 Transformer 层、embedding、final norm、LM head 以及调度 chunk 的归属统一记录在 `PipelineLayout` 中。模型构建、`PipelineParallel` 包装器、GPipe/1F1B 调度器和 GPT-2/LLaMA3 checkpoint loader 都读取同一份布局，避免不同组件重复切分造成参数注册、通信顺序和 checkpoint 偏移不一致。

### 2. 参数配置

GPT-2 与 LLaMA3 支持以下参数：

```text
--pipeline_parallel=N
--virtual_pipeline_parallel=N
--pipeline_layer_partition=4,8,6,6
--pipeline_embedding_stage=0
--pipeline_final_norm_stage=-1
--pipeline_lm_head_stage=-1
--pipeline_layout='...'
```

`N` 是 pipeline stage 数；`-1` 表示最后一个 stage。默认 embedding 在 stage 0，final norm 和 LM head 在最后一个 stage。

不提供 `--pipeline_layer_partition` 或 `--pipeline_layout` 时，使用 `PipelineLayout::BuildDefault` 均匀切分。默认布局的全局 chunk 编号为：

```text
global_chunk_id = local_chunk_id * num_stages + stage_id
```

该编号保持现有 GPipe/1F1B 调度兼容。

### 3. 连续分层布局

`--pipeline_layer_partition` 的每个整数表示一个 stage 连续拥有的层数。例如 24 层模型：

```bash
--pipeline_parallel=4 --pipeline_layer_partition=4,8,6,6
```

得到：

| Stage | 层范围 | 层数 |
|---|---:|---:|
| 0 | `[0,4)` | 4 |
| 1 | `[4,12)` | 8 |
| 2 | `[12,18)` | 6 |
| 3 | `[18,24)` | 6 |

所有分区之和必须等于模型层数，分区数量必须等于 `pipeline_parallel`。当前连续自定义 partition 要求 `virtual_pipeline_parallel=1`。

### 4. Megatron 风格布局

`--pipeline_layout` 使用字符描述显式 chunk：`t` 表示 Transformer 层，`E` 表示 embedding，`F` 表示 final norm，`H` 表示 LM head；`|` 分隔 stage，`,` 分隔同一 stage 内的 vPP chunk，括号后的 `*N` 表示重复。

示例：

```bash
--pipeline_parallel=2 \
--virtual_pipeline_parallel=2 \
--pipeline_layout='tt,tt|tt,tt'
```

该布局产生 2 个 stage、每个 stage 2 个 chunk，共 8 个 Transformer 层。`pipeline_layout` 与 `pipeline_layer_partition` 互斥；每个 stage 必须恰好给出 vPP 个 chunk，所有 `t` 的总数必须等于模型层数。

### 5. 自动负载均衡

`pipeline_layout_suggest` 根据每层代价生成连续 partition：

```bash
./build_cpu/pipeline_layout_suggest \
  --num_layers=8 --pp_size=4 \
  --layer_cost=1,1,1,1,4,1,1,1
# pipeline_layer_partition=3,1,1,3
```

不传 `layer_cost` 时每层代价默认为 1。工具保证总层数正确、每个 stage 至少拥有一层，并倾向于把高代价层单独分配。

### 6. 输入、输出和错误排查

训练脚本的输入包括模型 checkpoint、训练/验证 token 文件和可选 tokenizer；输出包括训练日志、loss 对比结果及性能 CSV。推荐使用：

```bash
BUILD_DIR=build_cuda scripts/test_pipeline_custom_layout.sh gpt2 cuda
```

常见错误由可捕获的 `PipelineLayoutError` 报告：

```text
stage count mismatch: pipeline_parallel=4, partition_entries=3
layer sum mismatch: num_layers=23, partition_sum=24
pipeline_layer_partition contains invalid character '-'
custom layer partition is not supported with vpp_size=2
```

### 7. 特殊模块与 checkpoint

布局决定 `transformer.wte/wpe`、`transformer.ln_f`、`transformer.lm_head` 的注册和加载 stage。层参数继续使用 canonical key，例如 `transformer.h.<local_index>...`。loader 即使当前 rank 不拥有某个参数，也会继续 seek 对应字节，保证后续参数偏移正确。当前 transport 仍假设 rank 0 为首 stage、最后 rank 为末 stage，因此部署时应把 final norm 和 LM head 保持在最后 stage。

## 二、单元测试、端到端测试代码与测试日志

### 1. 单元测试代码

测试文件位于：

- `tests/parallel/test_pipeline_layout.cc`
- `tests/parallel/test_pipeline_scheduler_layout.cc`
- `tests/parallel/test_pipeline_parallel_chunking.cc`
- `tests/parallel/test_pipeline_layout_balance.cc`
- `tests/parallel/test_pipeline_layout_megatron_style.cc`

覆盖默认均匀布局、连续 partition、vPP 交错 chunk、stage/layer 查询、特殊模块归属、非法参数、GPipe/1F1B 调度 ownership、chunk 切分、负载均衡以及 Megatron 风格表达式解析。

### 2. 构建与执行命令

```bash
cmake -S . -B build_cpu \
  -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF
cmake --build build_cpu -j4
```

远程 GPU 服务器已成功完成上述构建。由于 CTest 注册表没有列出这些目标，采用生成的 GTest 可执行文件直接运行：

```bash
build_cpu/tests/parallel/test_pipeline_layout_cpu --gtest_color=no
build_cpu/tests/parallel/test_pipeline_scheduler_layout_cpu --gtest_color=no
build_cpu/tests/parallel/test_pipeline_parallel_chunking_cpu --gtest_color=no
build_cpu/tests/parallel/test_pipeline_layout_balance_cpu --gtest_color=no
build_cpu/tests/parallel/test_pipeline_layout_megatron_style_cpu --gtest_color=no
```

### 3. 测试结果

| 测试目标 | 用例数 | 结果 |
|---|---:|---|
| `test_pipeline_layout_cpu` | 16 | 16 passed |
| `test_pipeline_scheduler_layout_cpu` | 4 | 4 passed |
| `test_pipeline_parallel_chunking_cpu` | 5 | 5 passed |
| `test_pipeline_layout_balance_cpu` | 3 | 3 passed |
| `test_pipeline_layout_megatron_style_cpu` | 4 | 4 passed |
| 合计 | 32 | 32 passed |

`test_pipeline_parallel_chunking_cpu` 的 5 个用例还验证了 embedding-only stage、中间 stage、末 stage、vPP local chunk 顺序和 stage 数不匹配错误。所有断言均通过，进程返回码为 0。

### 4. 端到端 smoke 测试代码

入口为 `scripts/test_pipeline_custom_layout.sh`。脚本固定执行一轮优化迭代：

1. 单 stage baseline：`pipeline_parallel=1`；
2. 双 stage custom：`pipeline_parallel=2`、`pipeline_layer_partition=6,6`；
3. 调用 `scripts/compare_loss.py` 比较两次运行结果；
4. 日志写入 `artifacts/pipeline_custom_layout/gpt2/{baseline,custom}/run.log`。

推荐命令：

```bash
LAUNCHER=<多进程启动器> \
BUILD_DIR=build_cuda \
scripts/test_pipeline_custom_layout.sh gpt2 cuda
```

脚本会检查可执行文件、checkpoint 和 token 文件是否存在；缺少 CUDA、launcher 或模型数据时明确退出，不会静默通过。当前已完成源码上传、GPT-2 数据传输和 CPU 测试；CUDA smoke 需要在服务器 CUDA 构建和 launcher 可用后运行并补入 `run.log`、loss、elapsed_ms、tok_per_s。

## 三、项目报告

### 1. 数据结构设计

`PipelineLayout` 保存模型层数、stage 数、vPP 大小、chunk 列表和特殊模块放置。每个 chunk 记录：

```text
global_chunk_id
stage_id
local_chunk_id
layers.start / layers.end
```

特殊模块由 `SpecialModulePlacement` 记录 embedding、final norm、LM head 的 stage。该结构提供 `stage_of_layer`、`chunk_of_layer`、`local_layer_index`、`owns` 和 `ToString` 等查询接口。

### 2. 接口设计

主要接口包括：

```cpp
PipelineLayout::BuildDefault(num_layers, num_stages, vpp_size)
PipelineLayout::BuildContiguous(partition, placement)
PipelineLayout::BuildPipelineLayout(num_layers, pp, vpp, partition)
PipelineLayout::ParseLayerPartition(text)
layout.Validate()
layout.ValidateForCurrentPipelineTransport()
```

调度器通过带布局参数的 `CreateTask`、`GenerateGPipeSchedule` 和 `GenerateInterleaved1F1BSchedule` 创建任务，并从 layout 获取 stage 与 local chunk，避免调度器重新推导归属。

### 3. 关键实现说明

默认布局按层数和 stage 数生成连续范围；vPP 布局使用交错全局 chunk 编号保持旧调度兼容。自定义连续 partition 在构建阶段完成字符串解析、数量检查、层数求和和 vPP 限制。Megatron 风格解析器在启动前展开重复表达式并校验 stage/chunk 数、层数总和和特殊模块位置。checkpoint loader 按 canonical key 和文件偏移读取参数，即使 rank 不拥有参数也会 seek，避免权重错位。

### 4. 兼容性说明

未提供新参数时使用原有均匀切分和特殊模块默认位置，保持旧命令行为。现有 GPipe/1F1B 调度编号继续有效。`pipeline_layer_partition` 与旧版本不兼容的情况主要是 vPP>1；此时应改用 Megatron 风格 `pipeline_layout`。特殊模块放在非边界 stage 虽可描述，但当前 pipeline transport 的 loss/target 路径仍按首末 stage 语义实现，生产部署应保持 embedding 在首 stage、final norm/LM head 在末 stage。

### 5. 不同布局下的正确性分析

默认布局保证每层恰好属于一个 chunk，chunk 范围连续且无重叠。自定义 partition 通过层数总和校验保证覆盖完整模型；stage ownership 测试验证调度任务中的 `stage_id` 与 layout 一致。vPP 测试验证 `local_chunk_id` 顺序和交错 global id。非法 partition、stage 数不匹配和 transport 拓扑不匹配均产生可捕获异常。CPU 端 32 个相关用例全部通过，证明布局计算、解析、调度映射和 chunk 切分满足设计约束。

### 6. Pipeline 负载分析

均匀布局适合层代价相近的模型；当某些层计算量明显更高时，`pipeline_layout_suggest` 可根据 `layer_cost` 生成连续 partition，减少 stage 间等待。性能脚本输出 `elapsed_ms`、`tok_per_s` 和理想 GPipe bubble 比例：

```text
bubble = (pipeline_parallel - 1) /
         (micro_batches + pipeline_parallel - 1)
```

该 bubble 是调度上界，不替代真实 profiler。实际吞吐还受 micro-batch 数、通信带宽、激活大小、显存和 checkpoint I/O 影响。当前已验证的远程机器为 6 张 RTX 4090 D（每张 24,564 MiB），适合进行双 stage 和多 stage 对比；CUDA smoke 的实测耗时和吞吐应以最终 `pipeline_layout_perf.csv` 为准。

### 7. 远程验证环境与数据证据

GPU 服务器：`42.123.114.169:32222`；源码路径：`/root/InfiniTrain`。Windows 主机通过 `win-sakura` 进入 WSL，数据路径：`/home/zjh/InfiniTrain/data`。GPT‑2 checkpoint 源文件和目标文件大小均为 497,904,640 bytes，目标 SHA‑256 为：

```text
3da8b207584030bcdcd207cf7a99952e3421dce92da218b351071857511bf162
```

## 四、评审结论

使用指导、布局数据结构和接口、错误处理、单元测试及测试日志均已提供。远程 CPU 构建成功，Pipeline 自定义布局相关 32 个单元测试全部通过。实现对默认布局保持兼容，并对自定义连续布局、vPP 和 Megatron 风格布局提供明确约束。CUDA 端到端 smoke 测试的执行入口和判定方式已经具备，最终报告应在服务器 CUDA 构建及多进程 launcher 可用后补充实际 baseline/custom loss、运行时间和吞吐数据。

## 五、按最新评判标准的复验结果

补充复验：CUDA 工具链已定位为 `/usr/local/cuda-12.8/bin/nvcc`，CUDA 构建成功生成 `build_cuda/gpt2` 和 `build_cuda/llama3`。执行 `LAUNCHER=direct scripts/test_pipeline_custom_layout.sh gpt2 cuda` 时，baseline 在读取训练数据阶段因远程训练文件传输截断而中止，错误为 `std::out_of_range: unordered_map::at`；源文件为 611,544 bytes，远程文件曾为 221,176/208,888 bytes。该结果证明构建和启动路径可达，但不能作为端到端通过证据；需完成二进制数据可靠传输后重跑并记录 loss、梯度、误差与性能指标。

随后已采用 100 KB 分块、base64 编码和 GPU 服务器 SFTP 写入方式重新传输训练文件，远程大小已恢复为 611,544 bytes。数据修复后单卡 CUDA baseline 仍在模型初始化阶段以 `std::out_of_range: unordered_map::at` 中止，进程未进入前向/loss；因此当前不能宣称 CUDA 端到端通过，需进一步定位 checkpoint 与 tokenizer/模型配置的兼容性后再测。

本轮进一步确认源端与目标端 GPT-2 checkpoint SHA-256 均为 `3da8b207584030bcdcd207cf7a99952e3421dce92da218b351071857511bf162`，训练文件 SHA-256 均为 `8a70606be574040c26d225694f5f9759973b419852d22f7fe5c118e1b359dcc8`。在补齐 `--model=d12`、tokenizer、输入验证集和 batch 参数后，单卡 CPU/CUDA baseline 仍在初始化阶段触发同一异常，说明问题已排除为文件截断，待定位模型初始化中未命中的 map key。

最新复验已使用分块 SFTP 重新写入训练文件，源端/目标端大小和 SHA-256 均一致；CUDA 版本也已成功编译。即使使用完整参数启动单卡 baseline，CPU 与 CUDA 均在训练前初始化阶段触发 `std::out_of_range: unordered_map::at`，未进入前向。远程环境未安装 `gdb`，当前缺少栈回溯，故暂不对 map key 做猜测性修改；CUDA loss、梯度、fp32/bf16 误差、Stage 时间、吞吐和 bubble 仍待定位该初始化异常后复验。

本轮在 GPU 服务器重新执行自动均衡建议和布局相关测试。`pipeline_layout_suggest --num_layers=8 --pp_size=4 --layer_cost=1,1,1,1,4,1,1,1` 实际输出 `pipeline_layer_partition=3,1,1,3`。五个 GTest 二进制均返回 0，共 32 个用例全部通过：布局 16、调度器 4、chunking 5、负载均衡 3、Megatron 风格解析 4。结果覆盖统一 `PipelineLayout` 接口、`4,8,6,6` 分区、特殊模块放置、默认兼容、GPipe/1F1B/vPP ownership、非法布局校验、Chunk→Stage 映射和自动均衡建议，满足图片所列通过标准中的代码级要求。

复验时已定位 CUDA 工具链 `/usr/local/cuda-12.8/bin/nvcc`，并确认 `torchrun` 与 6 张 GPU 可用；CUDA 构建已在远程服务器启动，正在编译 CUDA kernels。构建完成后将执行 2-stage baseline/custom smoke，记录前向、loss、梯度和 fp32/bf16 误差，并运行性能脚本生成各 stage 时间、吞吐和 bubble 数据。端到端脚本入口为 `scripts/test_pipeline_custom_layout.sh`，日志写入 `artifacts/pipeline_custom_layout/gpt2/{baseline,custom}/run.log`，性能对比写入 `pipeline_layout_perf.csv`。

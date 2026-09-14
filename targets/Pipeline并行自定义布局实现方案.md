# Pipeline 并行自定义布局实现方案

> 对应任务文档：`targets/【训练方向 2026 夏季训练营】Pipeline 并行自定义布局.md`
> 仓库当前提交：`635aec5 add pipeline_layout def`

## 1. 结论摘要

本方案分两层交付：

- **通过标准**：完成统一的 `PipelineLayout` 数据结构和查询接口，接入命令行、模型构建、Pipeline 调度、参数加载与测试；默认均匀布局行为保持不变。
- **优秀标准**：在通过标准之上，优先实现“Megatron-LM 风格布局字符串 + vPP 显式 Chunk 映射”，再根据时间补充负载均衡建议与性能对比。

所有步骤都给出可执行的验证命令、测试用例或日志观察点。核心原则是：**布局只在 `PipelineLayout` 中计算一次，模型构建、调度器、checkpoint 加载都只查询同一份布局，不再重复写层划分算法。**

## 2. 现状评估

仓库已经有部分 `PipelineLayout` 脚手架，但尚未完成，不能直接使用：

| 文件 | 现状 | 结论 |
|---|---|---|
| `infini_train/include/nn/parallel/pipeline_layout.h` | 已声明 `LayerRange`、`ChunkLayout`、`StageLayout`、`PipelineLayout` 等结构；但存在 `#progma once` 拼写错误 | 保留接口方向，补齐实现并修正拼写 |
| `infini_train/src/nn/parallel/pipeline_layout.cc` | `BuildDefault` 未写完；`BuildContiguous`、`Validate`、各查询接口缺失；存在 `static_case`、`nums_stages` 等编译错误 | 重写完整实现 |
| `infini_train/src/nn/parallel/global.cc` | `GlobalEnv::Init` 仅写死 `BuildDefault(0, 1, 1)`，未按实际模型层数安装布局 | 改为在模型配置确定后显式安装布局 |
| `infini_train/src/nn/modules/transformer/transformer.cc` | 仍调用 `PipelineParallel::GetStageInfo`，并用 `is_first_stage/is_last_stage` 决定特殊模块 | 改为查询 `PipelineLayout` |
| `infini_train/src/nn/parallel/pp/pipeline_parallel.cc` | 使用 `pp_rank` 直接拼 first/last stage 与 chunk | 改为根据 `PipelineLayout` 拼装本地 chunk |
| `infini_train/src/nn/parallel/pp/pipeline_schedule.cc` | 用 `%` 和 `/` 推导 stage/chunk 归属 | 改为从 `PipelineLayout` 查询归属 |
| `example/gpt2/checkpoint_loader.cc`、`example/llama3/checkpoint_loader.cc` | 使用 `GetStageInfo` 和 `is_first_stage/is_last_stage` 决定读取哪些权重 | 改为查询 `PipelineLayout` |

> 当前 CUDA 构建在本环境中因 `nvcc` 无法识别 `<format>` 而失败，属于与本次功能无关的工具链问题。逻辑类单测和 CPU-only 构建可独立验证；GPU 端到端验证在 CUDA 工具链可用后执行。

## 3. 总体设计

### 3.1 统一布局数据模型

保留并完善现有头文件中的核心结构：

```text
PipelineLayout
├── num_layers
├── num_stages
├── vpp_size
├── std::vector<StageLayout> stages
├── std::vector<ChunkLayout> chunks
├── SpecialModulePlacement placement
├── PipelineLayoutPolicy policy
└── std::vector<LayerLocation> layer_locations
```

关键查询接口：

```cpp
const StageLayout& stage(int stage_id) const;
const ChunkLayout& chunk(int global_chunk_id) const;
const LayerLocation& locate_layer(int layer_id) const;
int stage_of_layer(int layer_id) const;
int local_layer_index(int stage_id, int layer_id) const;
const ChunkLayout& chunk_of_layer(int layer_id) const;
bool owns(SpecialModule module, int stage_id) const;
std::string ToString() const;
void Validate() const;
void ValidateForCurrentPipelineTransport() const;
```

### 3.2 Chunk 排序约定

为兼容现有调度器，`BuildDefault` 生成的 `global_chunk_id` 按 vPP 交错顺序排列：

```text
global_chunk_id = local_chunk_id * num_stages + stage_id
```

也就是说 `chunks[0]` 属于 stage 0 的 virtual chunk 0，`chunks[1]` 属于 stage 1 的 virtual chunk 0，依此类推。这样 GPipe / 1F1B 现有调度表语义不需要推翻，只是把 `%`、`/` 推导改为查表。

### 3.3 通过标准的命令行语义

新增参数：

```bash
--pipeline_layer_partition 4,8,6,6
--pipeline_embedding_stage 0
--pipeline_final_norm_stage 3
--pipeline_lm_head_stage 3
```

行为：

- `--pipeline_layer_partition` 为空时，使用 `BuildDefault(num_layers, pp_size, vpp_size)`，完全保持当前默认均匀布局。
- `--pipeline_layer_partition` 非空时，使用 `BuildContiguous(stage_layer_counts)`，只允许 `vpp_size == 1`；该限制会在通过标准阶段明确报错。
- 三个特殊模块 stage 参数默认值分别为 `0`、`pp_size - 1`、`pp_size - 1`，保持旧行为。
- 自定义非均匀布局示例 `4,8,6,6` 对应 24 层：

```text
stage 0: [embedding] layers [0, 4)
stage 1: layers [4, 12)
stage 2: layers [12, 18)
stage 3: layers [18, 24) [final_norm, lm_head]
```

## 4. 通过标准：实施步骤与验证

### Phase 0：建立可验证的基线

**目标**：先让 CPU-only 测试基线可编译、可运行，避免把工具链问题混入功能开发。

改动：

1. 新建独立 build 目录，关闭 CUDA/NCCL，仅构建测试：

```bash
cmake -S . -B build_cpu -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF
```

2. 临时修正 `pipeline_layout.h` 的 `#progma` 为 `#pragma`，并修复 `transformer.h` 中错误的 `infini_train/include/nn/modules/parallel/global.h` include 路径；让 `pipeline_layout.cc` 至少提供全部声明函数的空实现或最小实现，保证链接。

**验证方式**：

```bash
cmake --build build_cpu -j
ctest --test-dir build_cpu --output-on-failure
```

预期：CPU 测试全部通过，至少不存在 `pipeline_layout.cc` 导致的编译/链接失败。

### Phase 1：完成 `PipelineLayout` 核心实现

**目标**：`PipelineLayout` 成为唯一可信的布局计算与查询来源。

改动：

1. 修正头文件拼写与 include guard。
2. 实现 `LayerRange::size/contains`。
3. 实现 `PipelineLayout::BuildDefault`：
   - `total_chunks = num_stages * vpp_size`；
   - 每 chunk 基础层数 `num_layers / total_chunks`，前 `num_layers % total_chunks` 个 chunk 多一层；
   - 为与当前 `GetStageInfo` 兼容，当 `num_layers < total_chunks` 时允许层数为 0 的 chunk 存在，只生成仍能覆盖层的 chunk；
   - chunk 顺序采用 3.2 节交错顺序；
   - 归一化特殊模块默认位置：embedding 在 stage 0，final_norm、lm_head 在 stage `num_stages - 1`。
4. 实现 `PipelineLayout::BuildContiguous`：
   - 入参为各 stage 层数 `stage_layer_counts`；
   - `num_stages = stage_layer_counts.size()`；
   - 生成连续 `LayerRange`；
   - 默认策略拒绝负数和空 stage；
   - `vpp_size = 1`。
5. 实现 `BuildIndexes`：构造 `stage_of_layer`、`layer_locations` 和 stage chunk 列表。
6. 实现 `Validate`：
   - 层范围完全覆盖 `[0, num_layers)` 且不重复；
   - 所有 chunk/stage 索引合法；
   - 特殊模块 stage 合法；
   - 默认策略下 stage 非空且执行顺序连续。
7. 实现 `ValidateForCurrentPipelineTransport`：当前 send/recv 假设任意两个相邻 stage 仍按 `0..num_stages-1` 连接，因此通过标准阶段要求每个 stage 恰好有可执行的 chunk；非法情况给出可定位错误。
8. 实现 `ToString`，输出示例见 3.3。

**验证方式**：

新增 `tests/parallel/test_pipeline_layout.cc`，覆盖：

```text
BuildDefault(24, 4, 1)  -> stage 范围 [0,6),[6,12),[12,18),[18,24)
BuildDefault(8, 2, 2)   -> chunk 交错顺序与 stage 归属正确
BuildContiguous({4,8,6,6}) -> 与文档示例一致
stage_of_layer/locate_layer/chunk_of_layer/owns 查询一致
BuildContiguous({-1,7,6,6}) -> 抛 PipelineLayoutError
BuildPipelineLayout(23,4,1,"4,8,6,5",default) -> 总和错误，抛 PipelineLayoutError
特殊模块默认位置和显式位置均正确
ToString 包含 num_layers/num_stages/各 stage 层范围
```

命令：

```bash
cmake --build build_cpu -j
./build_cpu/tests/parallel/test_pipeline_layout_cpu --gtest_color=yes
```

### Phase 2：命令行参数与布局解析

**目标**：用户能够通过命令行显式配置层数列表和特殊模块 stage。

改动：

1. 在 `example/gpt2/main.cc`、`example/llama3/main.cc` 中新增：

```cpp
DEFINE_string(pipeline_layer_partition, "", "Comma-separated layer counts per PP stage, e.g. 4,8,6,6");
DEFINE_int32(pipeline_embedding_stage, 0, "Stage that owns embedding.");
DEFINE_int32(pipeline_final_norm_stage, -1, "Stage that owns final norm; -1 means last PP stage.");
DEFINE_int32(pipeline_lm_head_stage, -1, "Stage that owns LM head; -1 means last PP stage.");
```

2. 在 `pipeline_layout.h/cc` 中新增：

```cpp
std::vector<int> ParseLayerPartition(const std::string& value);
PipelineLayout BuildPipelineLayout(
    int num_layers,
    int pp_size,
    int vpp_size,
    const std::string& layer_partition,
    SpecialModulePlacement placement,
    PipelineLayoutPolicy policy = {});
void InstallPipelineLayout(/* 同样参数 */);
```

3. 解析规则：
   - 严格接受 `1,2,3` 形式，项必须是非负整数；
   - 拒绝空项、非数字、负数、首尾逗号；
   - `BuildPipelineLayout` 校验项数等于 `pp_size`、层数总和等于 `num_layers`；
   - 自定义 partition 与 `vpp_size > 1` 同时出现时立即报错。

4. 在 GPT-2 / LLaMA 的模型配置确定后调用 `InstallPipelineLayout`：
   - 从 checkpoint 加载时：在 `LoadFromLLMC` 读取 header 得到 `n_layer` 并完成 `Sanitize*Config` 后调用；
   - 随机初始化路径：在 `Train` 中 `Sanitize*Config` 后、构造 `TransformerModel` 前调用。
   - `set_pipeline_layout` 增加线程安全保护，保证多线程直启时所有线程写入相同布局。

**验证方式**：

新增 `tests/parallel/test_pipeline_layout_parser.cc`：

```text
ParseLayerPartition("4,8,6,6") == {4,8,6,6}
ParseLayerPartition("4,,6") 失败
ParseLayerPartition("-1,25") 失败
ParseLayerPartition("4,8,6") 在 pp_size=4 时失败
BuildPipelineLayout(24,4,1,"4,8,6,6",default) 成功
BuildPipelineLayout(23,4,1,"4,8,6,6",default) 总和错误失败
BuildPipelineLayout(24,4,2,"4,8,6,6",default) 自定义与 vPP 冲突失败
```

命令：

```bash
./build_cpu/tests/parallel/test_pipeline_layout_parser_cpu --gtest_color=yes
```

### Phase 3：模型构建接入 `PipelineLayout`

**目标**：每个 rank 只创建本 stage 拥有的 Transformer 层与特殊模块。

改动：

1. 删除 `TransformerModel` 对 `PipelineParallel::GetStageInfo` 的依赖，构造函数中：

```cpp
const auto& layout = nn::parallel::global::GetPipelineLayout();
const int stage_id = nn::parallel::pp_rank;
const auto& stage = layout.stage(stage_id);
```

2. 遍历 `stage.global_chunk_ids`，用每个 chunk 的 `LayerRange` 创建 `TransformerChunk`，并将层加入 `transformer.h`。
3. 用 `layout.owns(kEmbedding, stage_id)` 决定是否创建 `TransformerFirstStage`。
4. 将 `TransformerLastStage` 改为可裁剪构造：

```cpp
TransformerLastStage(const TransformerConfig& config, bool has_final_norm, bool has_lm_head);
```

`Forward` 根据存在性分别执行 final norm 和 lm head；当二者分属不同 stage 时，中间隐藏状态通过 pipeline 传输。

5. 保持 module 名和 state dict key 兼容：

```text
transformer.wte
transformer.wpe
transformer.h.<local_index>.ln_1/attn/ln_2/mlp
transformer.ln_f
transformer.lm_head
```

6. `TransformerConfig::GetChunkSize()` 改为：

```cpp
return global::GetPipelineLayout().stage(global::GetPPRank()).global_chunk_ids.size();
```

**验证方式**：

在 `tests/transformer/test_transformer_architecture.cc` 中增加布局测试：

```cpp
// 临时安装 BuildContiguous({2,4,3,3})，n_layer=12
// 依次设置 pp_rank = 0..3，构造 TransformerModel
// 检查 StateDict 与 NamedParameters：
//   stage0 有 transformer.wte/wpe，无 ln_f/lm_head
//   stage1 只有对应 transformer.h 层
//   stage3 有 transformer.ln_f/lm_head
```

命令：

```bash
./build_cpu/tests/transformer/test_transformer_architecture_cpu \
  --gtest_filter='*PipelineLayout*' --gtest_color=yes
```

### Phase 4：PipelineStage 与 PipelineParallel 包装接入布局

**目标**：Pipeline 包装器不再依赖 `is_first_stage/is_last_stage` 和固定 chunk 下标，而是根据布局拼装本地模块。

改动：

1. `PipelineParallel` 构造函数读取：

```cpp
const auto& layout = global::GetPipelineLayout();
const auto& stage = layout.stage(rank_);
```

2. 对 `stage.global_chunk_ids` 中的每个 chunk：
   - 按需加入 `TransformerFirstStage`（`layout.owns(kEmbedding, rank_)` 且是该 stage 第一个 chunk）；
   - 加入对应 `TransformerChunk`；
   - 按需加入 `TransformerLastStage`（`layout.owns(kFinalNorm, rank_) || layout.owns(kLMHead, rank_)` 且是该 stage 最后一个 chunk）。
3. 移除 `GetStageInfo` 在包装路径中的使用；保留或删除需统一，避免两套算法并存。
4. 增加防御性检查：如果本地 chunk 列表为空或某 chunk 没有可执行 module，在 `ValidateForCurrentPipelineTransport` 阶段拒绝，而不是运行时崩溃。

**验证方式**：

新增 `tests/parallel/test_pipeline_parallel_chunking.cc`（CPU 逻辑测试，不发起真实通信）：

```text
安装 BuildContiguous({2,4,3,3})，n_layer=12
为 stage0 构造 PipelineParallel，mutable_chunks().size()==1
stage0 chunk 含 TransformerFirstStage，不含 TransformerLastStage
stage3 chunk 含 TransformerLastStage，不含 TransformerFirstStage
```

命令：

```bash
./build_cpu/tests/parallel/test_pipeline_parallel_chunking_cpu --gtest_color=yes
```

### Phase 5：Pipeline 调度器统一查询布局

**目标**：调度器不再用 `global_chunk % num_stages`、`global_chunk / num_stages` 推导所有权。

改动：

1. 给 `PipelineParallelScheduler::CreateTask`、`GenerateGPipeSchedule`、`GenerateInterleaved1F1BSchedule` 增加 `const PipelineLayout&` 参数。
2. `CreateTask` 改为：

```cpp
const auto& chunk = layout.chunk(global_chunk_id);
task.local_chunk_idx = chunk.local_chunk_id;
task.stage_id = chunk.stage_id;
```

3. `StepMicroBatches` 生成 schedule 时传入 `global::GetPipelineLayout()`。
4. `PrintScheduleTable` 同样从 layout 查 stage，而不是取模。
5. 现有默认布局测试保证 vPP 下的交错调度结果与旧实现完全一致。

**验证方式**：

新增 `tests/parallel/test_pipeline_scheduler_layout.cc`：

```text
BuildDefault(12, 3, 2) 下 GenerateGPipeSchedule 的每个 task.stage_id
  等于 layout.chunk(global_chunk_id).stage_id
BuildContiguous({2,4,3,3}) 下 CreateTask 查询结果正确
GPipe / 1F1B task 数量公式不变
```

命令：

```bash
./build_cpu/tests/parallel/test_pipeline_scheduler_layout_cpu --gtest_color=yes
```

### Phase 6：参数加载统一查询布局

**目标**：checkpoint 加载不再使用 `GetStageInfo`，而是根据同一份布局决定本 rank 应读取哪些参数。

改动：

1. 在 GPT-2 / LLaMA loader 中删除：

```cpp
auto [is_first_stage, is_last_stage, layer_ranges_per_chunk] = GetStageInfo(...);
```

2. 改为：

```cpp
const auto& layout = global::GetPipelineLayout();
const bool owns_embedding = layout.owns(SpecialModule::kEmbedding, pp_rank);
const bool owns_final_norm = layout.owns(SpecialModule::kFinalNorm, pp_rank);
const bool owns_lm_head = layout.owns(SpecialModule::kLMHead, pp_rank);
std::vector<bool> owned_layers(n_layer, false);
for (int layer = 0; layer < n_layer; ++layer) {
    owned_layers[layer] = layout.stage_of_layer(layer) == pp_rank;
}
```

3. GPT-2 loader 中所有 `is_first_stage/is_last_stage` 分支替换为上述 `owns_*`。
4. LLaMA loader 中 `ln_f` 与 `lm_head` 读取分支拆开，分别由 `owns_final_norm`、`owns_lm_head` 控制。
5. 保留文件流 seek 逻辑：即使本 rank 不拥有某权重，也必须按顺序跳过对应字节。

**验证方式**：

单元测试不直接读真实权重文件，采用以下组合验证：

- 新增小规模 fake LLMC bin 工具，或复用 `scripts/assets` 生成的小模型文件；
- 以 `--pipeline_parallel 2 --pipeline_layer_partition 6,6` 运行 GPT-2，日志应显示 rank0 加载 wte/wpe + layers 0..5，rank1 加载 layers 6..11 + ln_f/lm_head；
- 以 `--pipeline_parallel 2 --pipeline_layer_partition 8,4` 运行，检查层归属相应变化。

命令（GPU 可用时）：

```bash
./build/infini_run --nnodes=1 --nproc_per_node=2 \
  ./build/gpt2 --device cuda \
  --input_bin data/gpt2/tiny_shakespeare_train.bin \
  --input_val_bin data/gpt2/tiny_shakespeare_val.bin \
  --tokenizer_bin data/gpt2/gpt2_tokenizer.bin \
  --llmc_filepath data/gpt2/gpt2_124M.bin \
  --pipeline_parallel 2 --pipeline_layer_partition 6,6 \
  --num_iteration 1
```

### Phase 7：合法性校验与启动日志

**目标**：任何非法布局都在训练开始前被拒绝，并输出可定位错误。

改动：

1. 在 `InstallPipelineLayout` 后立即调用 `layout.Validate()` 和 `layout.ValidateForCurrentPipelineTransport()`。
2. 主 rank 启动时打印 `layout.ToString()`；其他 rank 至少打印 stage 归属摘要。
3. 统一错误格式：

```text
PipelineLayoutError: stage count mismatch: pipeline_parallel=4, layer_partition has 3 entries
PipelineLayoutError: layer sum mismatch: n_layer=23, partition sum=24
PipelineLayoutError: negative layer count in partition at stage 0: -1
PipelineLayoutError: custom layer partition is not supported with virtual_pipeline_parallel=2
```

**验证方式**：

使用错误参数启动，断言进程在进入训练循环前退出且日志包含上述原因：

```bash
./build/gpt2 --pipeline_parallel 4 --pipeline_layer_partition 4,8,6
./build/gpt2 --pipeline_parallel 4 --pipeline_layer_partition -1,8,6,6
./build/gpt2 --pipeline_parallel 4 --virtual_pipeline_parallel 2 --pipeline_layer_partition 4,8,6,6
```

同时新增 parser 单测覆盖这些错误。

### Phase 8：通过标准端到端验证

**目标**：至少完成一个 2-stage 端到端训练，且与单卡/默认布局数值对齐。

改动：

1. 新增脚本 `scripts/test_pipeline_custom_layout.sh`，负责：
   - 固定 `--overfit_single_batch=true`；
   - 固定随机种子和输入数据；
   - 依次运行 PP=1 基线与 PP=2 自定义 `6,6`；
   - 从日志解析 loss；
   - 调用现有 `scripts/compare_loss.py` 比较。
2. 在训练日志中额外输出可比较的标量：loss、当前 step、dtype。
3. FP32 容差使用 `1e-5`，BF16 容差使用 `1e-2`。

**验证方式**：

GPU 环境：

```bash
./scripts/test_pipeline_custom_layout.sh gpt2 cuda
./scripts/test_pipeline_custom_layout.sh llama3 cuda
```

预期输出：

```text
baseline loss: 3.141592
custom pp2 loss: 3.141594
max_abs_diff: 0.000002 < tolerance: 0.000010 -> PASS
```

同时确认训练全程无 send/recv 死锁。

## 5. 通过标准测试矩阵

| 测试 | 层级 | 命令/入口 | 覆盖点 |
|---|---|---|---|
| `test_pipeline_layout` | CPU 单测 | `ctest -R test_pipeline_layout_cpu` | 默认布局、连续自定义、查询接口、ToString |
| `test_pipeline_layout_parser` | CPU 单测 | `ctest -R test_pipeline_layout_parser_cpu` | CLI 解析、非法配置拒绝 |
| `test_pipeline_parallel_chunking` | CPU 单测 | `ctest -R test_pipeline_parallel_chunking_cpu` | 本地 stage/chunk 组装、特殊模块归属 |
| `test_pipeline_scheduler_layout` | CPU 单测 | `ctest -R test_pipeline_scheduler_layout_cpu` | 调度器查表而非取模 |
| `test_transformer_architecture` 扩展 | CPU/GPU 单测 | `ctest -R test_transformer_architecture` | GPT-2/LLaMA 模型构建与 state dict 归属 |
| GPT-2 2-stage e2e | GPU 分布式 | `scripts/test_pipeline_custom_layout.sh gpt2 cuda` | 通信、loss、fp32/bf16 对齐 |
| LLaMA 3 2-stage e2e | GPU 分布式 | `scripts/test_pipeline_custom_layout.sh llama3 cuda` | 通信、loss、fp32/bf16 对齐 |

## 6. 优秀标准：扩展方案

### 6.1 推荐路径：Megatron-LM 风格布局字符串 + vPP 显式 Chunk 映射

在通过标准之上新增 `--pipeline_layout` 字符串参数，兼容并扩展 Megatron-LM 的语义。

语法：

```text
E        = embedding
t        = transformer layer
F        = final norm
H        = lm head / loss head
|        = stage 分隔符
,        = 同一 stage 内 virtual chunk 分隔符
(...) * N = 重复表达
```

示例：

```bash
--pipeline_parallel 4 \
--virtual_pipeline_parallel 2 \
--pipeline_layout "E,(tt|)*2,FH|t,(t|)*1,FH|tt,tt|tt,tt"
```

对应目标：

- 显式描述任意 `chunk -> stage` 映射，不再依赖固定轮转；
- 支持重复层、特殊模块、空 stage 和 vPP chunk；
- 通过 `Validate()` 校验每 stage 的 chunk 数量、层覆盖、特殊模块唯一性。

**实施步骤**：

1. 新增 `ParseMegatronStyleLayout(const std::string&, int num_layers, int pp_size, int vpp_size)`。
2. 将字符串解析为 `StageLayout` / `ChunkLayout`，复用 `PipelineLayout` 私有构造与 `BuildIndexes`。
3. 调度器完全使用 layout 查询后，vPP 非轮转映射自然生效。
4. 模型构建和参数加载无需新增特殊逻辑，因为前文已按 layout 查询。

**验证方式**：

新增 `tests/parallel/test_pipeline_layout_megatron_style.cc`：

```text
解析空 stage：E||t|L 生成 stage1 无 layer chunk
解析重复层：(tt)*3 展开为 6 个 t
解析 vPP：同一 stage 的逗号生成多个 local chunk
非法表达式：不匹配括号、未知符号、层数总和错误均被拒绝
```

命令：

```bash
./build_cpu/tests/parallel/test_pipeline_layout_megatron_style_cpu --gtest_color=yes
```

GPU 冒烟：

```bash
./build/infini_run --nnodes=1 --nproc_per_node=4 \
  ./build/gpt2 --device cuda --pipeline_parallel 4 \
  --virtual_pipeline_parallel 2 \
  --pipeline_layout 'E,(tt|)*2,FH|t,(t|)*1,FH|tt,tt|tt,tt' \
  --num_iteration 1 ...
```

### 6.2 可选路径 A：自动负载均衡布局建议

目标：

- 提供 `--pipeline_layout_balance auto` 或独立工具，依据参数量、Profiler 统计或用户代价生成近似均衡布局。
- 输入形式至少支持 `--pipeline_layer_cost 1.0,2.0,...`。
- 输出推荐 partition 字符串并打印到日志。

实施：

1. 新增 `SuggestBalancedLayout(num_layers, num_stages, vpp_size, layer_costs)`。
2. 使用前缀和 + 目标总代价/阶段数贪心切分，保证连续层范围。
3. 若启用 Profiler，则优先用 profiler 统计替换 layer_costs。

**验证方式**：

```bash
./build/tools/pipeline_layout_suggest --num_layers 8 --pp_size 4 \
  --layer_cost 1,1,1,1,4,1,1,1
```

预期建议将 4 代价的层单独或与轻层组合，避免与相邻重层同阶段。

### 6.3 可选路径 B：Pipeline bubble 与吞吐对比

目标：

- 对默认均匀布局和自定义布局输出 pipeline bubble、各 stage 执行时间、吞吐。
- 证明自定义布局能改善负载不均场景。

实施：

1. 在 `PrintScheduleTable` 旁增加统计：理想时间、实际时间、bubble 比例。
2. 训练结束时输出 `per-stage forward/backward time` 与 `throughput`。
3. 新增脚本 `scripts/compare_pipeline_layout_perf.sh`，用不平衡模型/代价运行两种布局并绘图或输出 CSV。

**验证方式**：

```bash
./scripts/compare_pipeline_layout_perf.sh gpt2 cuda
```

预期生成 `targets/pipeline_layout_perf.csv`，包含默认与自定义布局的 bubble 和吞吐。

### 6.4 优秀标准建议提交物

至少满足一项即可，建议按优先级选择：

1. Megatron 风格字符串与 vPP 显式映射；
2. 负载均衡自动建议；
3. bubble/吞吐对比；
4. 仓库 PR review 流程。

## 7. 交付物清单

通过标准：

- 实现代码：
  - `PipelineLayout` 完整实现；
  - 模型构建、Pipeline 调度、参数加载的布局接入；
  - GPT-2 与 LLaMA 3 命令行参数。
- 测试：
  - `tests/parallel/test_pipeline_layout.cc`
  - `tests/parallel/test_pipeline_layout_parser.cc`
  - `tests/parallel/test_pipeline_parallel_chunking.cc`
  - `tests/parallel/test_pipeline_scheduler_layout.cc`
  - `tests/transformer/test_transformer_architecture.cc` 扩展
  - `scripts/test_pipeline_custom_layout.sh`
- 文档：
  - `docs/pipeline_custom_layout_usage.md`：参数配置、布局语法、默认行为、输入输出示例、错误排查。
- 报告：
  - `targets/Pipeline并行自定义布局报告.md`：数据结构与接口设计、关键实现说明、兼容性说明、正确性与负载分析。

优秀标准额外交付：

- `--pipeline_layout` 字符串解析器与测试；
- 负载均衡建议工具/接口；
- 性能对比脚本与 CSV；
- PR 提交与 review 记录。

## 8. 风险与兼容性

1. **默认行为回归**：所有改动都保留 `BuildDefault` 作为空参数默认路径；用当前 `GetStageInfo` 算法生成 golden 测试，替换后结果必须一致。
2. **vPP 兼容性**：通过标准明确限制自定义 partition 与 vPP 冲突；优秀阶段通过 Megatron 字符串扩展。
3. **checkpoint 格式兼容**：保持 `transformer.h.<local_index>`、`transformer.ln_f`、`transformer.lm_head` 等 state dict key 不变，避免影响已有权重加载和 LoRA/checkpoint。
4. **特殊模块分属不同 stage**：将 `TransformerLastStage` 改为可裁剪，避免 final norm 与 lm head 绑定；若暂时只支持二者同 stage，也应通过 `ValidateForCurrentPipelineTransport` 给出明确错误，不能静默错误加载。
5. **多线程安装布局**：`set_pipeline_layout` 需要加锁；布局对象对所有线程相同，避免数据竞争。
6. **CUDA 工具链**：当前环境 `nvcc` 对 `<format>` 的兼容问题与本任务正交；CPU-only 测试先行，GPU e2e 在工具链满足 `gcc 13+`/合适 CUDA 版本后执行。

## 9. 推荐执行顺序

1. Phase 0：CPU-only 基线可编译。
2. Phase 1：`PipelineLayout` 核心实现与单测。
3. Phase 2：CLI 解析与安装。
4. Phase 3：模型构建接入。
5. Phase 4：PipelineStage/包装接入。
6. Phase 5：调度器接入。
7. Phase 6：参数加载接入。
8. Phase 7：启动校验与日志。
9. Phase 8：2-stage 端到端与数值对齐。
10. 通过标准验收后，再进入第 6 节优秀扩展。

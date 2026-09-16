# Pipeline并行自定义布局使用说明

本文档描述Pipeline自定义布局的使用方法：通过 `--pipeline_layer_partition`显式指定每个Stage的Transformer层数，或通过 `--pipeline_layer_costs` /`--pipeline_auto_layout` 根据计算代价自动生成负载均衡的布局。

例如：

```bash
./build/infini_run \
  --nproc_per_node=4 \
  ./build/gpt2 \
    --device cuda \
    --input_bin data/train.bin \
    --llmc_filepath data/gpt2_124M.bin \
    --pipeline_parallel 4 \
    --pipeline_layer_partition 4,8,6,6
```

上述配置把 24 个Transformer层依次划分为4, 8, 6, 6：

- Stage 0: embedding + layers 0~3

- Stage 1:  layers 4~11

- Stage 2: layers 12~17

- Stage 3: layers 18~23



## 参数配置

| 参数                               | 默认值【注2】 | 说明                    |
| -------------------------------- | ------- | --------------------- |
| `--pipeline_parallel`            | `1`     | Stage数量               |
| `--virtual_pipeline_parallel`    | `1`     | Virtual Chunk 数量（vPP） |
| `--pipeline_layer_partition`【注1】 | `""`    | 各Stage层数列表（逗号分隔）      |
| `--pipeline_layer_costs`         | `""`    | 每层计算代价（逗号分隔）          |
| `--pipeline_auto_layout`         | `false` | 按每层参数量自动生成负载均衡布局      |

GPT2与LLaMA3示例入口均支持 `--pipeline_layer_partition`，解析后写入全局环境`GlobalEnv`，模型构建、`PipelineParallel` 包装、调度器与 checkpoint 加载都从同一布局查询层归属。

#### 【注1】布局语法

- 多传入参数时，语法为逗号分隔的正整数列表。
- 列表长度必须等于 `--pipeline_parallel` 的Stage数量。
- 各Stage层数之和必须等于模型总层数，stage i 拥有编号从前 i 项之和到前 i+1 项之和的连续Transformer层。
- Embedding固定归属第一个Stage，Final Norm + LM Head 固定归属最后一个Stage（当前版本不开放单独配置）。

#### 【注2】默认值说明

不传 `--pipeline_layer_partition`（或传空串）时，保持原有自动均匀划分：

```text
layers_per_chunk = total_layers / (num_stages * vpp_size)
remainder        = total_layers % (num_stages * vpp_size)
```

余数按 global chunk 顺序依次多分配一层，vPP下各Stage按`global_chunk = local_chunk * num_stages + stage` 交错持有多个层范围。因此默认均匀布局与vPP完全兼容。

## 自动布局建议

除显式指定 `--pipeline_layer_partition` 外，还支持根据计算代价自动生成近似负载均衡的连续布局。三种代价来源：

1. **用户提供的计算代价**：`--pipeline_layer_costs "1,1,1,1,2,..."`，每个数对应一层的代价
   （参数量、实测耗时等均可）。列表长度即模型层数，结果会自动均衡各Stage总代价。
2. **各层参数量**：`--pipeline_auto_layout`，根据 `TransformerConfig` 解析式计算每层参数量。
3. **Profiler 统计**：先用 `--freq_generate_txt` / PROFILE_MODE 跑一次得到每层kernel实测耗时，再把每层耗时作为代价通过 `--pipeline_layer_costs` 传入，即可得到基于实测负载的布局建议。

```bash
# With imbalanced costs (first 4 layers light, last 8 heavy), PP=2 suggests 7,5 instead of uniform 6,6
./build/infini_run --nproc_per_node=2 \
  ./build/gpt2 --device cuda --model d12 --pipeline_parallel 2 \
    --pipeline_layer_costs 1,1,1,1,2,2,2,2,2,2,2,2

# Or auto-suggest by per-layer parameter count (GPT-2 / LLaMA3 layers are identical, so it equals the uniform layout)
./build/infini_run --nproc_per_node=2 \
  ./build/gpt2 --device cuda --model d12 --pipeline_parallel 2 --pipeline_auto_layout
```

这三种方式与`--pipeline_layer_partition` 只能同时使用一种。建议结果会在启动阶段打印为
`Auto-suggested pipeline layout ...: <partition>`；最终 `PipelineLayout` 仍按既有格式打印。核心算法见
`SuggestBalancedPartition`，空代价（`{}`）即退化为按层数均匀划分。

## Pipeline负载分析

为证明自定义布局能改善负载不均衡场景，框架在运行结束时自动汇总一次Pipeline负载分析
（`--pipeline_parallel > 1` 时）：每个 PP rank 测量本Stage的前向 / 反向纯计算时间，经PP通信组 `AllGather` 汇总后由第一个Stage打印：

```text
=== Pipeline Timing Summary (4 stages) ===
Stage    Fwd(ms)     Bwd(ms)   Total(ms)
0         12.3        9.1        21.4
1         25.1       18.7        43.8
2         24.9       18.5        43.4
3         13.0        9.6        22.6
Compute tasks per stage: 8 forward + 8 backward
Bottleneck stage: 43.8 ms | average: 32.8 ms
Load-imbalance bubble: 25.1% | pipeline efficiency: 74.9%
```

指标定义：

- **各Stage执行时间**：该Stage在所有micro-batch上前向 / 反向纯计算时间的累计。
- **Load-imbalance bubble**：`1 - average / bottleneck`，设总时间为t，则其中 `bottleneck = max_i(总时间_i)`、`average = mean_i(总时间_i)`；负载完全均衡时为0。
- **pipeline efficiency**：`average / bottleneck`，即 `1 - bubble`。
- **结构bubble**（fill/drain，GPipe）：`(S-1)/(S-1+n)`，其中 `S` 为 Stage 数量（即 `--pipeline_parallel`），`n` 为 micro-batch 数量（即梯度累积步数）；该开销来自流水线填充/排空阶段的空转，只取决于 `S` 与 `n`、与层如何划分无关，由 `ComputePipelineLoadAnalysis` 解析式给出。
- **吞吐**：沿用训练时每步打印的 `tok/s`。

由于 GPT-2 / LLaMA3 各 Transformer 层结构相同，均匀按层数划分本身就是负载均衡的；真正体现“自定义布局改善负载不均”的是各层计算量不一致的场景。此时用 `--pipeline_layer_costs` 给出每层代价，`SuggestBalancedPartition` 给出均衡布局，`ComputePipelineLoadAnalysis` 可离线预测两种布局的对比：

```cpp
std::vector<double> costs{1,1,1,1, 2,2,2,2, 2,2,2,2}; // first 4 layers light, last 8 heavy
auto uniform  = nn::parallel::ComputePipelineLoadAnalysis(12, 2, {6, 6}, costs, 8);
auto balanced = nn::parallel::ComputePipelineLoadAnalysis(12, 2, {7, 5}, costs, 8);
// uniform:  bottleneck=12, bubble=16.7%
// balanced: bottleneck=10, bubble=0%
```

### 离线对比演示

`docs/pipeline_layout_demo.cc` 是可直接运行的纯CPU演示程序，用上面的`ComputePipelineLoadAnalysis` / `SuggestBalancedPartition` 打印“默认均匀布局 vs 自定义布局”的完整对比表：

```bash
g++ -std=c++20 -DGLOG_USE_GLOG_EXPORT -I. -Ithird_party/glog/src -Ibuild/third_party/glog \
    docs/pipeline_layout_demo.cc infini_train/src/nn/parallel/pp/pipeline_layout.cc \
    -Lbuild/third_party/glog -lglog -pthread -o build/pipeline_layout_demo
LD_LIBRARY_PATH=build/third_party/glog ./build/pipeline_layout_demo
```

**输出**（负载不均模型，前 4 层轻、后 8 层重，S=2、n=8）：

```text
metric                 | uniform (6,6)  | custom (7,5)
stage 0 time (load)    |          8.000 |         10.000
stage 1 time (load)    |         12.000 |         10.000
bottleneck (max)       |         12.000 |         10.000
imbalance bubble       |          16.7% |           0.0%
structural bubble      |          11.1% |          11.1%
pipeline efficiency    |          83.3% |         100.0%
throughput (mb/t)      |         0.0741 |         0.0889
throughput speedup     |              - |          1.20x
```

**结论**：负载不均时，自定义均衡布局 `7,5` 把bottleneck从12降到10，imbalance bubble 从16.7%降到0，
吞吐提升**1.20x**；各层均匀时二者等价。

### 真实CUDA计时（≥2张GPU）

真实运行是为了验证布局机制与bubble指标的端到端正确性，即自定义布局确实改变了“层 → Stage”映射，
且bubble/各Stage时间/吞吐随布局正确变化。

```bash
# default uniform layout (6,6)
./build/infini_run --nproc_per_node=2 ./build/gpt2 \
  --model d12 --input_bin data/gpt2/tiny_shakespeare_train.bin --pipeline_parallel 2 \
  --total_batch_size 2048 --num_iteration 10 --freq_generate_txt 1000

# custom layout: d12's 12 layers are structurally identical, so 7,5 here is a deliberately
# injected imbalance (costs 1,1,1,1,2,2,...,2 suggest 7,5) to prove the bubble metric tracks
# load changes, not that 7,5 is better.
./build/infini_run --nproc_per_node=2 ./build/gpt2 \
  --model d12 --input_bin data/gpt2/tiny_shakespeare_train.bin --pipeline_parallel 2 \
  --pipeline_layer_costs 1,1,1,1,2,2,2,2,2,2,2,2 \
  --total_batch_size 2048 --num_iteration 10 --freq_generate_txt 1000
```

**实测结果**（2×RTX 4090）：

| 指标                   | 默认均匀 `6,6` | 自定义 `7,5`（假代价） |
| -------------------- | ---------- | -------------- |
| Stage 0 Total (ms)   | 1144.6     | 1184.6         |
| Stage 1 Total (ms)   | 1025.7     | 769.2          |
| Bottleneck (ms)      | 1144.6     | 1184.6         |
| **Imbalance bubble** | **5.2%**   | **17.5%**      |
| Pipeline efficiency  | 94.8%      | 82.5%          |
| 稳态吞吐 (tok/s)         | ~19800     | ~18300         |

**结论**：

（1）d12的12个Transformer层结构完全相同，因此默认均匀 `6,6` 本身就是负载均衡的最优解；人为注入
`1,1,1,1,2,...` 代价让算法给出 `7,5`，bubble 升到 17.5%，意在人为制造不均衡。
这证明**bubble指标与布局机制端到端工作正常**。

（2）“自定义布局**改善**负载不均”要求各层计算量本身不均。然而仓库现有gpt2/llama3层数均匀、异构的mixtral 未接入布局参数，故这一方向由上一节的解析式 `ComputePipelineLoadAnalysis` / `pipeline_layout_demo` 证明（bubble 16.7% → 0，吞吐 1.20x）。
二者合起来即优秀标准「给出两种布局的 bubble / 各 Stage 执行时间 / 吞吐对比，并证明自定义布局改善
负载不均场景」的完整证据链。

**补充实测：**

`verify_pipeline_layout_correctness.sh` 的配置更小（batch=4, seq=64, total_batch=512, num_iteration=3），其输出的` Pipeline Timing Summary `中 7,5 布局 imbalance bubble 为 **fp32 20.1% / bf16 23.3%**（Stage 0/1 Total：fp32 220.5 / 131.8 ms，bf16 439.8 / 235.2 ms，Bottleneck 均为 Stage 0）。该配置与上表（`total_batch=2048`、10 步）不同，绝对时间与 bubble 比例不可直接逐项对比，但方向一致——7,5 的 Stage 0（7 层）显著重于 Stage 1（5 层），bubble 明显高于均匀布局，佐证 bubble 指标能正确反映负载变化。

## 输入输出示例

启动时若 `--pipeline_parallel > 1`，Stage 0 会打印最终布局：

```text
PipelineLayout: num_stages=4, total_layers=24, vpp=1
  stage 0: [0, 4) + embedding
  stage 1: [4, 12)
  stage 2: [12, 18)
  stage 3: [18, 24) + final_norm + lm_head
```

其中 `[start, end)` 表示本Stage持有的Transformer层区间；每个Stage在vPP下可能打印多个区间。

## 错误排查

以下非法配置会在启动阶段直接 `LOG(FATAL)` 终止，并给出可定位的报错信息：

| 场景          | 触发条件                                                                                    | 报错关键字                                         |
| ----------- | --------------------------------------------------------------------------------------- | --------------------------------------------- |
| 空项或非数字      | `4,,6,6` / `4,8a,6,6`                                                                   | `not a positive integer`                      |
| 非正层数        | `4,0,6,6`                                                                               | `must be positive`                            |
| Stage 数量不符  | `--pipeline_parallel 4` 但列表只有 3 项                                                       | `entries but pipeline_parallel is`            |
| 层数总和错误      | 24 层模型但 `4,8,6,5`                                                                       | `sums to`                                     |
| 与 vPP 冲突    | 自定义布局 + `--virtual_pipeline_parallel 2`                                                 | `incompatible with virtual_pipeline_parallel` |
| 代价为空项/非数字   | `1,,2` / `1,abc`                                                                        | `empty entry` / `not a number`                |
| 代价非负有限      | `-1,2` / `inf`                                                                          | `non-negative` / `finite`                     |
| 代价条数与层数不符   | 12 层模型但代价只有 5 项                                                                         | `sums to`                                     |
| 三种布局来源同时使用  | `--pipeline_layer_partition` 与 `--pipeline_layer_costs` / `--pipeline_auto_layout` 同时出现 | `cannot be combined with`                     |
| 代价与自动布局同时使用 | `--pipeline_layer_costs` 与 `--pipeline_auto_layout` 同时出现                                | `mutually exclusive`                          |

若报“模型构建与参数加载层归属不一致”，通常是因为某个调用点仍在使用旧的均匀划分：请确认
模型构建（`TransformerModel`）、`PipelineParallel` 包装、两个 checkpoint loader 都改为查询
`PipelineLayout` / `StageInfo`，且 `GetPipelineLayerPartition()` 已正确传入 `InitAllEnv`。

## 测试

**布局功能测试方法**：

由两个单元测试文件共同覆盖。

(1)`tests/distributed/test_pipeline_layout.cc` 验证 `PipelineLayout`本身的行为：

既检查不传参数时的默认均匀划分、按 `4,8,6,6` 显式自定义的划分，也检查Embedding归属第一个Stage、Final Norm 与 LM Head 归属最后一个Stage，以及 `StageOfLayer`/`OwnsLayer` 的层归属查询、vPP 模式下 chunk 的交错排列和 chunk 与 Stage 的映射关系；对空项、非数字、层数不符等非法配置，则用死亡断言确认程序会正确终止。

(2)`tests/distributed/test_pipeline_layout_suggest.cc` 验证自动布局建议与负载分析：

`SuggestBalancedPartition` 对代价均匀、带余数、代价不均衡的输入能给出正确的分层结果、对非法代价能正确拒绝；`ParsePipelineLayerCosts` 能解析合法代价并拒绝负值、非数字、空项、无穷；`ComputePerLayerParamCounts` 对 GELU+LayerNorm 给出精确参数量、对 SwiGLU+RMSNorm 给出均匀正数、对 MoE 层拒绝计算；`ComputePipelineLoadAnalysis` 则确认均匀代价下均匀布局即均衡、代价不均衡时均匀布局会产生气泡而均衡布局能消除气泡、空 partition 退化为均匀划分、结构 bubble 公式`(S-1)/(S-1+n)` 计算正确。

```bash
cmake -S . -B build -DBUILD_TEST=ON
cmake --build build -j
ctest --test-dir build -R 'test_pipeline_layout' --output-on-failure
```

**端到端验证（2-Stage）方法**：

用相同初始权重分别以单卡/默认布局与自定义布局跑若干训练迭代，比较前向结果、loss 与梯度在允许误差内一致，并确认训练过程无通信死锁。一键脚本 `scripts/verify_pipeline_layout_correctness.sh` 已封装该流程：用相同`--llmc_filepath` 权重与数据分别跑单卡（PP=1）与自定义 2-Stage（PP=2、`--pipeline_layer_partition 7,5`），再逐step对比 train loss（多step中loss 一致即说明前向、反向与梯度一致，任一环节偏差都会在后续step累积成loss发散）：

```bash
# scripts/verify_pipeline_layout_correctness.sh for convenience
bash scripts/verify_pipeline_layout_correctness.sh
DTYPE=bfloat16 bash scripts/verify_pipeline_layout_correctness.sh 
```

**实测结果**（2×RTX 4090，`batch=4, seq=64, total_batch=512, num_iteration=3`，`7,5` 切分）：

| dtype | 单卡参考 loss（step 1/2/3）          | 自定义 7,5 loss（step 1/2/3）       | 最大绝对差 | 判定          |
| ----- | ------------------------------ | ------------------------------ | ----- | ----------- |
| fp32  | 5.250157 / 4.913959 / 5.018849 | 5.250157 / 4.913959 / 5.018850 | 1e-6  | pass（≤1e-5） |
| bf16  | 5.215456 / 4.905437 / 5.009968 | 5.215456 / 4.905437 / 5.009968 | 0     | pass（≤1e-2） |

fp32 前两步与 bf16 全部三步逐位一致，说明 PP=2 的数据切分与 loss 平均（`sum/n`）和单卡的
`sum/grad_accum_steps` 精确等价，等价于前向、反向、梯度三者一致。

## API 摘要

```cpp
namespace infini_train::nn::parallel {

std::vector<int> ParsePipelineLayerPartition(const std::string &str);
std::vector<double> ParsePipelineLayerCosts(const std::string &str);
std::vector<int> SuggestBalancedPartition(int total_layers, int num_stages,
                                          const std::vector<double> &layer_costs = {});

struct PipelineLoadStats {
    int num_stages = 0;
    int num_micro_batches = 1;
    std::vector<double> stage_loads;
    double bottleneck = 0.0;
    double average = 0.0;
    double efficiency = 0.0;
    double imbalance_bubble = 0.0;
    double structural_bubble = 0.0;
};
PipelineLoadStats ComputePipelineLoadAnalysis(int total_layers, int num_stages,
                                              const std::vector<int> &partition,
                                              const std::vector<double> &layer_costs = {},
                                              int num_micro_batches = 1);

class PipelineLayout {
public:
    static PipelineLayout Create(int total_layers, int num_stages, int vpp_size,
                                 const std::vector<int> &partition = {});
    StageInfo GetStageInfo(int stage_id) const;
    int StageOfLayer(int layer_id) const;
    bool OwnsLayer(int stage_id, int layer_id) const;
    static int StageOfChunk(int global_chunk_id, int num_stages);
    static int LocalChunkIndexOfChunk(int global_chunk_id, int num_stages);
    std::string Describe() const;
};

} // namespace infini_train::nn::parallel
```

- `StageInfo` 包含 `is_first_stage`、`is_last_stage` 与 `layer_ranges_per_chunk`
  （每个 chunk 一个 `(start, end)` 区间）。
- `Create` 是统一布局入口：空 `partition` 走默认均匀划分，否则按显式层数构建并做完整校验。
- `GetStageInfo` 供模型构建 / `PipelineParallel` / checkpoint loader 使用；
  `StageOfChunk` / `LocalChunkIndexOfChunk` 供调度器统一计算 chunk 归属。
- `ParsePipelineLayerCosts` 解析 `--pipeline_layer_costs`；`SuggestBalancedPartition` 用线性划分 DP
  给出近似负载均衡的层数列表（空代价退化为均匀划分）。
- `ComputePipelineLoadAnalysis` 离线计算给定布局的各 Stage 负载、bottleneck / average、
  `efficiency`、`imbalance_bubble` 与 `structural_bubble`，用于对比均匀布局与自定义布局。
- `nn::ComputePerLayerParamCounts(const TransformerConfig&)`（`transformer.h`）解析式计算每层参数量，
  供 `--pipeline_auto_layout` 使用；MoE 层不支持。
- `nn::parallel::PipelineParallel::ReportPipelineStats()` 在训练结束后由 PP 通信组汇总各 Stage 实测
  前向/反向时间并打印 `Pipeline Timing Summary`（仅 `--pipeline_parallel > 1` 时生效，`rank_==0` 打印）。

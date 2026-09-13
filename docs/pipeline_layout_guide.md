# Pipeline 并行自定义布局使用说明

本文档描述 InfiniTrain 新增的 Pipeline 自定义布局能力：通过 `--pipeline_layer_partition`
显式指定每个 Pipeline Stage 的 Transformer 层数，或通过 `--pipeline_layer_costs` /
`--pipeline_auto_layout` 根据计算代价自动生成近似负载均衡的布局，并让模型构建、Pipeline 调度
与参数加载统一使用同一份 `PipelineLayout`，避免层归属逻辑在多处重复实现。

## 快速开始

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

上述配置把 24 个 Transformer 层依次划分为：

```text
stage 0: embedding + layers 0-3
stage 1: layers 4-11
stage 2: layers 12-17
stage 3: layers 18-23 + final_norm + lm_head
```

## 参数配置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--pipeline_parallel` | `1` | Pipeline Stage 数量 |
| `--virtual_pipeline_parallel` | `1` | 每个 Stage 的 virtual chunk 数量（vPP） |
| `--pipeline_layer_partition` | `""` | 逗号分隔的各 Stage 层数列表，例如 `4,8,6,6` |
| `--pipeline_layer_costs` | `""` | 逗号分隔的每层计算代价，用于自动生成负载均衡布局，例如 `1,2,1.5` |
| `--pipeline_auto_layout` | `false` | 按每层参数量自动生成负载均衡布局 |

GPT2 与 LLaMA3 示例入口均支持 `--pipeline_layer_partition`，解析后写入全局环境
`GlobalEnv`，模型构建、`PipelineParallel` 包装、调度器与 checkpoint 加载都从同一布局查询层归属。

## 布局语法

- 语法为逗号分隔的正整数列表，例如 `4,8,6,6`。
- 列表长度必须等于 `--pipeline_parallel` 的 Stage 数量。
- 各 Stage 层数之和必须等于模型总层数（GPT2-124M 为 12 层，需先选定层数与 Stage 数匹配的模型）。
- 布局是「连续划分」：`stage i` 拥有编号从 `sum(前 i 项)` 到 `sum(前 i+1 项)` 的连续 Transformer 层。
- Embedding 固定归属第一个 Stage，Final Norm + LM Head 固定归属最后一个 Stage（当前版本不开放单独配置）。

## 默认行为

不传 `--pipeline_layer_partition`（或传空串）时，保持原有自动均匀划分：

```text
layers_per_chunk = total_layers / (num_stages * vpp_size)
remainder        = total_layers % (num_stages * vpp_size)
```

余数按 global chunk 顺序依次多分配一层，vPP 下各 Stage 按
`global_chunk = local_chunk * num_stages + stage` 交错持有多个层范围。因此默认均匀布局与
vPP 完全兼容；自定义布局当前要求 `--virtual_pipeline_parallel 1`（二者不兼容，会报错）。

## 自动布局建议

除显式指定 `--pipeline_layer_partition` 外，还支持根据计算代价自动生成近似负载均衡的连续布局
（即「线性划分」问题：DP 最小化各 Stage 最大总代价）。三种代价来源：

1. **用户提供的计算代价**：`--pipeline_layer_costs "1,1,1,1,2,..."`，每个数对应一层的代价
   （参数量、实测耗时等均可）。列表长度即模型层数，结果会自动均衡各 Stage 总代价。
2. **各层参数量**：`--pipeline_auto_layout`，根据 `TransformerConfig` 解析式计算每层参数量
   （`ComputePerLayerParamCounts`，GPT-2 的 GELU+LayerNorm、LLaMA3 的 SwiGLU+RMSNorm+GQA 均精确支持；
   MoE 层暂不支持，需改用 `--pipeline_layer_costs`）。
3. **Profiler 统计**：先用 `--freq_generate_txt` / PROFILE_MODE 跑一次得到每层 kernel 实测耗时，再
   把每层耗时作为代价通过 `--pipeline_layer_costs` 传入，即可得到基于实测负载的布局建议。

```bash
# 代价不均衡（前 4 层轻、后 8 层重）时，PP=2 建议 7,5 而非均匀的 6,6
./build/infini_run --nproc_per_node=2 \
  ./build/gpt2 --device cuda --model d12 --pipeline_parallel 2 \
    --pipeline_layer_costs 1,1,1,1,2,2,2,2,2,2,2,2

# 或按每层参数量自动建议（GPT-2 / LLaMA3 各层结构相同，结果等价于均匀布局）
./build/infini_run --nproc_per_node=2 \
  ./build/gpt2 --device cuda --model d12 --pipeline_parallel 2 --pipeline_auto_layout
```

三种方式彼此互斥，且都不能与 `--pipeline_layer_partition` 同时使用。建议结果会在启动阶段打印为
`Auto-suggested pipeline layout ...: <partition>`；最终 `PipelineLayout` 仍按既有格式打印。核心算法见
`SuggestBalancedPartition`，空代价（`{}`）即退化为按层数均匀划分。

## Pipeline 负载分析（bubble / 各 Stage 执行时间 / 吞吐）

为证明「自定义布局能改善负载不均衡场景」，框架在运行结束时自动汇总一次 Pipeline 负载分析
（`--pipeline_parallel > 1` 时）：每个 PP rank 测量本 Stage 的前向 / 反向纯计算时间（CUDA 用
event 计时、CPU 用 `steady_clock`），经 PP 通信组 `AllGather` 汇总后由第一个 Stage 打印：

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

- **各 Stage 执行时间**：该 Stage 在所有 micro-batch 上前向 / 反向纯计算时间的累计（ms）。
- **Load-imbalance bubble**：`1 - average / bottleneck`，其中 `bottleneck = max_i(总时间_i)`、
  `average = mean_i(总时间_i)`；负载完全均衡时为 0。
- **pipeline efficiency**：`average / bottleneck`，即 `1 - bubble`。
- **结构 bubble**（fill/drain，GPipe）：`(S-1)/(S-1+n)`，与布局无关，由 `ComputePipelineLoadAnalysis`
  解析式给出。
- **吞吐**：沿用训练时每步打印的 `tok/s`（last rank 的 `step ... tok/s`）。

由于 GPT-2 / LLaMA3 各 Transformer 层结构相同，均匀按层数划分本身就是负载均衡的；真正体现
「自定义布局改善负载不均」的是各层计算量不一致的场景。此时用 `--pipeline_layer_costs` 给出每层
代价，`SuggestBalancedPartition` 给出均衡布局，`ComputePipelineLoadAnalysis` 可离线预测两种布局的
对比：

```cpp
std::vector<double> costs{1,1,1,1, 2,2,2,2, 2,2,2,2}; // 前 4 层轻、后 8 层重
auto uniform  = nn::parallel::ComputePipelineLoadAnalysis(12, 2, {6, 6}, costs, 8);
auto balanced = nn::parallel::ComputePipelineLoadAnalysis(12, 2, {7, 5}, costs, 8);
// uniform:  bottleneck=12, bubble=16.7%
// balanced: bottleneck=10, bubble=0%
```

### 离线对比演示（无需 GPU）

`docs/pipeline_layout_demo.cc` 是可直接运行的纯 CPU 演示程序，用上面的
`ComputePipelineLoadAnalysis` / `SuggestBalancedPartition` 打印「默认均匀布局 vs 自定义布局」的
完整对比表（各 Stage 负载、bubble、效率、吞吐）：

```bash
g++ -std=c++20 -DGLOG_USE_GLOG_EXPORT -I. -Ithird_party/glog/src -Ibuild/third_party/glog \
    docs/pipeline_layout_demo.cc infini_train/src/nn/parallel/pp/pipeline_layout.cc \
    -Lbuild/third_party/glog -lglog -pthread -o build/pipeline_layout_demo
LD_LIBRARY_PATH=build/third_party/glog ./build/pipeline_layout_demo
```

输出（负载不均模型，前 4 层轻、后 8 层重，S=2、n=8）：

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

结论：负载不均时，自定义均衡布局 `7,5` 把 bottleneck 从 12 降到 10，imbalance bubble 从 16.7% 降到 0，
吞吐提升 **1.20x**；各层均匀时二者等价（speedup 1.00x）。

### 真实 CUDA 计时（需 ≥2 张 GPU）

真实运行对比实验：用 `--pipeline_layer_costs` 生成的均衡布局与默认均匀布局各跑一次，比较输出末尾的
`Pipeline Timing Summary`（各 Stage 时间、bubble）与每步的 `tok/s`（吞吐）。负载不均场景下，均衡
布局的 bubble 更小、`tok/s` 更高。

> **注意（NCCL 硬限制）**：Pipeline 并行通过 NCCL 通信，而 NCCL 要求同一 communicator 里每个 rank
> 使用**互不相同的物理 GPU**（同一 GPU 被多个 rank 复用时 `ncclCommInitRank` 直接返回
> `invalid usage`）。因此单卡机器上无法运行 `--pipeline_parallel > 1` 的 CUDA 计时，需要至少 2 张
> 显存足够的 GPU。真机示例（默认均匀 `6,6` vs 自定义 `7,5`，负载不均代价见上）：

```bash
# 默认均匀布局
./build/infini_run --nproc_per_node=2 ./build/gpt2 \
  --model d12 --input_bin data/tiny_shakespeare_train.bin --pipeline_parallel 2 \
  --total_batch_size 2048 --num_iteration 10 --freq_generate_txt 1000

# 自定义均衡布局（根据代价自动建议 7,5）
./build/infini_run --nproc_per_node=2 ./build/gpt2 \
  --model d12 --input_bin data/tiny_shakespeare_train.bin --pipeline_parallel 2 \
  --pipeline_layer_costs 1,1,1,1,2,2,2,2,2,2,2,2 \
  --total_batch_size 2048 --num_iteration 10 --freq_generate_txt 1000
```

## 输入输出示例

启动时若 `--pipeline_parallel > 1`，Stage 0 会打印最终布局：

```text
PipelineLayout: num_stages=4, total_layers=24, vpp=1
  stage 0: [0, 4) + embedding
  stage 1: [4, 12)
  stage 2: [12, 18)
  stage 3: [18, 24) + final_norm + lm_head
```

其中 `[start, end)` 表示本 Stage 持有 `start`（含）到 `end`（不含）的 Transformer 层区间；
每个 Stage 在 vPP 下可能打印多个区间。

## 错误排查

以下非法配置会在启动阶段直接 `LOG(FATAL)` 终止，并给出可定位的报错信息：

| 场景 | 触发条件 | 报错关键字 |
| --- | --- | --- |
| 空项或非数字 | `4,,6,6` / `4,8a,6,6` | `not a positive integer` |
| 非正层数 | `4,0,6,6` | `must be positive` |
| Stage 数量不符 | `--pipeline_parallel 4` 但列表只有 3 项 | `entries but pipeline_parallel is` |
| 层数总和错误 | 24 层模型但 `4,8,6,5` | `sums to` |
| 与 vPP 冲突 | 自定义布局 + `--virtual_pipeline_parallel 2` | `incompatible with virtual_pipeline_parallel` |
| 代价为空项/非数字 | `1,,2` / `1,abc` | `empty entry` / `not a number` |
| 代价非负有限 | `-1,2` / `inf` | `non-negative` / `finite` |
| 代价条数与层数不符 | 12 层模型但代价只有 5 项 | `sums to` |
| 三种布局来源同时使用 | `--pipeline_layer_partition` 与 `--pipeline_layer_costs` / `--pipeline_auto_layout` 同时出现 | `cannot be combined with` |
| 代价与自动布局同时使用 | `--pipeline_layer_costs` 与 `--pipeline_auto_layout` 同时出现 | `mutually exclusive` |

若报「模型构建与参数加载层归属不一致」，通常是因为某个调用点仍在使用旧的均匀划分：请确认
模型构建（`TransformerModel`）、`PipelineParallel` 包装、两个 checkpoint loader 都改为查询
`PipelineLayout` / `StageInfo`，且 `GetPipelineLayerPartition()` 已正确传入 `InitAllEnv`。

## 测试

单元测试位于 `tests/distributed/test_pipeline_layout.cc`，覆盖默认均匀划分、自定义 `4,8,6,6`、
Embedding/FinalNorm/LMHead 归属、`StageOfLayer`/`OwnsLayer`、vPP 交错、chunk↔stage 映射以及
各类非法配置的死亡断言。`tests/distributed/test_pipeline_layout_suggest.cc` 额外覆盖
`SuggestBalancedPartition`（均匀/余数/代价不均衡/非法代价）、`ParsePipelineLayerCosts`
（合法解析/负值/非数字/空项/无穷）、`ComputePerLayerParamCounts`（GELU+LayerNorm 精确值、
SwiGLU+RMSNorm 均匀正数、MoE 拒绝）以及 `ComputePipelineLoadAnalysis`（均匀即均衡、代价不均衡
下均匀布局 skewed、均衡布局消除 bubble、空 partition 默认均匀、结构 bubble 公式）。

```bash
cmake -S . -B build -DBUILD_TEST=ON
cmake --build build -j
ctest --test-dir build -R 'test_pipeline_layout' --output-on-failure
```

端到端验证需至少 2 个 Pipeline Stage：用相同初始权重分别以单卡/默认布局与自定义布局跑若干
训练迭代，比较前向结果、loss 与梯度在允许误差内一致（fp32 1e-05，bf16 1e-02），并确认训练
过程无通信死锁。

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

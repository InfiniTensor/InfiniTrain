# Pipeline 自定义布局

InfiniTrain 的 GPT-2 和 LLaMA 3 示例支持用 `--pipeline_layer_partition` 指定每个物理 Pipeline Stage
拥有的连续 Transformer 层数。模型构建、PP Stage 构造和 LLMC 参数加载均使用同一个
`PipelineLayout`。

## 参数与语法

```bash
./gpt2 \
  --pipeline_parallel 4 \
  --virtual_pipeline_parallel 1 \
  --pipeline_layer_partition 4,8,6,6 \
  [其他训练参数]
```

该模型必须有 24 层，最终布局为：

```text
stage 0: embedding + layers [0, 4)
stage 1: layers [4, 12)
stage 2: layers [12, 18)
stage 3: layers [18, 24) + final_norm + lm_head
```

列表项必须是正整数；项数必须等于 `--pipeline_parallel`，总和必须等于 checkpoint 或配置中的
模型层数。空格可以出现在数字两侧。GPT-2 和 LLaMA 3 使用相同参数。

## 按逐层代价自动均衡

如果已经通过 profiler、FLOPs 估算或经验权重得到每个 Transformer 层的相对代价，可以让 InfiniTrain
自动生成连续分区：

```bash
./gpt2 \
  --pipeline_parallel 2 \
  --pipeline_layer_costs 10,1,1,1,1,1 \
  [其他训练参数]
```

上述 6 层模型会生成 `1,5`：stage 0 的建模代价为 10，stage 1 为 5。均匀 `3,3` 的代价为
12 和 3，因此最慢 Stage 的建模代价从 12 降到 10。自动布局保持层连续、顺序不变，并保证每个
Stage 至少拥有一层。

代价项必须是有限正数，项数必须和模型 Transformer 层数完全一致。
`--pipeline_layer_costs` 与 `--pipeline_layer_partition` 互斥，且当前同样要求
`--virtual_pipeline_parallel=1`。程序会在模型构建前打印自动生成的最终布局。

## 默认行为与 vPP

不传 `--pipeline_layer_partition` 时，保持原有均匀划分。余数从执行顺序靠前的 chunk 开始各多分一层；
`--virtual_pipeline_parallel` 的默认轮转 chunk 布局也保持不变。

当前自定义层数列表只描述物理 Stage，不描述虚拟 chunk，因此它与
`--virtual_pipeline_parallel` 大于 1 不兼容，程序会在创建模型前报错。Embedding 固定属于 stage 0，
Final Norm 和 LM Head 固定属于最后一个 stage，并显式记录在布局查询接口和启动日志中。

## 任意 vPP Chunk 映射

使用有序 STAGE:LAYER_COUNT 列表显式指定 Chunk owner：

    --pipeline_parallel=2 --virtual_pipeline_parallel=2 \
    --pipeline_chunk_layout=0:3,1:3,1:3,0:3

这表示逻辑 Chunk owner 为 [0,1,1,0]，层范围为 [0,3)、[3,6)、[6,9)、[9,12)。每个物理 Stage
必须获得相同的正数 Chunk；连续 Chunk 可以属于同一 Stage，此时直接保留本地 autograd 图。
Embedding 归属第一个逻辑 Chunk，Final Norm/LM Head 归属最后一个逻辑 Chunk。

## Megatron 风格表达式

pipeline_model_parallel_layout 支持 E（Embedding）、t（Transformer）、N（Final Norm）、L（LM Head）、
| 分隔符、x*n 和 (expr)*n 重复，以及相邻 || 空 Chunk。例如：

    --pipeline_parallel=2 --virtual_pipeline_parallel=2 \
    --pipeline_model_parallel_layout='Et*3||t*3|t*6NL'

表达式必须展开为 PP*vPP 个 Chunk；E/L 必须各出现一次且位于整体首尾，N 最多一次并与 L 同属末 Chunk，
t 数量必须等于模型层数。

## 自动布局建议

建议工具支持逐层参数量、用户代价和 PROFILE_MODE 记录：

    scripts/suggest_pipeline_layout.py \
      --profiler-records gpt2.records.log.rank0 \
      --profiler-warmup-samples=1 --pipeline-parallel=2 --microbatches=4

工具输出可直接复制的 pipeline_layer_partition、每 Stage 代价、均匀布局对比和理论 bubble。默认丢弃
每层第一个 profiler 样本，避免 CUDA warmup 污染。

## 启动输出

主 rank 会输出规范化后的最终布局，例如：

```text
Pipeline layout (24 layers, 4 stages):
  stage 0: embedding layers[0,4)
  stage 1: layers[4,12)
  stage 2: layers[12,18)
  stage 3: layers[18,24) final_norm lm_head
```

## 错误排查

- `has N entries, but --pipeline_parallel is M`：列表项数量和 PP stage 数不一致。
- `sums to N layers, but the model has M`：列表总层数和模型配置或 checkpoint 不一致。
- `entries must be positive integers`：存在零、负数或非整数。
- `contains an empty stage entry`：存在连续逗号、开头逗号或末尾逗号。
- `incompatible with --virtual_pipeline_parallel != 1`：自定义物理布局和 vPP 同时启用。
- `must contain exactly N entries`：逐层代价数量和模型层数不一致。
- `costs must be finite positive numbers`：逐层代价包含零、负数、NaN、无穷或非数字。
- `cannot be used together`：同时指定了手工分区和自动均衡代价。

## C++ 查询接口

`PipelineLayout::layer_ranges(stage_id)` 返回该 Stage 的半开层范围；
`stage_for_layer(layer_id)` 执行反向查询；`owns_embedding`、`owns_final_norm` 和 `owns_lm_head`
用于特殊模块归属判断。`PipelineParallel::GetStageInfo` 是面向现有调度代码的兼容投影。

## 并行组合与限制

| 组合 | 默认均匀布局 | 手工层数分区 | 任意 Chunk / Megatron 布局 |
| --- | --- | --- | --- |
| PP | 支持 | 支持 | 支持 |
| PP + DDP | 支持 | 支持 | 支持 |
| PP + TP | 支持 | 支持 | 支持 |
| PP + DDP + TP | 支持 | 支持 | 支持 |
| vPP | 默认轮转映射 | 拒绝物理分区参数 | 支持显式 Chunk owner |

`||` 表示空逻辑 Chunk；它不表示物理 Stage 没有 Chunk。当前调度器要求每个物理 Stage 拥有相同数量的
正数 Chunk，因此会拒绝 Chunk 数不平衡的映射。

提交前建议依次运行 CPU 单元测试、双 GPU E2E 和稳定多轮性能 benchmark。

## 梯度一致性调试

GPT-2 示例提供可选的 `--dump_gradients=DIR` 验证参数。它在第一次优化迭代后导出所有非空参数梯度，
并把 PP rank 的局部层号转换为全局层号，使单卡与自定义 PP 输出可以直接比较：

```bash
python3 scripts/precision_check/precision_compare.py \
  --dir1 /tmp/gpt2-grad-single \
  --dir2 /tmp/gpt2-grad-custom \
  --atol 1e-5 --rtol 0
```

该参数只用于正确性验证；导出会将梯度同步复制到 CPU，不应在性能测试中启用。

## 端到端回归

仓库提供双卡 GPT-2 E2E 脚本。它会自动运行单卡基线和两阶段代价布局，检查最终布局、fp32 loss、
梯度文件集合以及逐参数梯度误差：

~~~bash
tests/distributed/test_pipeline_layout_e2e.sh \
  /path/to/cuda-build \
  data/gpt2/tiny_shakespeare_train.bin \
  data/gpt2/gpt2_124M.bin \
  0,1
~~~

该测试需要 CUDA/NCCL 构建、两张 GPU、NumPy 以及 GPT-2 124M LLMC checkpoint。测试使用临时目录，
退出时自动清理梯度和日志。

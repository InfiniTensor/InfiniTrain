# Context Parallel 使用指南

Context Parallel（CP）将同一个样本的序列维度切分到多个 rank 上执行，各 rank 通过 attention 通信与局部计算，协同完成全局序列上的计算，结果与不使用 CP 时保持数学等价。

CP 与 DP 的区别在于：DP 的各 rank 处理不同样本，CP 的各 rank 处理同一样本的不同 token 区间。CP 适用于长序列训练，可以降低单卡的序列激活和 attention 计算开销。

GPT-2、LLaMA 3 和 Qwen3-8B 示例支持 CP。CP 可与 TP、SP、PP、vPP、DDP 以及 ZeRO Stage 1/2 组合使用。

## 快速开始

以下命令在单节点启动 4-way CP。总 world size 为 4，TP/PP 均为 1，因此 DP 为 1：

```bash
./build/infini_run \
  --nnodes=1 \
  --nproc_per_node=4 \
  ./build/llama3 \
    --device cuda \
    --input_bin data/llama3/tiny_shakespeare_train.bin \
    --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
    --num_iteration 10 \
    --context_parallel 4 \
    --cp_comm_type p2p \
    --attention_cp_layout zigzag
```

CP 大于 1 的并行训练要求加速器后端。CUDA 构建需要同时启用 CUDA 与 NCCL。

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--context_parallel` | `1` | CP rank 数，必须为正整数。 |
| `--cp_comm_type` | `p2p` | CP attention 通信策略，可选 `all_gather`、`p2p`、`a2a`、`a2a+p2p`。 |
| `--attention_cp_layout` | `zigzag` | 序列切分布局，可选 `contiguous`、`zigzag`。 |
| `--hierarchical_context_parallel_sizes` | 空 | 仅 `a2a+p2p` 可用，格式为 `A,P`，且必须满足 `A × P = context_parallel`。 |

gflags 支持下划线和连字符两种参数名形式，例如 `--context_parallel` 与 `--context-parallel` 等价。

## 拓扑与约束

总 world size 由启动配置决定：

```text
world_size = nnodes × nproc_per_node × nthread_per_process
```

并行 rank 轴的生成顺序为：

```text
TP -> CP -> EP -> DP -> PP
```

dense 模型不使用 EP 时，实际顺序为 `TP -> CP -> DP -> PP`。SP 与 TP 绑定，作用在 TP rank 组内。

DP 规模从剩余 rank 中推导：

```text
data_parallel_size =
    world_size / (tensor_parallel × context_parallel × pipeline_parallel)
```

配置需要满足以下约束：

- `world_size` 必须能被 `tensor_parallel × context_parallel × pipeline_parallel` 整除。
- `contiguous` 布局下，`sequence_length` 必须能被 CP 规模整除。
- `zigzag` 布局下，`sequence_length` 必须能被 `2 × CP` 整除；每个 CP rank 持有长度相等的前半块和后半块。
- 同时启用 SP 时，CP 切分后的本地 sequence length 必须能被 TP 规模整除。
- `a2a` 要求 TP 切分后的本地 query heads 与 KV heads 满足 CP 切分的整除约束。
- `a2a+p2p` 的层级配置必须包含且仅包含两个正整数，并满足 `A × P = CP`；A2A 子组内的本地 query/KV heads 必须满足 A 维度的整除约束。

例如，`world_size=8`、`TP=2`、`CP=2`、`PP=2` 时，DP 为 1。若还启用 SP，则 `sequence_length / 2` 也必须能被 2 整除。

## 序列布局

`contiguous` 布局将全局序列切成连续区间。以 CP=4、sequence length=16 为例：

| CP rank | 持有的全局 token 位置 |
| --- | --- |
| 0 | `[0, 1, 2, 3]` |
| 1 | `[4, 5, 6, 7]` |
| 2 | `[8, 9, 10, 11]` |
| 3 | `[12, 13, 14, 15]` |

`zigzag` 是默认布局。每个 rank 从序列头部和尾部各取一块，使 causal attention 在不同 CP rank 间的计算量更均衡：

| CP rank | 持有的全局 token 位置 |
| --- | --- |
| 0 | `[0, 1, 14, 15]` |
| 1 | `[2, 3, 12, 13]` |
| 2 | `[4, 5, 10, 11]` |
| 3 | `[6, 7, 8, 9]` |

布局只改变 token 的归属和 causal mask 中的全局位置，不改变模型配置里的全局 sequence length。

## 通信策略

| `--cp_comm_type` | 机制 | 适用性 |
| --- | --- | --- |
| `all_gather` | 每个 rank 收集所有 CP rank 的 KV 分片，获得完整序列的 K/V，再使用本地 Q 计算对应的 attention 输出。 | 实现路径简单，但需要完整 KV 的通信与显存开销。 |
| `p2p` | CP rank 之间按环传递 KV 分片，即 Ring Attention。各 rank 在接收 KV 分片的同时计算本地 Q 对应的 attention。 | 默认策略，通常更适合长序列。 |
| `a2a` | 通过 All-to-All 在 CP rank 间转换 sequence 与 attention head 的分布，使每个 rank 获得完整序列上的部分 attention heads，再执行本地 attention 计算。 | 适用于 head 数满足切分约束、希望采用 head 维度分布式计算的场景。 |
| `a2a+p2p` | 先在 A2A 子组内进行 sequence/head 布局转换，再在不同子组之间通过 P2P 传递 KV 分片。 | 两级 CP 拓扑，用 `A,P` 控制子组形状。 |

### 通信过程说明

**All-Gather**

每个 rank 收集所有 CP rank 的 K/V 分片，得到完整序列的 K/V。随后使用本地 Q 计算 attention 输出，输出仍对应当前 rank 持有的 query token。

**P2P（Ring Attention）**

各 rank 持有本地 Q、K、V 分片。计算过程中，KV 分片按照环形拓扑在 rank 之间传递，每个 rank 逐步使用接收到的 KV 分片计算本地 Q 对应的 attention，并累积 attention 结果。

该方式不需要在单个 rank 上保存完整的 KV，适合长序列场景。

**A2A**

A2A 通过 All-to-All 转换 sequence 与 attention head 的分布：

- 通信前：每个 rank 持有部分 sequence 和完整的本地 attention heads。
- 通信后：每个 rank 持有完整 sequence 和部分 attention heads。
- 随后执行本地 attention 计算。

该过程将 sequence 维度的切分转换为 head 维度的切分。具体 Q/K/V 的布局和输出恢复方式由实现决定。

**A2A+P2P**

该策略采用两级通信：

- A2A：在子组内部进行 sequence/head 布局转换。
- P2P：在不同 A2A 子组之间传递 KV 分片。

层级模式示例：

```bash
--context_parallel 8 \
--cp_comm_type a2a+p2p \
--hierarchical_context_parallel_sizes 2,4
```

此时 8 个 CP rank 会组成 4 个大小为 2 的 A2A 子组，并在 4 个子组之间使用 P2P 传递 KV。

`2,4` 与 `4,2` 都是合法配置，但表示不同的子组形状，通信拓扑和 head 切分方式不同。

## 与其他功能的组合

- **TP 与 SP**：CP 首先在 CP rank 之间切分全局 sequence；启用 SP 时，再在 TP rank 之间切分 CP-local sequence。因此，SP 的整除约束应基于 CP 切分后的本地序列长度计算。
- **PP/vPP**：CP 可与 PP/vPP 组合，模型切分后的每个 pipeline stage 都在其 CP rank 组内处理本地序列分片。
- **ZeRO Stage 1/2**：具体分片通信域取决于实现。如果当前实现将 DP 与 CP rank 纳入同一分片通信域，则需要按照对应通信组计算分片规模，并满足相应的启用条件。
- **Checkpoint**：checkpoint 会记录 `cp_size`，保存时根据 CP 坐标选择合适的 writer rank，避免同一模型分片被重复写出。
- **LoRA**：LoRA 保存同样需要将 CP 坐标纳入 writer 选择；加载时根据 checkpoint 中保存的完整 adapter tensor，由当前并行层按照目标拓扑进行切分。

## 测试

CP 布局与 causal mask 测试的 CTest 名称前缀为 `ContextParallelLayout`，label 为 `cpu`。注意 `ctest -R` 匹配测试名，`ctest -L` 才匹配 label。

在仓库根目录执行：

```bash
cmake --build build --target test_context_parallel -j
(cd build && ctest -R '^ContextParallelLayout\\.' --output-on-failure)
```

也可以直接运行测试二进制：

```bash
./build/tests/parallel/test_context_parallel \\
  --context-parallel=4 \\
  --attention-cp-layout=zigzag \\
  --gtest_filter='ContextParallelLayoutTest.*'
```

在启用 CUDA 与 NCCL 的构建中，CTest 还会注册多 GPU attention 前反向测试：

```bash
(cd build && ctest -L cuda -R '^ContextParallelCUDA\\.' --output-on-failure)
```
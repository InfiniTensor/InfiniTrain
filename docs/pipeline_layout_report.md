# Pipeline Layout 实现报告

## 数据结构与接口

`nn::parallel::PipelineLayout` 是 Pipeline 层归属的统一数据源。它保存物理 Stage 数、模型总层数、
每个 Stage 的半开层范围，并提供以下查询：

- `layer_ranges(stage_id)`：查询本 Stage 的一个或多个连续 chunk 范围。
- `stage_for_layer(layer_id)`：从全局 Transformer 层号反查物理 Stage。
- `owns_embedding/final_norm/lm_head(stage_id)`：查询特殊模块归属。
- `ToString()`：输出启动时使用的规范化布局。

布局存储为 `thread_local`。InfiniTrain 支持一个进程创建多个训练线程，每个线程代表独立 global rank；
线程本地存储可保证不同 PP rank 构建和加载时共享本线程的一份布局，同时不产生跨线程数据竞争。

## 关键实现

`PipelineLayout::Parse` 将 `4,8,6,6` 转换为按执行顺序连续且不重叠的范围。解析时一次性校验
Stage 数、正整数、总层数和 vPP 兼容性。`Uniform` 封装原有默认均匀算法，并保留 vPP 的
`global_chunk = local_chunk * pp_size + stage` 轮转语义。

`PipelineLayout::FromLayerCosts` 接收每层有限正代价，通过动态规划在所有非空连续分区中最小化
最大 Stage 总代价。状态为前 `i` 层分到 `s` 个 Stage 时的最优最大代价，转移枚举最后一个 Stage
的起点；时间复杂度为 `O(S * L^2)`，空间复杂度为 `O(S * L)`。该方法适合模型启动阶段，结果
确定且不改变层的执行顺序。`ResolvePipelineLayout` 统一选择手工分区、代价均衡或默认均匀布局，
并拒绝多个布局来源同时生效。

GPT-2 和 LLaMA 3 在模型配置确定后设置布局。对于 LLMC checkpoint，布局在读取 header 中真实
`n_layer` 后解析。`TransformerModel` 用布局创建本 rank 的层和特殊模块；`TransformerConfig::GetChunkSize`
和 `PipelineParallel` 用相同布局构造调度 Stage；两个 checkpoint loader 用布局筛选本 rank 权重。

自定义物理分区当前要求 `virtual_pipeline_parallel=1`。现有调度器对 vPP 使用固定轮转
`Chunk -> Stage` 映射，层数列表无法无歧义表达虚拟 chunk；启动时拒绝该组合比隐式产生错误执行顺序更安全。
未配置自定义参数时仍走 `Uniform`，因此 GPipe、1F1B/vPP、TP 和 DDP 的既有入口保持不变。

优秀项实现取消了 vPP 固定轮转限制：布局保存有序逻辑 Chunk 的 owner、local index 和层范围，调度器
不再使用 global_chunk % pp_size 推导 Stage。Megatron 风格解析支持 E/t/N/L、|、重复表达式和空 Chunk，
最终仍投影到同一 PipelineLayout 查询接口。

## 正确性与测试

本次验证范围如下：

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| GPT-2 / LLaMA3 PP + DDP/TP 接口接入 | 已完成 | 模型构建和 checkpoint loader 查询同一 PipelineLayout |
| 双卡 GPT-2 PP E2E | 已实测 | H200，loss、梯度与单卡一致 |
| 任意 vPP Chunk owner | 已实测 | H200，owner `[0,1,1,0]` 无死锁 |
| Megatron 风格布局 | 已实测 | H200，包含空逻辑 Chunk |

DDP/TP 组合保留既有 InfiniTrain 并行入口；本次新增回归重点是布局解析、PP 调度、参数加载和
跨布局数值一致性。若提交环境要求完整 DP×TP×PP 组合矩阵，应在目标集群补跑对应资源规模的回归。

CPU 单元测试覆盖 `4,8,6,6`、完整 layer-to-stage 反查、特殊模块、默认 vPP 轮转，以及错误的
Stage 数、总和、负数、零、空项、越界查询和自定义布局/vPP 冲突。验证命令：

```bash
cmake -S . -B /tmp/infinitrain-pipeline-build \
  -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF -DUSE_OMP=OFF
cmake --build /tmp/infinitrain-pipeline-build --target test_pipeline_layout gpt2 llama3 -j2
ctest --test-dir /tmp/infinitrain-pipeline-build -R PipelineLayoutTest --output-on-failure
```

结果：10/10 布局与建议测试通过，CPU 全量测试通过，GPT-2、LLaMA3 和 Mixtral 目标编译、
链接通过。CUDA 13.0/NCCL 构建后，在两张 H200 上完成 GPT-2 124M 自定义 `4,8` 两阶段训练，
两步 loss 为 `5.250158`、`4.913960`，无通信死锁。同参数单 GPU loss 完全一致；默认 `6,6` PP
第二步 loss 为 `4.913958`，最大打印差值 `2e-6`，满足 fp32 `1e-5` 容差。逐参数梯度自动 diff
使用规范化全局参数名比较单 GPU 和自定义 PP 的 149 个梯度；`atol=1e-5, rtol=0` 下
149/149 通过且无缺失文件。完整命令与日志见 `docs/pipeline_layout_test_log.md`。
仓库中的 `tests/distributed/test_pipeline_layout_e2e.sh` 将双卡启动、布局断言、loss 比较、梯度文件
集合比较和逐参数数值比较固化为一个非零失败的自动化入口；由于依赖两张 GPU 和外部模型资产，普通
CPU `ctest` 不会默认注册该用例。

## 负载分析方法

默认均匀布局只平衡层数。对已知重层或显存热点，先记录各层 forward/backward 时间或峰值显存，
再调整每 Stage 层数，使各 Stage 总代价接近。比较时固定模型、batch、microbatch 和 dtype，分别记录
稳定迭代的 Stage 时间、整步吞吐与峰值显存。理论 bubble 由 microbatch 数和 Stage 数主导；自定义布局
主要通过降低最慢 Stage 的执行时间改善有效吞吐，并不改变相同调度下的 bubble step 数。

例如逐层代价 `10,1,1,1,1,1` 在两个 Stage 上，默认均匀 `3,3` 的 Stage 代价为 `12,3`；
自动布局生成 `1,5`，Stage 代价为 `10,5`，最大建模代价下降 16.7%。这是代价模型上的上界改善，
实际吞吐还取决于通信、特殊模块、microbatch 数和运行时噪声，应使用稳定多轮 profiler 数据复测。

两张 H200、GPT-2 124M、4 个 microbatch、12 个训练迭代，去掉首 3 步 warmup 后：

| 布局 | 平均 step | 平均吞吐 | 较高 Stage 峰值显存 |
| --- | ---: | ---: | ---: |
| 默认 6,6 | 89.532 ms | 11,437 tok/s | 1473 MB |
| Profiler 建议 7,5 | 81.799 ms | 12,519 tok/s | 1343 MB |

建议布局实测吞吐提升 9.45%，峰值显存降低 130 MB；理论 bubble（4 microbatch、2 Stage）为 20%。
Profiler 输入为 3 步 PROFILE_MODE 记录，每层丢弃一个 warmup 样本后得到 7,5，模型代价上界下降 6.84%。

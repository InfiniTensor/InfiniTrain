# 【训练方向 2026 夏季训练营】Pipeline 并行自定义布局

# **一、项目背景**

Pipeline Parallelism（PP）通过将模型的不同层划分到多个 Pipeline Stage 上，使超出单卡显存容量的大模型能够进行分布式训练。当前 InfiniTrain 已支持 GPipe、1F1B 和 Virtual Pipeline Parallelism（vPP），并能够按照 Pipeline Stage 数量自动划分 Transformer 层。

当前层划分策略主要采用均匀分配方式，并默认将 Embedding 放置在第一个 Stage、Final Norm 和 LM Head 放置在最后一个 Stage。该方式适合结构规则、各层计算量接近的模型，但在以下场景中存在限制：

- 不同 Transformer 层的计算量或显存占用不一致，均匀按层数划分不能实现负载均衡。

- Embedding、Final Norm、LM Head 等特殊模块的计算量没有纳入布局配置。

- 用户无法显式指定每个 Stage 所包含的层范围。

- 模型构建、Pipeline 调度和参数加载分别计算层归属，扩展自定义布局时容易产生不一致。

为此，本项目旨在为 InfiniTrain 增加统一的 **Pipeline 自定义布局（Pipeline Layout）** 能力，使用户可以显式配置各 Stage 的 Transformer 层和首尾特殊模块，并保证模型构建、训练调度及参数加载使用同一份布局信息。

# 二、项目目标

- 设计统一的 \`PipelineLayout\` 数据结构，描述每个 Pipeline Stage 所拥有的 Transformer 层及特殊模块。

- 支持 Transformer 层在不同 Stage 之间进行非均匀但连续的划分。

- 支持显式配置 Embedding、Final Norm 和 LM Head 的归属 Stage。

- 未指定自定义布局时，保持当前自动均匀划分行为和已有命令行参数兼容。

- 对布局进行完整合法性校验，并在训练启动阶段输出清晰的布局信息和错误提示。

- 使模型构建、Pipeline 调度和参数加载统一查询 \`PipelineLayout\`，避免层归属逻辑重复。

# **三、任务拆解**

## pipeline Layout 数据结构设计

定义用于表达 Pipeline 布局的数据结构及查询接口。具体命名和组织形式可结合现有代码设计，应该能表述：

- Pipeline Stage 数量

- 每个 Stage 对应的 Transformer 层范围

- Embedding、Final Norm、LM Head 等特殊模块的归属

- 根据 \`stage\_id\` 查询本 Stage 所拥有的层和特殊模块

- 根据 \`layer\_id\` 查询对应的 Stage

布局信息应作为模型构建、Pipeline Stage 构造及参数加载的统一数据来源，不应在不同模块中重复实现层划分算法。

## 命令行参数扩展

新增必要参数，用于指定自定义 Pipeline 布局。基础实现可采用清晰、易校验的层数列表形式，例如：

```Bash
--pipeline_parallel 4 \
--pipeline_layer_partition 4,8,6,6
```

上述配置表示 24 个 Transformer 层依次划分为：

```Plain Text
stage 0: embedding + layers 0-3
stage 1: layers 4-11
stage 2: layers 12-17
stage 3: layers 18-23 + final_norm + lm_head
```

参数名称和具体语法可在设计阶段调整，但需要满足：未指定参数时默认使用当前均匀划分策略；指定参数时，各 Stage 层数之和必须等于模型总层数。

## 布局解析与合法性校验

在训练启动阶段完成布局解析和校验，至少覆盖以下情况：

- Stage 数量与 `pipeline_parallel` 配置一致。

- Transformer 层不重复、不遗漏，并保持正确的执行顺序。

- 布局与 Virtual Pipeline 配置不兼容时，给出明确错误。

程序启动后支持打印输出各 Stage 的最终布局，便于用户检查配置并定位问题。

## 模型构建与 Pipeline Stage 集成

修改 Pipeline 模型构建流程，使每个 rank 仅创建当前 Stage 所拥有的模块：

- 根据 `PipelineLayout` 构建本地 Transformer 层

- 如果没显示指定`PipelineLayout` 保留当前自动均匀布局作为默认实现

- 完成 GPT\-2 和 LLaMA 3 示例模型的接入

- 支持和DDP、TP等多种并行模型组合运行。

## Pipeline 调度与参数加载集成

Pipeline 调度器应从统一布局中获得 Stage 和 Chunk 的归属信息，不再仅依靠固定的取模关系推导所有权。模型参数加载流程应根据同一份 `PipelineLayout` 判断本 rank 需要加载的 Transformer 层和特殊模块。默认均匀布局、自定义非均匀布局应使用相同的查询接口。

## 测试与验证

新增单元测试和端到端测试，至少覆盖：

- 自定义非均匀布局，例如 `4,8,6,6`

- Embedding、Final Norm 和 LM Head 的归属正确

- Transformer 层无重复、无遗漏且执行顺序正确

- 层数总和错误、Stage 数量错误、负数等非法配置能够在启动阶段被拒绝

- 至少使用 2 个 Pipeline Stage 完成 GPT\-2 或 LLaMA 3 的若干训练迭代，训练过程无通信死锁

- 相同初始参数和输入下，自定义 PP 布局与单卡或默认布局的前向结果、loss 和梯度在允许误差范围内一致

# **四、评判标准**

请提供以下内容：

- Pipeline 自定义布局使用指导文档，包括参数配置、布局语法、默认行为、输入输出示例和错误排查方法。

- 单元测试、端到端测试代码及测试日志。

- 项目报告，主要包括数据结构与接口设计、关键实现说明、兼容性说明，以及不同布局下的正确性和 Pipeline 负载分析。

## **通过标准**

- 实现统一的 \`PipelineLayout\` 数据结构和必要的布局查询接口。

- 支持通过命令行配置各 Pipeline Stage 的非均匀连续层数，例如 \`4,8,6,6\`。

- 支持显式记录并正确放置 Embedding、Final Norm 和 LM Head。

- 未配置自定义布局时，现有均匀划分、GPipe、1F1B 及 vPP 使用方式不受影响。

- GPT\-2 和 LLaMA 3 的模型构建及参数加载使用统一布局判断层归属。

- 对非法布局进行完整校验，并输出可以定位问题的错误信息。

- 提供单元测试和至少一个 2\-Stage 端到端训练测试；与单卡或默认布局相比，**前向结果、loss 和梯度在允许误差范围内一致（fp32：1e\-05，bf16：1e\-02）**。

## **优秀标准**

在达到通过标准的基础上，可完成以下一项或多项：

- 支持 vPP 下显式配置任意 \`Chunk \-\> Stage\` 映射，而不是依赖固定轮转关系。

- 支持类似 Megatron\-LM 的 Pipeline Layout 字符串表达，可描述重复层、特殊模块、空 Stage 和 Virtual Pipeline Chunk。

- 支持根据各层参数量、Profiler 统计或用户提供的计算代价，自动生成近似负载均衡的布局建议。

- 给出默认均匀布局和自定义布局的 Pipeline bubble、各 Stage 执行时间及吞吐对比，证明自定义布局能够改善负载不均衡场景。

- 代码通过 **仓库 PR review 流程**（提交 → 审查 → 修改 → 达到可合入标准）。

# **五、参考资料**

1\. [Megatron\-LM Pipeline Parallelism](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/core/pipeline_parallel.md)

2\. [Megatron\-LM Pipeline Parallel Layout](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/pipeline_parallel_layout.md)

3\. [Megatron\-LM Parallelism Guide](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/parallelism-guide.md)

4\. [GPipe: Efficient Training of Giant Neural Networks using Pipeline](https://arxiv.org/abs/1811.06965) 

5\. [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377)




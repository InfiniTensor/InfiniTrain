# Pipeline 并行自定义布局实现报告

## 范围与结论

本次实现覆盖 Phase 1--7 的代码接入和 CPU 可执行校验，并提供 Phase 8 的端到端
验证脚本。除模型训练本身外，布局构造、模型裁剪、包装器组装、调度器查表、
checkpoint ownership、错误校验和启动日志均已落地。GPU 分布式训练与 loss 数值
对齐需要真实 CUDA、输入数据和 launcher，本环境未执行。

## 统一数据模型

`PipelineLayout` 保存模型层数、stage 数、vPP 数、`ChunkLayout` 列表、每个 stage
的 chunk 列表、特殊模块 placement 和 `LayerLocation` 反向索引。`BuildDefault` 保持
vPP 交错 chunk 编号；`BuildContiguous` 用 stage 层数列表生成连续范围。

所有查询（`stage_of_layer`、`chunk_of_layer`、`local_layer_index`、`owns`）都从该
对象读取。`Validate` 检查范围覆盖/不重叠、索引和特殊模块 stage；
`ValidateForCurrentPipelineTransport` 进一步检查当前 send/recv 假设下每个 stage
都有可执行 chunk。

## 组件接入

* CLI 在 GPT-2/LLaMA3 中解析 partition 和三个特殊模块 stage，并在模型配置确定后
  安装布局。
* `TransformerModel` 只构造当前 stage 所拥有的层和特殊模块；内部 alias 不污染
  `NamedParameters`，checkpoint 使用 canonical `transformer.lm_head.*` key。
* `PipelineParallel` 按 `stage.global_chunk_ids` 组装 chunk，保存有效的
  `wrapped_module`，并对 rank、stage 数、空 chunk 抛出明确异常。
* `PipelineStage` 对非法 stage/chunk 输入使用 `std::invalid_argument`。
* GPipe/Interleaved 1F1B 的 task stage/local chunk 由 `layout.chunk(global_chunk_id)`
  查询，不再在调度器中使用 `%` 或 `/` 推导归属。
* GPT-2/LLaMA3 loader 根据 `owns` 和 `stage_of_layer` 决定读取哪些 tensor，同时
  保持未拥有参数的文件 seek 顺序。

## 错误语义

布局/拓扑配置错误统一抛 `PipelineLayoutError`，普通 API 参数错误抛
`std::invalid_argument`。没有新增 glog `CHECK` 作为配置错误替代；既有训练路径的
`CHECK`/`LOG(FATAL)` 保持不变。main 对布局异常做捕获并返回非零状态。

## 校验结果

已完成：

* `infini_train`、`gpt2`、`llama3` CPU 构建成功。
* `test_pipeline_layout_cpu`：16/16 通过。
* `test_pipeline_parallel_chunking_cpu`：5/5 通过。
* `test_transformer_cpu --gtest_filter='TransformerPipelineLayoutTest.*'`：1/1 通过。
* 完整 `test_transformer_cpu`：7 通过、10 个 CPU 不支持的架构测试跳过。
* scheduler 布局测试覆盖 GPipe、1F1B、连续自定义布局和拓扑错误；修正 task 数量
  断言后应为每个 microbatch/chunk 各一次 forward/backward。

待真实环境执行：

* CUDA 编译/运行（当前工具链曾报告 nvcc 与 `<format>` 兼容问题）。
* 两 stage send/recv 无死锁验证。
* checkpoint 文件真实加载和 FP32/BF16 loss 对齐。
* `scripts/test_pipeline_custom_layout.sh` 的实际训练步骤。

## 当前限制与后续

`GetStageInfo` 兼容 API 仍保留，但包装器和两个 loader 已不再调用；默认布局内部
生成算法中的取模仅用于构造 chunk。调度器仍按连续全局 chunk 序列生成任务，尚未支持
任意 Megatron 风格显式 chunk 映射。特殊模块跨 stage 的完整 loss/target transport
也需要单独的通信协议改造。

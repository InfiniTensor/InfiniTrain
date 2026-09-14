# Pipeline 并行自定义布局使用说明

本功能把层和特殊模块的归属统一记录在 `PipelineLayout` 中。模型构建、
`PipelineParallel` 包装器、调度器和 GPT-2/LLaMA3 checkpoint loader 都查询同一份
布局，因此不会因为不同组件各自重新计算切分而产生不一致。

## 命令行参数

GPT-2 和 LLaMA3 都支持：

```text
--pipeline_parallel=N
--virtual_pipeline_parallel=N
--pipeline_layer_partition=4,8,6,6
--pipeline_embedding_stage=0
--pipeline_final_norm_stage=-1
--pipeline_lm_head_stage=-1
```

`-1` 表示最后一个 pipeline stage。特殊模块参数默认保持旧行为：embedding 在
stage 0，final norm 和 lm head 在最后一个 stage。

不提供 `--pipeline_layer_partition` 时，使用 `BuildDefault` 的均匀布局。chunk 按
`global_chunk_id = local_chunk_id * num_stages + stage_id` 交错编号，以保持现有
GPipe/1F1B 调度编号兼容。

提供 partition 时，每个逗号分隔的整数表示一个 stage 拥有的连续层数。例如
24 层模型使用 `--pipeline_parallel=4 --pipeline_layer_partition=4,8,6,6`：

```text
stage 0: layers [0,4)
stage 1: layers [4,12)
stage 2: layers [12,18)
stage 3: layers [18,24)
```

当前通过标准限制自定义 partition 必须使用 `--virtual_pipeline_parallel=1`。
需要 vPP 的非轮转 chunk 映射尚未接入 Megatron 风格布局字符串解析器。

### Megatron 风格布局（增强功能）

`--pipeline_layout` 可显式描述每个 stage 的 chunk 和特殊模块：`t` 表示
Transformer 层，`E`/`F`/`H` 分别表示 embedding、final norm、LM head，`|`
分隔 stage，`,` 分隔同一 stage 内的 vPP chunk，括号后使用 `*N` 重复。
例如 `--pipeline_parallel=2 --virtual_pipeline_parallel=2
--pipeline_layout='tt,tt|tt,tt'` 将 8 层映射为 4 个显式 chunk。该参数与
`--pipeline_layer_partition` 互斥；每个 stage 必须恰好给出 `vpp` 个 chunk，
所有 `t` 的总数必须等于模型层数。布局解析和校验在训练启动前完成。

### 自动负载均衡建议与性能对比

可使用 `pipeline_layout_suggest` 根据每层代价生成连续 partition：

```bash
./build_cpu/pipeline_layout_suggest \
  --num_layers=8 --pp_size=4 --layer_cost=1,1,1,1,4,1,1,1
# pipeline_layer_partition=3,1,1,3
```

不传 `--layer_cost` 时默认每层代价为 1。工具会保证层数总和正确、每个
stage 至少有一层，并倾向于将高代价层单独切分。

可用性能脚本对默认单卡和双卡显式布局做一次可重复的单步对比：

```bash
BUILD_DIR=build_cuda scripts/compare_pipeline_layout_perf.sh gpt2 cuda
```

脚本在 `targets/pipeline_layout_perf.csv` 写出 `elapsed_ms`、`tok_per_s`
和理想 GPipe bubble 比例。`BATCH_SIZE`、`SEQUENCE_LENGTH`、
`TOTAL_BATCH_SIZE` 可通过环境变量覆盖；bubble 使用
`(pipeline_parallel-1)/(micro_batches+pipeline_parallel-1)` 估算，属于
调度上界，不代替真实 profiler。

## 特殊模块和 checkpoint

布局查询决定 `transformer.wte/wpe`、`transformer.ln_f` 和
`transformer.lm_head` 在哪个 stage 注册和加载。层参数仍使用 canonical key：
`transformer.h.<local_index>...`。checkpoint loader 即使当前 rank 不拥有某个参数，
也会继续 seek 对应字节，避免后续权重错位。

当前 pipeline transport 的既有语义仍按 rank 0 为首 stage、最后 rank 为末 stage。
因此把 final norm/lm head 放到非末 stage 虽可由布局描述，但真实 loss/target 的跨
stage transport 尚未完全重构；部署前应保持它们在最后 stage，或先扩展 transport。

## 错误排查

布局错误使用可捕获的 `PipelineLayoutError`，不会用新的 glog `CHECK` 替代：

```text
stage count mismatch: pipeline_parallel=4, partition_entries=3
layer sum mismatch: num_layers=23, partition_sum=24
pipeline_layer_partition contains invalid character '-'
custom layer partition is not supported with vpp_size=2
```

普通空模块/非法指针参数使用 `std::invalid_argument`；训练代码中原有的
`CHECK`/`LOG(FATAL)` 仍保留其原语义。

## 校验命令

CPU 构建和单测：

```bash
cmake -S . -B build_cpu -DBUILD_TEST=ON -DUSE_CUDA=OFF -DUSE_NCCL=OFF
cmake --build build_cpu -j2
ctest --test-dir build_cpu -R 'test_pipeline_(layout|scheduler_layout|parallel_chunking)_cpu' --output-on-failure
./build_cpu/tests/transformer/test_transformer_cpu --gtest_filter='TransformerPipelineLayoutTest.*' --gtest_color=no
```

有完整数据和 launcher 时，可以运行一迭代 smoke test（这会实际执行训练步骤）：

```bash
./scripts/test_pipeline_custom_layout.sh gpt2 cuda
./scripts/test_pipeline_custom_layout.sh llama3 cuda
```

脚本默认查找 `build/`、`data/<model>/` 下的可执行文件和输入；可用
`BUILD_DIR`、`INPUT_BIN`、`INPUT_VAL_BIN`、`TOKENIZER_BIN`、`LLMC_FILE`、
`OUT_DIR`、`LAUNCHER` 覆盖。`LAUNCHER=direct` 可在单进程环境执行，但不能验证
两 stage 的真实通信。没有数据、CUDA 或 launcher 时脚本会明确退出，不会静默通过。

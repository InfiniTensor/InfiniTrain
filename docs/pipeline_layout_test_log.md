# Pipeline Layout 测试日志

日期：2026-08-21（UTC）

环境：GNU C++ 13.3.0；CPU build；CUDA 13.0.88、NCCL、2 x NVIDIA H200 GPU build。

## 构建结果

```text
[100%] Built target test_pipeline_layout
[100%] Built target gpt2
[100%] Built target llama3
[100%] Built target mixtral
```

## 单元测试结果

```text
PipelineLayoutTest.ParsesNonUniformContinuousPartition .............. Passed
PipelineLayoutTest.AssignsSpecialModulesToPipelineEndpoints ......... Passed
PipelineLayoutTest.PreservesUniformAndVirtualPipelineDistribution ... Passed
PipelineLayoutTest.BalancesUserProvidedLayerCosts ................... Passed
PipelineLayoutTest.SupportsArbitraryVirtualChunkOwnership ........... Passed
PipelineLayoutTest.ParsesMegatronRepetitionAndEmptyChunks ........... Passed
PipelineLayoutTest.RejectsInvalidAutomaticLayoutInputs ............. Passed
PipelineLayoutTest.RejectsInvalidPartitions ......................... Passed
PipelineLayoutTest.RejectsOutOfRangeQueries ......................... Passed
PipelineLayoutSuggestionTest ........................................ Passed

100% tests passed, 0 tests failed out of 10
```

自动均衡用例验证代价 `10,1,1,1,1,1` 在两个 Stage 上生成连续分区 `1,5`，并覆盖代价数量
错误、零、负数、非数字、Stage 多于层数、vPP 冲突及与手工分区同时配置等启动错误。
CPU 全量回归共 282 个注册测试；3 个 disabled，279 个已启用测试全部通过，其中 CPU 标签 246 个。

## 2-Stage GPU 验证

具备 CUDA、NCCL 和两张 GPU 的构建环境后，可使用下列方式运行非均匀 2-Stage GPT-2。模型为 12 层，
Stage 分区为 4 层和 8 层。

```bash
./build/infini_run --nproc_per_node=2 ./build/gpt2 \
  --device=cuda \
  --input_bin=data/gpt2/tiny_shakespeare_train.bin \
  --llmc_filepath=data/gpt2/gpt2_124M.bin \
  --pipeline_parallel=2 \
  --virtual_pipeline_parallel=1 \
  --pipeline_layer_partition=4,8 \
  --batch_size=4 \
  --sequence_length=64 \
  --total_batch_size=512 \
  --num_iteration=2
```

实际执行完成，无通信死锁：

```text
custom PP 4,8 step 1: loss 5.250158,  970 tok/s, peak used 1247 MB
custom PP 4,8 step 2: loss 4.913960, 8982 tok/s, peak used 1247 MB
```

使用相同 checkpoint、输入、batch、dtype 和优化参数执行单 GPU 与默认 2-Stage `6,6` 基线：

```text
single GPU step 1: loss 5.250158
single GPU step 2: loss 4.913960
default PP step 1: loss 5.250158
default PP step 2: loss 4.913958
```

自定义 PP 与单 GPU 的打印 loss 完全一致；与默认 PP 的最大打印差值为 `2e-6`，满足 fp32
`1e-5` 容差。本次短跑的稳定步吞吐为自定义 `8982 tok/s`、默认 `3410 tok/s`，说明布局能够运行且
存在改善空间，但两次短样本不能代替隔离环境下的多轮性能统计。

## 逐参数梯度一致性

单 GPU 与自定义 `4,8` PP 使用相同 checkpoint、输入及训练参数运行一步，并通过
`--dump_gradients` 导出规范化全局参数名的梯度。比较命令：

```bash
python3 scripts/precision_check/precision_compare.py \
  --dir1 /tmp/infinitrain-grad-single-20260821 \
  --dir2 /tmp/infinitrain-grad-custom-20260821 \
  --atol 1e-5 --rtol 0
```

实际结果：

```text
Directory 1: 149 files
Directory 2: 149 files
Summary: 149 passed, 0 failed, 0 errors
Missing: 0 in dir1 only, 0 in dir2 only
```

因此 Transformer 层、Embedding、Final Norm 和 LM Head 的全部 149 个参数梯度均满足 fp32
绝对误差 `1e-5`。

## 自动代价布局 GPU 验证

GPT-2 12 层使用 `--pipeline_layer_costs=10,1,1,1,1,1,1,1,1,1,1,1` 启动双卡训练。
程序生成并打印：

```text
Pipeline layout (12 layers, 2 stages):
  stage 0: embedding layers[0,1)
  stage 1: layers[1,12) final_norm lm_head
step 1/1 | train loss 5.250158 | 789.06 ms | 649 tok/s
```

训练正常退出，无通信死锁；该 loss 与相同输入和 checkpoint 的单卡及手工 PP 首步结果一致。

## 自动化 E2E 测试

上述单卡和自动 PP 验证已固化为 tests/distributed/test_pipeline_layout_e2e.sh。实际运行结果：

~~~text
single GPU loss: 5.250158
automatic PP loss: 5.250158
Directory 1: 149 files
Directory 2: 149 files
Summary: 149 passed, 0 failed, 0 errors
Missing: 0 in dir1 only, 0 in dir2 only
PASS: automatic PP layout, loss, and gradients match the single-GPU reference
~~~

脚本同时断言自动布局为 [0,1)、[1,12)，任何训练进程失败、布局不符、loss 超过 fp32
1e-5、梯度文件缺失或梯度数值超差都会返回非零退出码。

## 任意 vPP 与 Megatron 布局 GPU 验证

Chunk owner `[0,1,1,0]`（`--pipeline_chunk_layout=0:3,1:3,1:3,0:3`）在双 H200 上完成训练，
自动布局打印为 stage 0 `[0,3)`、`[9,12)`，stage 1 `[3,6)`、`[6,9)`，loss `5.250158`。
Megatron 表达式 `Et*3||t*3|t*6NL` 含空 Chunk，同样完成训练并得到 loss `5.250158`。

## 稳定负载基准

两张 H200、GPT-2 124M、4 个 microbatch、12 个迭代，去掉首 3 步：

```text
default 6,6:       mean 89.532 ms, 11,437 tok/s, peak 1473 MB
profiler 7,5:      mean 81.799 ms, 12,519 tok/s, peak 1343 MB
improvement:       +9.45% throughput, -130 MB peak memory
```

Profiler 建议来自 3 步单卡 PROFILE_MODE 记录，每层丢弃一个 warmup 样本；建议工具单测和实际记录
解析均已通过。

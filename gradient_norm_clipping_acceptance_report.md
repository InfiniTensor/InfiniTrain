# 梯度裁剪（Gradient Norm Clipping）最终验收报告

## 1. 验收结论

梯度裁剪已纳入 `Optimizer` 公共多态接口，完成 CPU/CUDA 范数统计、CPU/CUDA 原地缩放、`DistributedOptimizer` 的 DP/TP/SP/PP/ZeRO 归约、GPT-2/LLaMA3 example 参数和训练调用、测试与文档集成。

- CPU/CUDA 裁剪专项：16 个用例，15 passed，1 个 CUDA CPU-only death-test 按设计 skipped，退出码 0。
- 两进程 CUDA 分布式专项：5/5 passed，退出码 0；覆盖 ZeRO-1/2 shard、empty shard、全局 norm、p=0、p=-∞、`foreach=true`。
- 单卡 BF16+p=0、2 卡 ZeRO-2+foreach、4 卡 TP4/SP、2 卡 PP2/vPP2+LoRA 真实训练均退出码 0，并输出 `total_grad_norm`。
- `PROFILE_MODE=ON` 下 `ScaleInplaceMulti=1`、逐 tensor `ScaleInplace=0`，证明 `foreach=true` 使用批量 kernel 路径。

## 2. 提交要求对照

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 公共接口与默认语义 | 通过 | `Optimizer::ClipGradNorm_` virtual API、wrapper、配置入口 |
| CPU/CUDA 实现 | 通过 | `ScaleInplace` 与 `ScaleInplaceMulti`，CMake CUDA 构建 |
| 分布式和分片语义 | 通过 | 两进程 5 个专项、2 卡 ZeRO-2 smoke |
| example 集成 | 通过 | GPT-2/LLaMA3 CLI、普通训练和 PP schedule 均调用裁剪 |
| 报告与文档 | 通过 | 本报告、`docs/gradient_norm_clipping.md` |
| 代码提交/PR review/approve | 按要求保留 | 本轮不执行 commit/push/review/approve |

## 3. 公共 API 与行为

```cpp
virtual std::shared_ptr<Tensor> ClipGradNorm_(
    const std::vector<std::shared_ptr<Tensor>>& parameters,
    float max_norm,
    float norm_type = 2.0f,
    bool error_if_nonfinite = false,
    std::optional<bool> foreach = std::nullopt);
```

`ClipGradNorm` 返回裁剪前的 CPU FP32 scalar，并原地缩放选中参数的梯度；无梯度参数被忽略，重复参数指针只计算一次，梯度存储不被替换。缩放系数为：

```text
min(max_norm / (total_norm + 1e-6), 1)
```

支持的 `norm_type`：

- 有限正 p 和有限负 p：按通用 p-norm 公式计算；
- `0`：按 PyTorch 语义，先对每个梯度 tensor 计非零元素，再对 tensor 列表取 0-norm；
- `+∞`：最大绝对值；
- `-∞`：最小绝对值；
- `error_if_nonfinite=true`：在缩放前检查并终止。

GPT-2 与 LLaMA3 暴露：

```text
--clip_grad_norm=-1
--grad_norm_type=2
--clip_grad_error_if_nonfinite=true
--clip_grad_foreach=auto|true|false
```

## 4. 实现与数据流

`infini_train/src/optimizer.cc` 负责重复指针过滤、FP32 统计、p=0/负 p/±∞ 分支、非有限检查和按 device/dtype 分组的缩放。

`infini_train/src/kernels/cpu/accumulate_grad.cc`：

- `ScaleInplace`：单 tensor OpenMP 缩放；
- `ScaleInplaceMulti`：多个 tensor 的批量 OpenMP 调度。

`infini_train/src/kernels/cuda/accumulate_grad.cu`：

- `ScaleInplace`：单 tensor CUDA kernel；
- `ScaleInplaceMulti`：一次 pointer/offset 元数据上传和一次多 tensor CUDA kernel launch，直接写回原始 view，保留 ZeRO flat-buffer alias。

`DistributedOptimizer::ClipGradNorm_` 的顺序为：

1. `FinishGradSync()`，等待梯度 collective；
2. 选择请求参数对应的 shard；
3. TP replicated bias 只在 TP group rank 0 计入 norm；
4. base optimizer 计算本地 shard 统计；
5. DP、TP、PP group 按 norm 类型执行 sum/max/min 归约；
6. 所有 rank 使用同一全局系数批量缩放本地 shard。

Pipeline schedule 在所有 micro-batch backward 完成后裁剪一次，并透传最后一次 `total_grad_norm` 到 example 日志。

## 5. 测试结果

主要测试文件：

- `tests/optimizer/test_clip_grad_norm.cc`
- `tests/optimizer/test_optimizer_parameter_names.cc`
- `tests/optimizer/CMakeLists.txt`

覆盖内容：

- L2、L1、非整数 p、`+∞`；
- p=0、有限负 p、`-∞`；
- `max_norm=0`、空梯度、重复参数、非有限梯度；
- CPU FP16/BF16/FP32 混合 dtype；
- `foreach=true` 批量语义；
- ZeRO local/global/empty shard、TP owner filtering。

远程 CUDA 裁剪专项：

```text
16 tests ran
15 passed
1 skipped (CUDA/ClipGradNormTest.ErrorOnNonFiniteBeforeScaling is CPU-only)
STATUS:0
```

远程两进程分布式专项：

```text
5 tests ran
5 passed on each rank
STATUS:0
```

通过的专项包括：

```text
DistributedOptimizerPropagatesNamesToShardOptimizer
DistributedOptimizerClipGradNormUsesZero2LocalShard
DistributedOptimizerClipGradNormHandlesEmptyLocalShard
DistributedOptimizerClipGradNormUsesGlobalShardNorm
DistributedOptimizerClipGradNormSupportsZeroAndNegativeInfinity
```

本地 CPU 优化器完整回归：

```text
36 tests ran, 31 passed, 5 CUDA-only skipped, exit code 0
```

远程 CUDA 优化器完整回归：

```text
70 tests ran, 59 passed, 11 design-skipped (CPU-only or distributed-only filters), exit code 0
```

## 6. 真实 GPU 训练验收

### 单卡 FP32 + foreach

```text
train loss 5.358113
total_grad_norm 65.678352
DP=1, TP=1, SP=1, PP=1
exit code 0
```

### 单卡 BF16 + p=0

```text
train loss 5.342469
total_grad_norm 149.000000
DP=1, TP=1, SP=1, PP=1
exit code 0
```

### 2 卡 ZeRO-2 + foreach=true

```text
train loss 5.439159
total_grad_norm 51.047932
DP=2, TP=1, SP=1, PP=1
exit code 0
```

### 4 卡 TP4/SP

```text
train loss 5.356194
total_grad_norm 18.435957
DP=1, TP=4, SP=4, PP=1
exit code 0
```

### 2 卡 PP2/vPP2 + LoRA

```text
train loss 5.358113
total_grad_norm 0.896690
DP=1, TP=1, SP=1, PP=2
exit code 0
```

PP 日志已补齐 `total_grad_norm`，不再出现 pipeline 路径只有 loss 而没有 norm 的情况。

## 7. Foreach/Profile 证据

profile 构建命令使用 `-DPROFILE_MODE=ON`，同一轮 GPT-2 FP32 + `clip_grad_foreach=true` 训练结果：

```text
ScaleInplaceMulti: 1
ScaleInplace: 0
GPT2_STATUS:0
```

示例 profile 记录：

```text
ScaleInplaceMulti Device(CUDA, 0)
```

这证明 `foreach=true` 没有退化到逐 tensor `ScaleInplace`。缩放 kernel 仍在 CUDA stream 上提交；范数最终 scalar 的 host copy 属于当前返回 CPU scalar 的 API 行为。

## 9. 配置矩阵

`scripts/test_config.json` 新增 `gradient_clipping_excellent` 组：

- `single_fp32_foreach`
- `single_bf16_p0`
- `tp4_sp_zero2`
- `pp2_vpp2_lora`

每个 case 都配置了 `clip_grad_norm`、`grad_norm_type` 和 `clip_grad_foreach`；GPT-2 和 LLaMA3 测试组入口均已包含该 tag。

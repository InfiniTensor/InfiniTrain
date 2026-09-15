# 分布式梯度范数计算与梯度裁剪

## 公开接口

公开调用入口如下：

```cpp
auto total_norm = optimizer->ClipGradNorm_(
    parameters,
    max_norm,
    norm_type,
    error_if_nonfinite,
    std::nullopt);
```

该接口返回一个位于 CPU 上的 FP32 标量，表示**裁剪前的梯度总范数**，并对选中的梯度进行原地缩放。

- 没有梯度的参数会被忽略。
- 对于重复出现的参数指针，仅计算一次。

## 梯度范数类型

`norm_type` 遵循 PyTorch 风格的向量范数定义：

- `0`：首先判断每个梯度张量是否至少包含一个非零元素，然后对这些梯度张量组成的列表计算 0-范数，即统计包含非零元素的梯度张量数量。
- 有限的正数或负数 `p`：按照 p-范数公式进行计算。
- `+inf`：取所有梯度元素绝对值的最大值。
- `-inf`：取所有梯度元素绝对值的最小值。

梯度缩放系数的计算公式为：

```text
min(max_norm / (total_norm + 1e-6), 1)
```

因此，当 `max_norm == 0` 时，所有梯度都会被置零，而不是禁用梯度裁剪。

## 多张量缩放

当设置 `foreach=true` 时，将使用多张量批量缩放路径。

CPU 和 CUDA 的 `ScaleInplaceMulti` 内核会直接写入现有的梯度视图，从而保留其与以下存储区域之间的别名关系：

- 扁平缓冲区（flat buffer）
- ZeRO 梯度分片

同时，每个设备和数据类型分组仅进行一次批量分派。

CUDA 梯度范数检查仅在获取标量统计信息时执行同步；梯度缩放操作仍在 CUDA 流上执行。

## 分布式梯度裁剪

`DistributedOptimizer::ClipGradNorm_` 的处理流程如下：

1. 等待所有尚未完成的梯度集合通信操作结束。
2. 针对底层优化器实际使用的分片参数计算梯度范数。
3. 在数据并行（DP）进程组中对范数统计量执行归约。
4. 使用相同的裁剪系数缩放每个进程上的本地梯度分片。

启用流水线并行（PP）后，范数统计量还会在 PP 进程组中进一步归约。

流水线调度器会在所有微批次执行完成后调用 `ClipGradNormConfigured`。

## GPT-2 和 LLaMA-3 配置参数

GPT-2 和 LLaMA-3 示例提供了以下命令行选项。

### 梯度裁剪阈值

```text
--clip_grad_norm=-1
```

负值表示禁用梯度裁剪。

### 梯度范数类型

```text
--grad_norm_type=2
```

指定梯度范数类型，此处表示使用 2-范数。

### 非有限值检查

```text
--clip_grad_error_if_nonfinite=true
```

当梯度总范数为 `NaN`、`+inf` 或 `-inf` 等非有限值时触发错误。

### 多张量缩放模式

```text
--clip_grad_foreach=auto|true|false
```

控制是否使用多张量批量缩放路径：

- `auto`：自动选择。
- `true`：强制启用。
- `false`：禁用。

## 日志记录

启用梯度裁剪后，日志中会输出一条 `total_grad_norm` 记录，用于显示裁剪前的梯度总范数。

## 并行训练支持

对于张量并行（TP）中的复制参数，`DistributedOptimizer` 会根据参数所有权进行过滤，以避免重复统计。

完整的 TP/SP/PP/vPP 混合并行训练测试由以下文件中的可选测试矩阵提供：

```text
scripts/test_config.json
```

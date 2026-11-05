# 数值容差策略

本文用于显式 `--alignment deep` 中的数值检查。历史日志比较沿用原脚本的绝对 loss 容差（FP32 `1e-5`、BF16 `1e-2`）及 20% 吞吐降速阈值，可通过命令行覆盖；日志比较不使用本策略注册表。

`tolerances.json` 集中定义测试使用的数值容差。新增普通策略或修改某个 backend 的特定指标只需要编辑 JSON，再在场景的检查项中选择策略。

## 默认行为

| 策略 | dtype | atol | rtol |
|---|---|---:|---:|
| `oracle`、`distributed_equivalence` | `float32` | 1e-5 | 1e-5 |
| `oracle`、`distributed_equivalence` | `bfloat16` | 1e-2 | 1e-2 |
| `lr_schedule` | 两种 dtype | 1e-9 | 1e-5 |

判定公式为 `abs(actual-reference) <= atol + rtol*abs(reference)`。这些默认值保留原检查器的数值行为。

指标名称为 `loss`、`lr`、`consumed_samples`、`parameter`、`gradient`、`optimizer`。`consumed_samples` 和整数／布尔 tensor state 始终精确比较；optimizer 策略的浮点容差只作用于浮点状态。初始参数一致性和真正的 DP 副本一致性也仍要求精确比较。

## 配置结构与覆盖顺序

顶层必须包含 `schema_version: 1`、`backends` 和 `policies`。每个策略包含两种 dtype 的完整 `defaults`，可以增加 `metrics` 和 `backends`：

1. 选择策略的 `defaults[dtype]`。
2. 应用 `metrics[metric][dtype]`。
3. 应用 `backends[backend][metric][dtype]`。

每个叶子都必须同时提供有限、非负的 `atol` 和 `rtol`。没有继承链或隐式策略合并。已登记 backend 缺少专项覆盖时使用通用配置；未知 backend、策略、指标、dtype、字段或重复 JSON 键会在规划阶段报错。默认 backend 为 `cpu`、`cuda` 和 `privateuseone`，`privateuse1` 是后者的别名。

例如，在现有 `policies` 中增加一个条目：

```json
"attention_strict": {
  "defaults": {
    "float32": {"atol": 0.00001, "rtol": 0.00001},
    "bfloat16": {"atol": 0.01, "rtol": 0.01}
  },
  "metrics": {
    "loss": {
      "float32": {"atol": 0.0000001, "rtol": 0.000001}
    }
  },
  "backends": {
    "cuda": {
      "gradient": {
        "bfloat16": {"atol": 0.002, "rtol": 0.003}
      }
    }
  }
}
```

这个条目让 FP32 loss 使用更严格的阈值，并单独设置 CUDA BF16 gradient；其他指标和 backend 沿用条目中的 defaults。具体阈值应由已知正确版本的重复测量及独立参考校准，经审查后写入配置。

## 在场景中选择

独立参考检查：

```json
{"kind": "oracle", "tolerance": "attention_strict"}
```

场景生成的等价性参考：

```json
"reference": {
  "single_device": true,
  "metrics": ["loss", "gradient", "parameter"],
  "tolerance": "attention_strict"
}
```

`oracle` 和 `lr_schedule` 检查同样支持 `tolerance` 字段。不指定时分别使用其默认命名策略。性能的时间、显存和噪声阈值在场景的 `performance` 配置中设置。

用 `./scripts/run_all_tests.sh --alignment deep --suite pr --model gpt2 --tag accumulation --dry-run --explain` 查看解析后的 `tolerance_policies` 与 `resolved_tolerances`。

## 配置与历史匹配

规划器将实际生效的非默认策略值纳入 case 配置哈希。修改某个 backend 的指标阈值只会影响使用该值的 case。历史日志仍按配置匹配，不会把配置不同的运行当作已经完成对齐；具体参考运行与解析后的策略保存在本次运行产物中。

# FlashAttention 项目提交报告

## 1. 功能正确性验证

### 1.1 接入与开关

- 已在训练路径接入 FlashAttention 风格 SDPA，支持可训练的 forward/backward。
- 已提供 `--flash` 参数进行切换：
  - `--flash=false`：baseline（小算子拼接）
  - `--flash=true`：FlashAttention 路径

### 1.2 对齐对象与结论

- 对齐对象：同一框架内 baseline 实现（`--flash=false`）与 FlashAttention 实现（`--flash=true`）。
- 训练结果对齐：在两组都成功完成的 case 上，loss 差异整体很小，符合“允许浮点差异”的要求。
- 证据文件：
  - `scripts/logs/require_baseline/`
  - `scripts/logs/require_flash/`
  - `scripts/logs/require_run_summary.json`
  - `scripts/logs/require_analysis.md`

### 1.3 GPT-2 / LLaMA-3 运行日志

- GPT-2 与 LLaMA-3 的端到端训练日志均已保留在上述日志目录中。
- 说明：部分大并行配置存在 timeout/OOM（baseline 与 flash 均可能出现），不影响可比 case 的正确性与性能结论。

## 2. 性能评估报告

### 2.1 实验环境

- GPU：NVIDIA A100-SXM4-80GB（多卡）
- 单卡显存：81920 MiB
- 驱动版本：570.133.20
- CUDA：12.8
- GCC：13.3.0
- OS：Linux

### 2.2 实验配置

- 主脚本：`scripts/run_models_and_profile.bash`
- 配置文件：`scripts/test_config.json`
- 对比方式：同一组 test cases 分别运行 baseline / flash
- 运行参数：通过 `MODEL_EXTRA_ARGS` 注入 `--flash` 开关，使用 `--continue-on-error` 保留全量可运行证据

### 2.3 对比方案

- baseline：`--flash=false`
- 实验组：`--flash=true`
- 统计来源：`scripts/logs/require_run_summary.json` 与 `scripts/logs/require_analysis.md`

### 2.4 指标定义

- 平均迭代耗时（avg latency, ms）
- 吞吐率（tokens/s）
- GPU 显存占用（MB）
- 加速比（Speedup = baseline / flash）
- 显存节省比例（MemSave = (baseline - flash) / baseline）

### 2.5 结果展示（代表性可比 case）

| Case | Baseline ms | Flash ms | Speedup | Baseline tok/s | Flash tok/s | Baseline MB | Flash MB | MemSave |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_basic_2_bfloat16 | 2552.07 | 2589.62 | 0.985x | 2006 | 1977 | 7889 | 7889 | 0.00% |
| gpt2_basic_3_bfloat16 | 3721.09 | 3453.64 | 1.077x | 1376 | 1482 | 3405 | 3405 | 0.00% |
| llama3_basic_1_bfloat16 | 1489.90 | 933.81 | 1.596x | 172 | 274 | 24499 | 19189 | 21.67% |
| llama3_basic_3_bfloat16 | 9750.78 | 2601.98 | 3.747x | 525 | 1968 | 31636 | 26082 | 17.56% |
| llama3_zero_3_bfloat16_distopt | 7431.83 | 5842.90 | 1.272x | 689 | 876 | 21632 | 16078 | 25.67% |
| llama3_lora_2_lora_bfloat16 | 16522.24 | 5800.78 | 2.848x | 310 | 883 | 27850 | 20418 | 26.69% |

### 2.6 汇总结论

- 在 GPT-2 小配置场景，flash 与 baseline 多数接近持平。
- 在 LLaMA-3 BF16 场景，flash 在多组可比 case 上获得显著吞吐提升，并明显降低显存占用。
- 功能正确性（loss 对齐）与性能对比证据均满足项目提交要求。

## 3. 代码提交与可复现性
  
### 3.1 一键复现命令

```bash
cd scripts
MODEL_EXTRA_ARGS='--flash=false --num_iteration=10' ./run_models_and_profile.bash --continue-on-error
MODEL_EXTRA_ARGS='--flash=true --num_iteration=10' ./run_models_and_profile.bash --continue-on-error
```

### 3.2 复现产物

- baseline 日志：`scripts/logs/require_baseline/`
- flash 日志：`scripts/logs/require_flash/`
- 汇总数据：`scripts/logs/require_run_summary.json`
- 分析报告：`scripts/logs/require_analysis.md`

reviewer 可直接基于默认脚本和默认配置复现并核对。

## 4. 验收对齐清单

- FlashAttention 已接入 GPT-2 / LLaMA-3 训练路径并可运行：已完成
- `--flash` 开关可切换 baseline / flash：已完成
- 训练结果与 baseline 对齐（允许浮点差异）：已完成
- 提供完整性能对比（latency / tokens/s / 显存 / speedup / mem save）：已完成
- 提供可复现脚本与日志证据：已完成

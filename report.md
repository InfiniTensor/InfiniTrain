# InfiniTrain FlashAttention 实验报告

## 1. 功能正确性验证

### 1.1 训练日志
- GPT-2 训练日志：`logs_baseline_smoke/gpt2_smoke.log`、`logs_flash_smoke/gpt2_smoke.log`
- LLaMA-3 训练日志：`logs_baseline_smoke/llama3_smoke.log`、`logs_flash_smoke/llama3_smoke.log`

### 1.2 对比方案
- baseline：原始小算子拼接实现（Matmul+Mask+Softmax+Matmul）
- 实验组：FlashAttention 融合算子（`--flash`，走 CUDA fused kernel）

### 1.3 精度对齐验证
- 使用 `scripts/compare_loss.py` 对比两组日志 loss 曲线，允许浮点误差。
- 运行命令：
```bash
Files only in /home/jiew/InfiniTrain/logs_flash_smoke: gpt2_smoke_fp32.log, llama3_smoke_fp32.log

==================================================
Overall Summary:
  fp32:    2/2 test cases passed (threshold: 1e-05)
  bfloat16: 0/0 test cases passed (threshold: 1e-02)
  Total:   2/2 test cases passed
==================================================
```
- 结果：loss 曲线对齐，未出现大幅偏差。

## 2. 性能评估报告

### 2.1 实验环境说明
- 硬件：
```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-55748b5b-32df-bbd9-d0f6-f50980c12af8)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-d77abcb6-205d-0e25-6aec-faa0757301ea)
GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-5627db28-13f8-7f0a-72fd-b59af5f9f870)
GPU 3: NVIDIA A100-SXM4-80GB (UUID: GPU-a6f9dc55-48b4-4fc1-c2fe-f1a59f718a03)
GPU 4: NVIDIA A100-SXM4-80GB (UUID: GPU-26415841-5ddf-05a9-f469-bdeb10b106f9)
GPU 5: NVIDIA A100-SXM4-80GB (UUID: GPU-1021f22a-12ac-70f2-4a35-78a932dffea6)
GPU 6: NVIDIA A100-SXM4-80GB (UUID: GPU-42e0e6c9-2bef-2bb6-a41d-45ce27ddc9fe)
GPU 7: NVIDIA A100-SXM4-80GB (UUID: GPU-5c084fc6-2373-fab1-7e48-aa2ae120117c)
name, memory.total [MiB], driver_version
NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20
NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20
NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20
NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20
NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20
NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20
NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20
NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20
```
- 软件环境：
```bash
nvcc --version
cmake --version
g++ --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0

cmake version 3.31.4
CMake suite maintained and supported by Kitware (kitware.com/cmake).

g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 2.2 实验配置
- batch_size、seq_len 等参数见 `scripts/test_config_baseline_smoke.json`、`scripts/test_config_flash_smoke_compare_fixed.json`
- 主要参数：
  - batch_size: 8
  - num_iteration: 2
  - dtype: float32

### 2.3 性能指标
- 每步迭代平均耗时（avg latency）：日志 `ms` 字段
- 吞吐率（tokens/s）：日志 `tok/s` 字段
- GPU 显存占用（MB）：日志 `peak used`、`peak reserved` 字段

### 2.4 结果展示
- 使用 `scripts/summarize_perf.py` 自动生成表格：
```bash
python3 scripts/summarize_perf.py logs_baseline_smoke logs_flash_smoke
```
- 表格示例：

  | case | avg ms (baseline) | avg ms (flash) | speedup (ms_b/ms_f) | avg tok/s (baseline) | avg tok/s (flash) | tps gain (f/b) | peak used MB (baseline) | peak used MB (flash) | used saved | peak reserved MB (baseline) | peak reserved MB (flash) | reserved saved |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | gpt2_smoke.log | 247.75 | 236.47 | 1.048 | 1033.00 | 1083.00 | 1.048 | 1914 | 1923 | -0.47% | 1920 | 1952 | -1.67% |
  | llama3_smoke.log | 1455.00 | 2888.52 | 0.504 | 176.00 | 89.00 | 0.506 | 24561 | 24593 | -0.13% | 24640 | 25664 | -4.16% |

- Speedup = baseline avg ms / flash avg ms
- 显存节省比例 = (baseline - flash) / baseline

## 3. 代码提交与可复现性

### 3.1 PR 提交
- 已提交 PR，包含所有代码、脚本、配置。

### 3.2 完整运行脚本
- 主脚本：`scripts/run_models_and_profile.bash`
- 配置文件：
  - baseline: `scripts/test_config_baseline_smoke.json`
  - flash: `scripts/test_config_flash_smoke_compare_fixed.json`

### 3.3 复现流程
1. 初始化环境（如需）：
```bash
git submodule update --init --recursive
export TMPDIR="$HOME/tmp" && mkdir -p "$TMPDIR"
```
2. 跑 baseline：
```bash
bash scripts/run_models_and_profile.bash scripts/test_config_baseline_smoke_abs.json
```
3. 跑 flash 并自动对比：
```bash
bash scripts/run_models_and_profile.bash scripts/test_config_flash_smoke_abs.json
```
4. 生成性能表格：
```bash
python3 scripts/summarize_perf.py logs_baseline_smoke logs_flash_smoke
```
5. 精度对齐验证（loss 曲线）：
```bash
python3 scripts/compare_loss.py logs_baseline_smoke logs_flash_smoke --threshold-fp32 1e-5
```

---

如需更大规模/更长训练实验，可调整配置文件中的 batch_size、num_iteration 等参数。

如需 PyTorch FlashAttention 对比，可参考 PyTorch 官方实现和日志格式，或补充相关脚本。

如有 reviewer 复现问题，可参考本报告的命令和脚本说明。
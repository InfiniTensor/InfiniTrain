# InfiniTrain 运行指南

本指南介绍如何使用 `scripts` 目录下的脚本进行模型训练、性能分析和结果统计。

---

## 快速开始

### 基本运行

最简单的运行方式：

```bash
cd ~/InfiniTrain/scripts
bash run_models_and_profile.bash
```

这将使用默认配置文件 `test_config.json` 运行所有测试。

### 强制重新构建

如果需要重新构建项目：

```bash
cd ~/InfiniTrain/scripts
bash run_models_and_profile.bash --rebuild
```

### 使用自定义配置文件

```bash
cd ~/InfiniTrain/scripts
bash run_models_and_profile.bash /path/to/custom_config.json
```

---

## 脚本说明

### 1. run_models_and_profile.bash

主运行脚本，负责：
- 从配置文件读取测试配置
- 构建项目（如果需要）
- 运行 GPT2 和 LLaMA3 模型测试
- 收集训练日志和性能分析数据
- 自动运行比较脚本

**依赖工具**：
- `jq`：用于解析 JSON 配置文件

**安装 jq**：
```bash
sudo apt-get install -y jq
```

**运行流程**：
1. 读取配置文件（默认：`test_config.json`）
2. 检查是否需要构建（或使用 `--rebuild` 强制重新构建）
3. 对每个测试配置，分别运行：
   - GPT2 模型（noflash 模式）
   - GPT2 模型（flash 模式）
   - LLaMA3 模型（noflash 模式）
   - LLaMA3 模型（flash 模式）
4. 如果启用了性能分析，收集性能报告
5. 如果设置了 `COMPARE_LOG_DIR`，自动运行比较脚本

**日志输出**：
- 训练日志：`logs/` 目录（noflash 结果）
- 训练日志：`compare_logs/` 目录（flash 结果）
- 性能日志：`profile_logs/` 目录（noflash 结果）
- 性能日志：`compare_profile_logs/` 目录（flash 结果）

---

### 2. test_config.json

配置文件包含三个主要部分：

#### 2.1 variables - 变量定义

```json
{
  "variables": {
    "BUILD_DIR": "../build",
    "GPT2_INPUT_BIN": "/path/to/gpt2/input.bin",
    "GPT2_LLMC_FILEPATH": "/path/to/gpt2/model.bin",
    "LLAMA3_INPUT_BIN": "/path/to/llama3/input.bin",
    "LLAMA3_LLMC_FILEPATH": "/path/to/llama3/model.bin",
    "PROFILE_LOG_DIR": "./profile_logs",
    "LOG_DIR": "./logs",
    "COMPARE_LOG_DIR": ""
  }
}
```

**变量说明**：
- `BUILD_DIR`：构建目录路径
- `GPT2_INPUT_BIN`：GPT2 输入数据文件路径
- `GPT2_LLMC_FILEPATH`：GPT2 模型文件路径
- `LLAMA3_INPUT_BIN`：LLaMA3 输入数据文件路径
- `LLAMA3_LLMC_FILEPATH`：LLaMA3 模型文件路径
- `PROFILE_LOG_DIR`：性能分析日志目录
- `LOG_DIR`：训练日志目录
- `COMPARE_LOG_DIR`：基准日志目录（用于比较，留空则不运行比较）

#### 2.2 builds - 构建配置

```json
{
  "builds": [
    {
      "id": "build_2",
      "profile": true,
      "cmd": "cmake -DUSE_CUDA=ON -DUSE_NCCL=ON -DPROFILE_MODE=ON .. && make -j"
    }
  ]
}
```

**配置说明**：
- `id`：构建 ID（用于日志命名）
- `profile`：是否启用性能分析（true/false）
- `cmd`：CMake 构建命令

#### 2.3 tests - 测试配置

每个测试包含以下参数：

```json
{
  "id": "1",
  "args": {
    "dtype": "float32",
    "num_iteration": 10,
    "batch_size": 80,
    "total_batch_size": 5120,
    "nthread_per_process": 8,
    "tensor_parallel": 4,
    "sequence_parallel": true,
    "pipeline_parallel": 8,
    "virtual_pipeline_parallel": 2,
    "use_distributed_optimizer": true
  }
}
```

**参数说明**：
- `dtype`：数据类型（float32/bfloat16）
- `num_iteration`：迭代次数
- `batch_size`：批次大小
- `total_batch_size`：总批次大小
- `nthread_per_process`：每进程线程数（数据并行）
- `tensor_parallel`：张量并行度
- `sequence_parallel`：序列并行（true/false）
- `pipeline_parallel`：流水线并行度
- `virtual_pipeline_parallel`：虚拟流水线并行度
- `use_distributed_optimizer`：使用分布式优化器（true/false）

**测试 ID 命名规则**：
- `1`：基础测试
- `1_bfloat16`：基础测试（bfloat16）
- `2`：批次大小测试
- `3`：数据并行测试
- `3_distopt`：数据并行 + 分布式优化器
- `4`：张量并行测试
- `5`：张量并行 + 序列并行测试
- `6`：流水线并行测试
- `7`：虚拟流水线并行测试
- `8`：混合并行测试

---

### 3. extract_training_stats.py

从日志文件中提取训练统计信息，包括：
- 平均损失
- 平均时间
- 平均 tokens/s
- 峰值内存使用
- 并行配置（DP、TP、SP、PP）

**基本用法**：
```bash
python3 extract_training_stats.py <log_dir1> <log_dir2>
```

**选项**：
- `--threshold-fp32`：FP32 损失差异阈值（默认：1e-5）
- `--threshold-bf16`：BF16 损失差异阈值（默认：1e-2）
- `--markdown`：输出 Markdown 格式表格
- `--output <file>`：指定输出文件路径
- `--speedup`：输出加速比表格

**示例**：

```bash
# 显示训练统计信息和对比
python3 extract_training_stats.py logs compare_logs

# 输出 Markdown 格式
python3 extract_training_stats.py logs compare_logs --markdown

# 输出 Markdown 并保存到文件
python3 extract_training_stats.py logs compare_logs --markdown --output results.md

# 只显示加速比（不显示训练统计和对比信息）
python3 extract_training_stats.py logs compare_logs --speedup

# 只显示加速比并保存为 Markdown
python3 extract_training_stats.py logs compare_logs --speedup --markdown --output speedup.md
```

**输出说明**：

**不使用 `--speedup` 时**：

1. **训练统计表**：
   - Row ID：配置标识（模型_数据类型_flash/noflash_分布式优化器_测试ID）
   - Avg Loss：平均损失
   - Avg Time：平均时间（ms）
   - Avg Tok/s：平均 tokens/s
   - Peak Used：峰值内存使用（MB）
   - Peak Reserved：峰值内存保留（MB）
   - DP/TP/SP/PP：并行配置

2. **Flash vs Noflash 对比**：
   - 损失对比及差异百分比
   - 时间对比及差异百分比
   - Tokens/s 对比及差异百分比
   - 内存使用对比及差异百分比

**使用 `--speedup` 时**：

3. **加速比表**：
   - Configuration：配置名称
   - Speedup Ratio：加速比（noflash_time / flash_time）
   - Flash Time：Flash 模式平均时间
   - Noflash Time：Noflash 模式平均时间

**加速比说明**：
- 加速比 > 1.0：Flash Attention 更快
- 加速比 < 1.0：Flash Attention 更慢

---

### 4. compare_loss.py

比较两个日志目录中的训练损失，用于验证正确性。

**基本用法**：
```bash
python3 compare_loss.py <dir1> <dir2>
```

**选项**：
- `--threshold-fp32`：FP32 损失差异阈值（默认：1e-5）
- `--threshold-bf16`：BF16 损失差异阈值（默认：1e-2）
- `--verbose`：显示所有文件的详细输出（包括通过的）

**示例**：

```bash
# 使用默认阈值比较
python3 compare_loss.py logs compare_logs

# 自定义阈值
python3 compare_loss.py logs compare_logs --threshold-fp32 1e-5 --threshold-bf16 1e-2

# 显示详细输出
python3 compare_loss.py logs compare_logs --verbose
```

**输出说明**：

1. **文件差异**：
   - 只在 dir1 中的文件
   - 只在 dir2 中的文件

2. **每个测试的对比**：
   - 不匹配的步骤及损失值
   - 差异值和相对误差百分比
   - 匹配步骤数统计

3. **总体统计**：
   - FP32 测试通过率
   - BF16 测试通过率
   - 总体通过率

**返回值**：
- 0：所有测试通过
- 1：存在不匹配的测试

---

### 5. compare_tps.py

比较两个日志目录中的 tokens per second，用于性能对比。

**基本用法**：
```bash
python3 compare_tps.py <dir1> <dir2>
```

**选项**：
- `--threshold`：相对误差阈值（默认：0.20 = 20%）
- `--verbose`：显示所有文件的详细输出（包括通过的）

**示例**：

```bash
# 使用默认阈值比较
python3 compare_tps.py logs compare_logs

# 自定义阈值
python3 compare_tps.py logs compare_logs --threshold 0.15

# 显示详细输出
python3 compare_tps.py logs compare_logs --verbose
```

**输出说明**：

1. **文件差异**：
   - 只在 dir1 中的文件
   - 只在 dir2 中的文件

2. **每个测试的对比**：
   - 平均 tokens/s 对比
   - 相对误差百分比
   - 比较的步骤数（排除第一步）

3. **总体统计**：
   - 通过的测试用例数
   - 总测试用例数
   - 通过率

**注意**：
- 比较时会自动排除第一步（因为第一步可能有初始化开销）
- 相对误差计算：`|avg1 - avg2| / max(avg1, avg2)`

**返回值**：
- 0：所有测试通过
- 1：存在不匹配的测试

---

## 完整使用流程

### 1. 准备配置

编辑 `test_config.json`：
- 确保 `variables` 中的路径正确
- 配置 `builds` 中的构建命令
- 在 `tests` 中添加或修改测试配置

### 2. 运行测试

```bash
cd ~/InfiniTrain/scripts
bash run_models_and_profile.bash
```

### 3. 查看日志

- 训练日志：`logs/` 和 `compare_logs/`
- 性能日志：`profile_logs/` 和 `compare_profile_logs/`

### 4. 分析结果

**提取训练统计**：
```bash
python3 extract_training_stats.py logs compare_logs
```

**验证正确性**：
```bash
python3 compare_loss.py logs compare_logs --threshold-fp32 1e-5 --threshold-bf16 1e-2
```

**对比性能**：
```bash
python3 compare_tps.py logs compare_logs --threshold 0.20
```

**计算加速比**：
```bash
python3 extract_training_stats.py logs compare_logs --speedup
```

---

## Flash Attention 对比说明

脚本会自动为每个测试运行两个版本：

- **noflash**：不使用 Flash Attention
  - 日志保存在 `logs/` 目录
  - 性能报告保存在 `profile_logs/` 目录

- **flash**：使用 Flash Attention
  - 日志保存在 `compare_logs/` 目录
  - 性能报告保存在 `compare_profile_logs/` 目录

这种设计允许：
1. 验证 Flash Attention 的正确性（通过 compare_loss.py）
2. 对比 Flash Attention 的性能（通过 compare_tps.py）
3. 计算加速比（通过 extract_training_stats.py --speedup）



## 日志文件命名规则

- 训练日志：`{model}_{test_id}_{dtype}_{distopt}.log`
- 性能日志：`{model}_{test_id}_{dtype}_{distopt}_profile.log`
- 性能报告：`{model}_{test_id}_{dtype}_{distopt}_profile_{model}.report.rank{N}`

示例：
- `gpt2_1_bfloat16.log`
- `llama3_4_distopt_profile.log`
- `gpt2_5_bfloat16_distopt_profile_gpt2.report.rank0`

# Precision Checker 使用指南

精度检查工具，用于检测模型训练过程中的数值稳定性问题（NaN、Inf 等），支持 tensor 统计信息输出、MD5 哈希对比和 NPY 文件保存。

## 功能特性

- **自动检测 NaN/Inf**：在前向和反向传播过程中自动检测异常值
- **多级别检查**：支持 Module 级别和 Function 级别的精度检查
- **灵活配置**：通过 key=value 字符串配置所有选项
- **统计信息**：输出 tensor 的 min、max、mean 等统计值
- **MD5 哈希**：支持输出 tensor 的 MD5 值用于快速对比
- **NPY 保存**：支持保存 tensor 为 .npy 文件，便于离线分析
- **上下文追踪**：支持 GAS（梯度累积步）和层号追踪
- **多卡支持**：每个 rank 独立输出到 rank_N 目录
- **多 iter 覆盖**：同一次运行中，后续 iteration 的文件会覆盖前一个

## 配置方式

### 配置结构

```cpp
struct PrecisionCheckConfig {
    PrecisionCheckLevel level = PrecisionCheckLevel::OFF;  // 0=关闭, 1=MODULE, 2=FUNCTION
    std::string output_path = "./log_precision_check";     // 输出目录
    std::string format = "simple";                         // "simple" 或 "md5"
    bool save_tensors = false;                             // 是否保存 .npy 文件
    double md5_tolerance = 0.0;                            // MD5 量化容差（0=不量化）
};
```

### 配置选项说明

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | int | 0 | 0=关闭, 1=MODULE级别, 2=FUNCTION级别 |
| `path` | string | `./log_precision_check` | 输出目录（自动创建时间戳子目录） |
| `format` | string | `simple` | `simple`=统计信息+前6个值, `md5`=MD5哈希 |
| `save_tensors` | bool | false | 是否保存 tensor 为 .npy 文件 |
| `md5_tolerance` | double | 0.0 | MD5 量化容差（如 1e-3），0=不量化 |

### 配置字符串格式

使用 `key=value,key=value` 格式：

```bash
--precision_check "level=1,path=./my_output,format=simple,save_tensors=true"
```

## 输出格式

### 目录结构

```
log_precision_check/
└── 20260122_143052/              # 时间戳子目录 (YYYYMMDD_HHMMSS)
    ├── precision_check_rank_0.log  # 文本日志
    ├── rank_0/                     # NPY 文件目录 (save_tensors=true)
    │   ├── Block_0_forward.npy
    │   ├── Block_1_forward.npy
    │   ├── Block_0_backward.npy
    │   └── ...
    └── rank_1/                     # 多卡时每个 rank 独立目录
        ...
```

### Simple 格式 (format=simple)

```
[GAS-0] [L-0] Block_0_Forward Output tensor[0]: dtype=float32 shape=(2,1024,768) min=-2.34 max=3.56 mean=0.12 [1.23, 4.56, 7.89, ...] [NaN:0 Inf:0]
[GAS-0] [L-0] Block_0_Forward Output tensor[0]: dtype=float32 shape=(2,1024,768) min=-2.34 max=3.56 mean=0.12 [1.23, NaN, ...] [NaN:5 Inf:0] <- ERROR
```

### MD5 格式 (format=md5)

```
[GAS-0] [L-0] Block_0_Forward Output tensor[0]: dtype=float32 shape=(2,1024,768) md5=a1b2c3d4e5f6...
```

### NPY 文件命名规则

文件名格式：`{ModuleName}_{idx}_{stage}.npy`

- `ModuleName`: 模块名称（如 Block、LayerNorm）
- `idx`: 同名模块在当前 iteration 内的执行顺序索引
- `stage`: `forward` 或 `backward`

**多 iteration 行为**：每个 iteration 开始时索引重置为 0，文件会被覆盖。最终只保留最后一个 iteration 的数据。

## 命令行使用

### GPT2 示例

```bash
# 基本检查（Simple 格式，输出到文件）
./build/gpt2 --device cuda \
    --input_bin /path/to/data.bin \
    --llmc_filepath /path/to/model.bin \
    --precision_check "level=1" \
    --num_iteration 1

# 保存 NPY 文件
./build/gpt2 --device cuda \
    --input_bin /path/to/data.bin \
    --llmc_filepath /path/to/model.bin \
    --precision_check "level=1,save_tensors=true" \
    --num_iteration 1

# MD5 格式（用于快速对比）
./build/gpt2 --device cuda \
    --input_bin /path/to/data.bin \
    --llmc_filepath /path/to/model.bin \
    --precision_check "level=1,format=md5" \
    --num_iteration 1

# 自定义输出路径
./build/gpt2 --device cuda \
    --input_bin /path/to/data.bin \
    --llmc_filepath /path/to/model.bin \
    --precision_check "level=1,path=./my_precision_check,save_tensors=true" \
    --num_iteration 1
```

### LLaMA3 示例

```bash
./build/llama3 --device cuda \
    --input_bin /path/to/data.bin \
    --llmc_filepath /path/to/model.bin \
    --precision_check "level=1,save_tensors=true" \
    --num_iteration 1
```

## 离线对比工具

### precision_compare.py

用于对比两次运行的 NPY 文件：

```bash
python scripts/precision_check/precision_compare.py \
    --dir1 ./precision_check/20260122_143052 \
    --dir2 ./precision_check/20260122_143105 \
    --atol 1e-5 \
    --rtol 1e-3
```

输出示例：
```
Comparing Block_0_forward.npy:
  Shape: (2, 1024, 768) vs (2, 1024, 768) ✓
  Dtype: float32 vs float32 ✓
  Max abs diff: 1.23e-06 ✓
  Max rel diff: 2.34e-07 ✓

Summary: 433/433 files passed
```

## 测试验证

使用 `test_precision_check` 二进制进行功能验证：

```bash
# 运行全部测试（Simple/MD5格式、NPY保存、多iter覆盖）
./build/test_precision_check

# 运行特定级别测试
./build/test_precision_check "level=1"    # Module 级别
./build/test_precision_check "level=2"    # Function 级别

# 测试不同配置选项
./build/test_precision_check "level=1,format=simple"   # Simple 格式
./build/test_precision_check "level=1,format=md5"      # MD5 格式
./build/test_precision_check "level=1,save_tensors=true"  # 保存 NPY 文件
```

## 上下文追踪

使用 `PrecisionCheckContext` 设置 GAS（梯度累积步）和层号信息：

```cpp
#include "infini_train/include/utils/precision_check_context.h"

for (int gas_step = 0; gas_step < grad_accum_steps; ++gas_step) {
    PrecisionCheckContext::Instance().SetGAS(gas_step);

    for (int layer = 0; layer < num_layers; ++layer) {
        PrecisionCheckContext::Instance().SetLayer(layer);
        // 运行该层的前向传播
        // 输出会包含 [GAS-X] [L-Y] 前缀
    }
}
```

## 手动注册（高级用法）

除了通过命令行自动注册，也可以手动为特定模块注册：

```cpp
#include "infini_train/include/utils/precision_checker.h"

utils::PrecisionChecker::Config config;
config.check_nan = true;
config.check_inf = true;
config.abort_on_error = false;

// 为特定模块注册
utils::PrecisionChecker::RegisterForModule(model.get(), "MyModel", config);
```

## 实现原理

### Hook 机制

精度检查器通过 Hook 机制实现：

```
Forward Pass:
  └─> Post-Hook: 检查输出 tensor

Backward Pass:
  └─> Post-Hook: 检查梯度 tensor
```

### Counter 机制

为了支持多 iteration 文件覆盖，使用 `thread_local` 计数器：

```cpp
// 每个 iteration 开始时重置
PrecisionChecker::ResetCounters();

// 每次 CheckTensors 时递增
int idx = PrecisionCheckEnv::GetAndIncrementCounter(counter_key);
// 文件名: Block_{idx}_forward.npy
```

这确保了：
- 同一 iteration 内，同名模块有不同的索引（Block_0, Block_1, ...）
- 不同 iteration 之间，索引重置，文件被覆盖

## 使用建议

| 场景 | 推荐配置 |
|------|----------|
| 快速调试 | `level=1` |
| 详细分析 | `level=1,save_tensors=true` |
| 快速对比 | `level=1,format=md5` |
| MD5 容差对比 | `level=1,format=md5,md5_tolerance=1e-3` |
| 生产环境 | `level=0`（关闭） |

## 相关文件

- `infini_train/include/utils/precision_checker.h` - API 定义
- `infini_train/include/utils/precision_check_config.h` - 配置结构
- `infini_train/include/utils/precision_check_context.h` - 上下文追踪
- `infini_train/include/utils/global_module_hook_registry.h` - 全局模块 Hook 注册
- `scripts/precision_check/precision_compare.py` - 离线对比工具
- `test/hook/test_precision_check.cc` - 精度检查测试
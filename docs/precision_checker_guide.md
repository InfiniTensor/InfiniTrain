# Precision Checker 使用指南

精度检查工具，用于检测模型训练过程中的数值稳定性问题（NaN、Inf 等），支持 MD5 哈希对比和多种输出格式。

## 功能特性

- **自动检测 NaN/Inf**：在前向和反向传播过程中自动检测异常值
- **多级别检查**：支持 Module 级别和 Function 级别的精度检查
- **灵活配置**：通过 key=value 字符串配置所有选项
- **MD5 哈希**：支持输出 tensor 的 MD5 值用于对比
- **表格格式**：支持表格化输出，便于查看和对比
- **基准对比**：支持加载基准文件进行自动对比
- **上下文追踪**：支持 GAS（梯度累积步）和层号追踪
- **性能优化**：仅在需要时计算 MD5，避免冗余计算

## 配置方式

### 配置结构

```cpp
struct PrecisionCheckConfig {
    int level = 0;                    // 0=关闭, 1=MODULE级别, 2=FUNCTION级别
    std::string output_path = "";     // 空=控制台(仅rank0), 非空=文件(所有rank)
    bool output_md5 = false;          // 输出 MD5 还是 tensor 值
    std::string format = "simple";    // "simple" 或 "table"
    std::string baseline_path = "";   // 基准文件路径（用于对比），指定后默认开启 format=table
};
```

### 配置字符串格式

使用 `key=value,key=value` 格式：

```cpp
auto config = utils::PrecisionCheckConfig::Parse("level=2,format=table,output_md5=true");
nn::parallel::global::InitAllEnv(nthread, tp_size, sp_enabled, pp_size, vpp_size, config);
```

### 配置选项说明

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | int | 0 | 0=关闭, 1=MODULE级别, 2=FUNCTION级别 |
| `output_path` | string | "" | 空=控制台(仅rank0), 非空=文件路径(所有rank) |
| `output_md5` | bool | false | true=输出MD5哈希, false=输出tensor值 |
| `format` | string | "simple" | "simple"=简单格式, "table"=表格格式 |
| `baseline` | string | "" | 基准文件路径，用于对比 |

## 使用方法

### 1. 基本用法（简单格式）

```cpp
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/utils/precision_check_config.h"

// 启用 Function 级别检查，输出 tensor 值
auto config = utils::PrecisionCheckConfig::Parse("level=2");
nn::parallel::global::InitAllEnv(1, 1, false, 1, 1, config);

// 创建并运行模型
auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
x->Fill<float>(2.0f);
x->RequiresGrad();

auto y = x->Mul(x);
auto loss = y->Sum(0, false);
loss->Backward();
```

输出示例：
```
I0113 06:44:10.575 [Rank 0][PrecisionCheck] Forward Input MulFunction tensor[0]: [2, 2, 2, 2, 2, 2]
I0113 06:44:10.575 [Rank 0][PrecisionCheck] Forward Output MulFunction tensor[0]: [4, 4, 4, 4, 4, 4]
```

### 2. MD5 哈希输出

```cpp
// 输出 MD5 而不是 tensor 值
auto config = utils::PrecisionCheckConfig::Parse("level=2,output_md5=true");
nn::parallel::global::InitAllEnv(1, 1, false, 1, 1, config);
```

输出示例：
```
I0113 06:44:37.751 [Rank 0][PrecisionCheck] Forward Input MulFunction tensor[0]: md5=522b4223c3a2f0dd964caa87cb6eab65
I0113 06:44:37.751 [Rank 0][PrecisionCheck] Forward Output MulFunction tensor[0]: md5=91d1e78bf226d8735a3bc0ca6968339c
```

### 3. 表格格式输出

```cpp
// 使用表格格式，便于查看和对比
auto config = utils::PrecisionCheckConfig::Parse("level=2,format=table,output_md5=true");
nn::parallel::global::InitAllEnv(1, 1, false, 1, 1, config);
```

输出示例：
```
+--------------------------------------------------+-------+------------------+---------------+----------+----------+
| key                                              | level | shape            | dtype         | same_hash| diff_order|
+--------------------------------------------------+-------+------------------+---------------+----------+----------+
| [GAS-0] [L-0] Forward Input MulFunction          | 2     | (2, 3)           | float32       | True     | 0        |
| [GAS-0] [L-0] Forward Output MulFunction         | 2     | (2, 3)           | float32       | True     | 0        |
```

### 4. 基准对比

```cpp
// 第一次运行：生成基准文件
auto config1 = utils::PrecisionCheckConfig::Parse("level=2,output_md5=true,output_path=./baseline");
nn::parallel::global::InitAllEnv(1, 1, false, 1, 1, config1);
// ... 运行模型 ...
// 生成文件: ./baseline/precision_check_rank_0.log

// 第二次运行：与基准对比
auto config2 = utils::PrecisionCheckConfig::Parse("level=2,format=table,baseline=./baseline/precision_check_rank_0.log");
nn::parallel::global::InitAllEnv(1, 1, false, 1, 1, config2);
// ... 运行模型 ...
// 输出会显示 same_hash 列，标识是否与基准一致
```

### 5. 文件输出（所有 rank）

```cpp
// 输出到文件，所有 rank 都会输出
auto config = utils::PrecisionCheckConfig::Parse("level=2,output_path=./logs");
nn::parallel::global::InitAllEnv(8, 2, false, 2, 1, config);
// 生成文件: ./logs/precision_check_rank_0.log, ./logs/precision_check_rank_1.log, ...
```

## 命令行使用

### GPT2 示例

```bash
# 基本检查（简单格式，输出 tensor 值）
./gpt2 <other params> --precision_check "level=2"

# 输出 MD5 哈希
./gpt2 <other params> --precision_check "level=2,output_md5=true"

# 表格格式
./gpt2 <other params> --precision_check "level=2,format=table,output_md5=true"

# 生成基准文件
./gpt2 <other params> --precision_check "level=2,output_md5=true,output_path=./baseline"

# 与基准对比
./gpt2 <other params> --precision_check "level=2,format=table,baseline=./baseline/precision_check_rank_0.log"
```

### LLaMA3 示例

```bash
# 基本检查
./llama3 <other params> --precision_check "level=2"

# 表格格式 + MD5
./llama3 <other params> --precision_check "level=2,format=table,output_md5=true"
```

## 上下文追踪

使用 `PrecisionCheckContext` 设置 GAS（梯度累积步）和层号信息：

```cpp
#include "infini_train/include/utils/precision_check_context.h"

// 在训练循环中设置上下文
for (int gas_step = 0; gas_step < grad_accum_steps; ++gas_step) {
    PrecisionCheckContext::Instance().SetGAS(gas_step);

    for (int layer = 0; layer < num_layers; ++layer) {
        PrecisionCheckContext::Instance().SetLayer(layer);
        PrecisionCheckContext::Instance().SetLayerName("transformer_block");

        // 运行该层的前向传播
        // 输出会包含 [GAS-X] [L-Y] 前缀
    }
}
```

输出示例：
```
[GAS-0] [L-0] Forward Input MulFunction
[GAS-0] [L-1] Forward Input MulFunction
[GAS-1] [L-0] Forward Input MulFunction
```

## 性能优化

### MD5 计算优化

MD5 仅在以下情况计算：
- `output_md5=true` 时
- `baseline_path` 非空时（需要对比）

默认情况下（`output_md5=false` 且无基准对比），不会计算 MD5，避免性能开销。

### 使用建议

| 场景 | 推荐配置 |
|------|----------|
| 快速调试 | `level=2` |
| 详细调试 | `level=2,format=table` |
| 生成基准 | `level=2,output_md5=true,output_path=./baseline` |
| 对比测试 | `level=2,format=table,baseline=./baseline/...` |
| 生产环境 | `level=0`（关闭） |

## 输出格式对比

### Simple 格式

```
I0113 06:44:10.575 [Rank 0][PrecisionCheck] Forward Input MulFunction tensor[0]: [2, 2, 2, 2, 2, 2]
```

优点：紧凑，易于阅读
缺点：不便于对比多个 tensor

### Table 格式

```
+--------------------------------------------------+-------+------------------+---------------+----------+----------+
| key                                              | level | shape            | dtype         | same_hash| diff_order|
+--------------------------------------------------+-------+------------------+---------------+----------+----------+
| [GAS-0] [L-0] Forward Input MulFunction          | 2     | (2, 3)           | float32       | True     | 0        |
```

优点：结构化，便于对比和分析
缺点：占用更多空间

## 手动注册（高级用法）

除了通过 `InitAllEnv` 自动注册，也可以手动为特定模块注册：

```cpp
#include "infini_train/include/utils/precision_checker.h"

// 配置精度检查器
utils::PrecisionChecker::Config config;
config.check_nan = true;
config.check_inf = true;
config.print_stats = true;
config.abort_on_error = false;

// 为特定模块注册
utils::PrecisionChecker::RegisterForModule(model.get(), "MyModel", config);

// 为特定 Function 注册
utils::PrecisionChecker::RegisterForFunction(my_function.get(), "MyFunction", config);
```

## 实现原理

精度检查器通过 Hook 机制实现：

1. **Forward Pre-Hook**：检查输入 tensor
2. **Forward Post-Hook**：检查输出 tensor
3. **Backward Hooks**：自动检查梯度

检查流程：
```
Forward Pass:
  ├─> Pre-Hook: 检查输入
  ├─> Forward: 执行计算
  └─> Post-Hook: 检查输出

Backward Pass:
  ├─> Backward Pre-Hook: 检查梯度输入
  ├─> Backward: 执行梯度计算
  └─> Backward Post-Hook: 检查梯度输出
```

## 示例代码

参见：
- `test/hook/test_precision_check.cc` - 完整使用示例
- `infini_train/include/utils/precision_checker.h` - API 文档
- `infini_train/include/utils/precision_check_config.h` - 配置结构
- `infini_train/include/utils/precision_check_context.h` - 上下文追踪

## 测试

```bash
# 运行测试（默认：简单格式）
./test_precision_check

# Function 级别 + MD5
./test_precision_check "level=2,output_md5=true"

# 表格格式
./test_precision_check "level=2,format=table,output_md5=true"

# Module 级别
./test_precision_check "level=1"
```

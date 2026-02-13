# LoRA 使用说明

本文档介绍如何在 InfiniTrain 中使用 LoRA (Low-Rank Adaptation) 进行高效微调。

## 目录

1. [快速开始](#快速开始)
2. [核心概念](#核心概念)
3. [命令行使用](#命令行使用-gpt2-示例)
4. [LoRAModel 包装器](#lora模型-包装器-推荐模式)
5. [API 参考](#api-参考)
6. [使用示例](#使用示例)
7. [最佳实践](#最佳实践)

## 快速开始

### 头文件引入

```cpp
#include "nn/lora/lora_config.h"
#include "nn/lora/lora_linear.h"
#include "nn/lora/lora_utils.h"
// 如果使用张量并行
#include "nn/lora/lora_parallel_linear.h"
```

### 最简示例

```cpp
using namespace infini_train::nn::lora;

// 1. 创建 LoRA 配置
LoRAConfig config;
config.rank = 8;       // 低秩维度
config.alpha = 16.0f;  // 缩放因子

// 2. 获取 LoRA 模型
auto* lora_model = GetLoRAModel(model, config);

// 3. 获取 LoRA 参数用于优化器
auto lora_params = lora_model->TrainableParameters();
auto optimizer = std::make_shared<Adam>(lora_params, lr);

// 4. 训练循环
for (int step = 0; step < num_steps; ++step) {
    auto loss = (*model)(inputs);
    loss->Backward();
    optimizer->Step();
    optimizer->ZeroGrad();
}

// 6. 保存 LoRA 权重
SaveLoRAWeights(model, "lora_weights.bin");
```

## 核心概念

### LoRA 原理

LoRA 通过低秩分解来近似权重更新：

```
原始: y = Wx + b
LoRA: y = Wx + b + (α/r) × x × A^T × B^T
```

其中：
- `W` 是冻结的原始权重
- `A` 是形状为 `[rank, in_features]` 的可训练矩阵
- `B` 是形状为 `[out_features, rank]` 的可训练矩阵
- `α/r` 是缩放因子

### 参数效率

假设原始 Linear 层参数量为 `in × out`，LoRA 只需训练 `rank × (in + out)` 个参数。

例如：`in=4096, out=4096, rank=8`
- 原始参数：16,777,216
- LoRA 参数：65,536 (仅 0.39%)

## LoRAModel 包装器类

### LoRAModel

遵循 PEFT 模式的 LoRA 包装器，封装基础模型和 LoRA 配置。使用 `NamedModules()` 自动遍历模型层次结构。

```cpp
class LoRAModel : public Module {
public:
    // 构造函数 - 自动遍历模型层次结构
    LoRAModel(std::shared_ptr<Module> base_model,
              const LoRAConfig &config);

    // 获取可训练参数
    std::vector<std::shared_ptr<Tensor>> TrainableParameters() const;

    // 获取所有参数
    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

    // LoRA 权重管理
    void SaveLoRA(const std::string &filepath) const;
    void LoadLoRA(const std::string &filepath);
    void Merge();
    void Unmerge();
    bool IsMerged() const;

    // 打印摘要
    void PrintSummary() const;

    // 访问基础模型
    std::shared_ptr<Module> base_model() const;

    // 获取 LoRA 配置
    const LoRAConfig &config() const;
};
```

### 工厂函数

```cpp
template <typename ModelType, typename ConfigType>
std::shared_ptr<LoRAModel> CreateLoRAModel(
    const ConfigType &model_config,
    const LoRAConfig &lora_config) {
    auto base_model = std::make_shared<ModelType>(model_config);
    return std::make_shared<LoRAModel>(base_model, lora_config);
}
```

## API 参考

### LoRAConfig - 配置结构

```cpp
struct LoRAConfig {
    int64_t rank = 8;       // 低秩维度 r
    float alpha = 16.0f;    // 缩放因子 α
    float dropout = 0.0f;   // Dropout 概率（暂未实现）

    // 目标模块名称（默认只对 attention 层应用）
    std::unordered_set<std::string> target_modules = {"c_attn", "attn.c_proj"};

    // 初始化参数
    bool use_kaiming_a = true;     // A 矩阵使用 Kaiming 初始化
    float kaiming_a_param = 1.0f;  // Kaiming 初始化参数

    // 计算缩放因子
    float Scaling() const;  // 返回 alpha / rank

    // 检查模块是否应该应用 LoRA
    bool ShouldApplyLoRA(const std::string &module_name) const;
};
```

### 模型应用函数

#### GetLoRAModel

PEFT-style 运行时包装器，使用 `NamedModules()` 自动遍历模型层次结构，创建 LoRA 模型。

```cpp
LoRAModel* GetLoRAModel(
    std::shared_ptr<Module> model,           // 目标模型
    const LoRAConfig &config                 // LoRA 配置
);
```

**参数说明：**
- `model`: 要包装的模型
- `config`: LoRA 配置（包含 `target_modules` 指定目标层）

**返回值：** `LoRAModel*`，可用于调用 `LoadLoRA()`, `SaveLoRA()`, `PrintSummary()` 等方法

**使用示例：**
```cpp
// 配置 LoRA
LoRAConfig config{8, 16.0f};
config.SetTargetModules("c_attn,attn.c_proj");       // 只对 attention
// config.SetTargetModules("c_attn,attn.c_proj,c_fc,c_fc2,mlp.c_proj");  // 包含 MLP

// 一行启用 LoRA
auto* lora_model = nn::lora::GetLoRAModel(model, config);
```

#### InjectLoRALayers

使用 `NamedModules()` 自动遍历模型层次结构，将 LoRA 注入到所有匹配的层中。

```cpp
void InjectLoRALayers(
    std::shared_ptr<Module> model,           // 目标模型
    const LoRAConfig &config                 // LoRA 配置
);
```

**参数说明：**
- `model`: 要注入 LoRA 的模型
- `config`: LoRA 配置（通过 `target_modules` 指定目标层）

### 参数管理函数

#### FreezeBaseModel / UnfreezeModel

```cpp
// 冻结基础模型所有参数
void FreezeBaseModel(std::shared_ptr<Module> model);

// 解冻所有参数
void UnfreezeModel(std::shared_ptr<Module> model);
```

#### GetLoRAParameters / GetBaseParameters

```cpp
// 获取 LoRA 参数（用于优化器）
std::vector<std::shared_ptr<Tensor>> GetLoRAParameters(
    const std::shared_ptr<Module> &model);

// 获取基础模型参数
std::vector<std::shared_ptr<Tensor>> GetBaseParameters(
    const std::shared_ptr<Module> &model);
```

### 权重合并函数

#### MergeLoRAWeights / UnmergeLoRAWeights

```cpp
// 合并 LoRA 权重到基础权重: W' = W + (α/r) × B × A
void MergeLoRAWeights(std::shared_ptr<Module> model);

// 恢复原始基础权重
void UnmergeLoRAWeights(std::shared_ptr<Module> model);
```

**使用场景：**
- 推理时合并权重可以消除额外计算开销
- 导出模型时合并权重得到标准模型格式

### 保存/加载函数

```cpp
// 保存 LoRA 权重到文件
void SaveLoRAWeights(const std::shared_ptr<Module> &model,
                     const std::string &filepath);

// 从文件加载 LoRA 权重
void LoadLoRAWeights(std::shared_ptr<Module> model,
                     const std::string &filepath);

// 获取 LoRA 状态字典
std::unordered_map<std::string, std::shared_ptr<Tensor>>
LoRAStateDict(const std::shared_ptr<Module> &model);

// 加载 LoRA 状态字典
void LoadLoRAStateDict(
    std::shared_ptr<Module> model,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);
```

### 统计函数

```cpp
// 打印 LoRA 模型摘要
void PrintLoRASummary(const std::shared_ptr<Module> &model);

// 统计可训练参数数量
int64_t CountTrainableParameters(const std::shared_ptr<Module> &model);

// 统计总参数数量
int64_t CountTotalParameters(const std::shared_ptr<Module> &model);
```

## 使用示例

### 示例 1: GPT2 微调

```cpp
#include "example/gpt2/gpt2.h"
#include "nn/lora/lora_utils.h"

using namespace infini_train::nn::lora;

int main() {
    // 创建 GPT2 模型
    auto model = std::make_shared<GPT2>(config);
    model->LoadWeights("gpt2_weights.bin");

    // 配置 LoRA
    LoRAConfig lora_config;
    lora_config.rank = 8;
    lora_config.alpha = 16.0f;
    lora_config.SetTargetModules("c_attn,attn.c_proj");  // 只对 attention 层

    // 获取 LoRA 模型
    auto* lora_model = GetLoRAModel(model, lora_config);

    // 打印参数统计
    PrintLoRASummary(lora_model);
    // 输出示例:
    // ========== LoRA Model Summary ==========
    // Total parameters:     124,439,808
    // Trainable parameters: 294,912 (0.24%)
    // Frozen parameters:    124,144,896
    // =========================================

    // 创建优化器（只优化 LoRA 参数）
    auto lora_params = lora_model->TrainableParameters();
    auto optimizer = std::make_shared<Adam>(lora_params, /*lr=*/1e-4);

    // 训练循环
    for (int step = 0; step < num_steps; ++step) {
        auto [loss, logits] = (*lora_model)({input_ids});
        loss->Backward();
        optimizer->Step();
        optimizer->ZeroGrad();

        if (step % 100 == 0) {
            std::cout << "Step " << step << ", Loss: " << loss->Item<float>() << std::endl;
        }
    }

    // 保存 LoRA 权重（仅几 MB）
    lora_model->SaveLoRA("gpt2_lora.bin");

    return 0;
}
```

### 示例 2: LLaMA3 分布式微调

```cpp
#include "example/llama3/llama3.h"
#include "nn/lora/lora_utils.h"
#include "nn/parallel/process_group.h"

using namespace infini_train::nn::lora;

int main(int argc, char **argv) {
    // 初始化分布式环境
    InitDistributed(argc, argv);

    // 创建 LLaMA3 模型（带张量并行）
    LLaMA3Config config;
    config.n_layers = 32;
    config.tensor_parallel = 2;

    auto model = std::make_shared<LLaMA3>(config);
    model->LoadWeights("llama3_weights/");

    // 配置 LoRA（包含 MLP 层以获得更好效果）
    LoRAConfig lora_config{16, 32.0f};
    lora_config.SetTargetModules("c_attn,attn.c_proj,c_fc,c_fc2,mlp.c_proj");

    // 获取 LoRA 模型（通过 target_modules 配置包含 MLP 层）
    auto* lora_model = GetLoRAModel(model, lora_config);

    PrintLoRASummary(lora_model);

    // 训练...

    // 保存
    if (GetRank() == 0) {
        SaveLoRAWeights(model, "llama3_lora.bin");
    }

    return 0;
}
```

### 示例 3: 推理时合并权重

```cpp
// 加载基础模型
auto model = std::make_shared<GPT2>(config);
model->LoadWeights("gpt2_weights.bin");

// 获取 LoRA 模型
auto* lora_model = GetLoRAModel(model, lora_config);

// 加载 LoRA 权重
lora_model->LoadLoRA("gpt2_lora.bin");

// 合并权重（推理时无额外开销）
lora_model->Merge();

// 现在可以像普通模型一样推理
auto output = (*lora_model)({input_ids});

// 如果需要继续训练，先解除合并
lora_model->Unmerge();
```

### 示例 4: 自定义目标层

```cpp
// 或者对所有线性层应用
config.SetTargetModules("c_attn,attn.c_proj,c_fc,c_fc2,mlp.c_proj,lm_head");

// 获取 LoRA 模型
auto* lora_model = GetLoRAModel(model, config);
```

## 最佳实践

### 1. 选择合适的 rank

| 任务类型 | 推荐 rank | 说明 |
|---------|----------|------|
| 简单分类任务 | 4-8 | 参数少，训练快 |
| 文本生成微调 | 8-16 | 平衡效果和效率 |
| 复杂任务适配 | 16-64 | 更强表达能力 |

### 2. alpha 设置

- 通常设置 `alpha = 2 × rank`
- 较大的 alpha 会增加 LoRA 的影响
- 可以通过调整 alpha 来控制微调强度

### 3. 目标层选择

```cpp
// 推荐：只对 attention 层（参数效率最高）
config.SetTargetModules("c_attn,attn.c_proj");

// 可选：包含 MLP 层（效果可能更好，但参数更多）
config.SetTargetModules("c_attn,attn.c_proj,c_fc,c_fc2,mlp.c_proj");
```

### 4. 学习率

- LoRA 通常使用比全量微调更高的学习率
- 推荐范围：1e-4 到 1e-3
- 可以使用学习率预热和衰减

### 5. 内存优化

```cpp
// 只保存 LoRA 权重（几 MB vs 几 GB）
SaveLoRAWeights(model, "lora.bin");

// 推理时合并权重，消除额外计算
MergeLoRAWeights(model);
```

## 模型层名称参考

### GPT2 模型结构

```
transformer.wte          # Token Embedding
transformer.wpe          # Position Embedding
transformer.h.{i}.ln_1   # LayerNorm 1
transformer.h.{i}.attn.c_attn   # QKV 投影 (ColumnParallel)
transformer.h.{i}.attn.c_proj   # Output 投影 (RowParallel)
transformer.h.{i}.ln_2   # LayerNorm 2
transformer.h.{i}.mlp.c_fc      # MLP 第一层 (ColumnParallel)
transformer.h.{i}.mlp.c_proj    # MLP 第二层 (RowParallel)
transformer.ln_f         # Final LayerNorm
lm_head                  # Language Model Head
```

### LLaMA3 模型结构

```
transformer.tok_emb      # Token Embedding
transformer.h.{i}.attn_norm     # RMSNorm (attention)
transformer.h.{i}.attn.c_attn   # QKV 投影 (ColumnParallel)
transformer.h.{i}.attn.c_proj   # Output 投影 (RowParallel)
transformer.h.{i}.ffn_norm      # RMSNorm (FFN)
transformer.h.{i}.mlp.c_fc      # FFN gate (ColumnParallel)
transformer.h.{i}.mlp.c_fc2     # FFN up (ColumnParallel)
transformer.h.{i}.mlp.c_proj    # FFN down (RowParallel)
transformer.norm         # Final RMSNorm
lm_head                  # Language Model Head
```

## 命令行使用 (GPT2 示例)

### 启用 LoRA 训练

```bash
./build/gpt2 \
    --device cuda \
    --input_bin data/train.bin \
    --llmc_filepath data/gpt2_124M.bin \
    --batch_size 4 \
    --sequence_length 64 \
    --num_iteration 10 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_target_modules "c_attn,attn.c_proj" \
    --lora_save_path data/lora_weights
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_rank` | 0 | LoRA 秩 (0 = 禁用) |
| `--lora_alpha` | 16.0 | LoRA 缩放因子 |
| `--lora_target_modules` | "c_attn,attn.c_proj" | 目标模块 (逗号分隔: c_attn,attn.c_proj,c_fc,c_fc2,mlp.c_proj) |
| `--lora_load_path` | "" | 加载已有 LoRA 权重 |
| `--lora_save_path` | "" | 保存 LoRA 权重路径 |

### 加载已有 LoRA 权重

```bash
./build/gpt2 \
    ...
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_load_path data/lora_weights
```

## LoRAModel 包装器 (推荐模式)

### 概述

`LoRAModel` 是一个包装器类，遵循 PEFT 设计模式，将 LoRA 作为包装器应用于基础模型，而不是直接修改模型代码。

### 优势

- **透明性**: 训练循环无需修改，直接使用 `(*model)(inputs)`
- **参数管理**: 自动获取可训练参数
- **权重管理**: 内置 Save/Load/Merge 方法

### 使用示例

```cpp
#include "infini_train/include/nn/lora/lora_model.h"

using namespace infini_train::nn::lora;

int main() {
    // 1. 创建基础模型
    auto base_model = std::make_shared<GPT2>(config);
    base_model->LoadWeights("gpt2_weights.bin");

    // 2. 创建 LoRA 配置
    LoRAConfig lora_config{8, 16.0f};
    lora_config.SetTargetModules("c_attn,attn.c_proj");  // 只对 attention 层

    // 3. 创建 LoRA 包装器 (一行代码)
    auto lora_model = std::make_shared<LoRAModel>(base_model, lora_config);

    // 4. 获取可训练参数用于优化器
    auto trainable_params = lora_model->TrainableParameters();
    auto optimizer = std::make_shared<Adam>(trainable_params, 1e-5);

    // 5. 打印摘要
    lora_model->PrintSummary();
    // 输出:
    // ========== LoRA Model Summary ==========
    // Total parameters:     176062464
    // Trainable parameters: 442368 (0.251256%)
    // Frozen parameters:    175620096
    // =========================================

    // 6. 训练循环 (无需修改)
    for (int step = 0; step < num_steps; ++step) {
        auto logits = (*lora_model)({x, y})[0];
        auto loss = (*loss_fn)({logits, y})[0];
        loss->Backward();
        optimizer->Step();
        optimizer->ZeroGrad();
    }

    // 7. 保存 LoRA 权重
    lora_model->SaveLoRA("lora_weights.bin");

    return 0;
}
```

### 工厂函数

对于任意模型类型，可以使用模板工厂函数：

```cpp
#include "infini_train/include/nn/lora/lora_model.h"

auto lora_model = CreateLoRAModel<GPT2, GPT2Config>(
    model_config,           // GPT2 模型配置
    lora_config             // LoRA 配置
);
```

## 常见问题

### Q: LoRA 权重文件有多大？

A: 取决于 rank 和目标层数量。以 GPT2-small (12层) 为例：
- rank=8, attention only: ~1.2 MB
- rank=16, attention + MLP: ~4.8 MB

### Q: 如何在不同任务间切换 LoRA？

A: 保存和加载不同的 LoRA 权重文件：
```cpp
// 任务 A
LoadLoRAWeights(model, "task_a_lora.bin");
// 推理...

// 任务 B
LoadLoRAWeights(model, "task_b_lora.bin");
// 推理...
```

### Q: 可以同时使用多个 LoRA 吗？

A: 当前实现不支持多 LoRA 组合。如需此功能，可以：
1. 合并多个 LoRA 权重后加载
2. 扩展实现支持 LoRA 堆叠

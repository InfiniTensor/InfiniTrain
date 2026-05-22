# LoRA 使用说明

本文档描述 InfiniTrain 当前 LoRA 实现的实际用法。当前实现采用 PEFT-style **原地注入**：
`GetLoRAModel(model, config)` 会替换匹配的 Linear 模块、冻结非 LoRA 参数，并返回修改后的
`std::shared_ptr<Module>`。仓库里没有独立的 `LoRAModel` 包装器类或 `lora_model.h`。

## 快速开始

### 头文件

```cpp
#include "infini_train/include/nn/lora/lora_config.h"
#include "infini_train/include/nn/lora/lora_linear.h"
#include "infini_train/include/nn/lora/lora_parallel_linear.h"
#include "infini_train/include/nn/lora/lora_utils.h"
```

### 最小示例

```cpp
using namespace infini_train::nn::lora;

LoRAConfig config;
config.rank = 8;
config.alpha = 16.0f;
config.target_modules = ParseLoRATargetModules("c_attn,attn.c_proj");

// 原地注入 LoRA，并冻结非 LoRA 参数。
model = GetLoRAModel(model, config);
PrintLoRASummary(model);

auto params = GetLoRAParameters(model);
auto optimizer = optimizers::Adam::Create(/*learning_rate=*/1e-4)(params);

for (int step = 0; step < num_steps; ++step) {
    optimizer->ZeroGrad();
    auto logits = (*model)({input})[0];
    auto loss = (*loss_fn)({logits, labels})[0];
    loss->Backward();
    optimizer->Step();
}

SaveLoRAWeights(model, "adapter_lora.bin");
```

## 核心行为

LoRA 对 Linear 层追加低秩增量：

```text
y = x @ W^T + b + (alpha / rank) * x @ A^T @ B^T
```

- `W` 和 `bias` 来自原基础层，注入后默认冻结。
- `lora_A` 形状为 `[rank, in_features]`，默认 Kaiming uniform 初始化。
- `lora_B` 形状为 `[out_features, rank]`，零初始化，因此注入初始时不改变基础模型输出。
- LoRA 参数当前固定创建为 `float32`。
- `dropout` 字段存在于 `LoRAConfig`，当前未实现。

`GetLoRAModel` 会遍历 `NamedModules()`，按 `target_modules` 匹配模块路径：

- 匹配规则是完整组件后缀匹配，例如 `attn.c_proj` 可匹配
  `transformer.h.0.attn.c_proj`。
- `c_proj` 会同时匹配 attention 和 MLP 中名为 `c_proj` 的层。
- `attn.c_proj` 只匹配 attention output projection。
- 若根模块本身匹配，返回值可能是新的 LoRA 模块，因此应使用返回的 `model`。

## API 参考

### LoRAConfig

```cpp
struct LoRAConfig {
    int64_t rank = 8;
    float alpha = 16.0f;
    float dropout = 0.0f; // not implemented
    std::unordered_set<std::string> target_modules = {"c_attn", "c_proj"};
    bool use_kaiming_a = true;
    float kaiming_a_param = sqrtf(5.0f);

    LoRAConfig() = default;
    LoRAConfig(int64_t r, float a, float d,
               const std::unordered_set<std::string> &targets);

    float Scaling() const;
    bool ShouldApplyLoRA(const std::string &module_name) const;
};
```

推荐用法：

```cpp
LoRAConfig config;
config.rank = 8;
config.alpha = 16.0f;
config.target_modules = ParseLoRATargetModules("c_attn,attn.c_proj");
```

### 注入与参数管理

```cpp
std::shared_ptr<Module> GetLoRAModel(std::shared_ptr<Module> model,
                                     const LoRAConfig &config);

std::shared_ptr<Module> InjectLoRALayers(std::shared_ptr<Module> model,
                                         const LoRAConfig &config);

void FreezeBaseModel(std::shared_ptr<Module> model);
void UnfreezeModel(std::shared_ptr<Module> model);

std::vector<std::shared_ptr<Tensor>>
GetLoRAParameters(const std::shared_ptr<Module> &model);

std::vector<std::shared_ptr<Tensor>>
GetBaseParameters(const std::shared_ptr<Module> &model);
```

- `GetLoRAModel` = 注入 + 冻结基础参数 + 重新打开 LoRA 参数。
- `InjectLoRALayers` 只做结构替换，不冻结参数。
- 示例训练入口在启用 LoRA 时只把 `GetLoRAParameters(model)` 传给优化器。
- 对 merged 状态的 LoRA 模块调用 `GetLoRAParameters` 会触发检查；继续训练前先
  `UnmergeLoRAWeights(model)`。

### 合并、卸载、保存和加载

```cpp
void MergeLoRAWeights(std::shared_ptr<Module> model);
void UnmergeLoRAWeights(std::shared_ptr<Module> model);
std::shared_ptr<Module> MergeAndUnload(std::shared_ptr<Module> model);

std::unordered_map<std::string, std::shared_ptr<Tensor>>
LoRAStateDict(const std::shared_ptr<Module> &model);

void LoadLoRAStateDict(
    std::shared_ptr<Module> model,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);

void SaveLoRAWeights(const std::shared_ptr<Module> &model,
                     const std::string &filepath);

void LoadLoRAWeights(std::shared_ptr<Module> model,
                     const std::string &filepath);
```

- `MergeLoRAWeights` 将 `W += (alpha / rank) * B @ A` 写回基础权重，并冻结 LoRA 参数。
- `UnmergeLoRAWeights` 撤销一次已合并的 LoRA 增量，并恢复 LoRA 参数可训练。
- `MergeAndUnload` 会先合并，再把 LoRA 模块替换回普通 `Linear` / TP Linear，并解冻返回模型参数。
- `SaveLoRAWeights` 只保存名字包含 `lora_A` / `lora_B` 的参数。
- LoRA 权重文件是二进制文件，包含 magic `"LORA"`、version `1`、tensor 名称、维度和 float 数据。
- 加载前模型必须已经用相同目标层注入 LoRA；找不到的 LoRA 参数会打印 warning。
- 加载时如果文件里的 tensor 和当前参数同形状，会直接拷贝；如果当前参数是 TP 分片，会按第一个不同维度用 `parallel::tp_rank` 切片后再拷贝。

### 统计和解析工具

```cpp
int64_t CountTrainableParameters(const std::shared_ptr<Module> &model);
int64_t CountTotalParameters(const std::shared_ptr<Module> &model);
void PrintLoRASummary(const std::shared_ptr<Module> &model,
                      int global_rank = -1);

std::unordered_set<std::string>
ParseLoRATargetModules(const std::string &targets);
```

`ParseLoRATargetModules("c_attn, attn.c_proj")` 会去掉空白并忽略空项。

## 支持的模块

`GetLoRAModel` 会自动替换匹配到的线性层，目前支持：

- `nn::Linear`
- `parallel::ColumnParallelLinear`
- `parallel::RowParallelLinear`

如果模型用了 TP，不需要手动创建 `LoRAColumnParallelLinear` 或 `LoRARowParallelLinear`；
只要正常调用 `GetLoRAModel(model, config)`，注入逻辑会根据基础层类型自动选择对应的 LoRA 实现。

加载 LoRA 权重时，如果文件里保存的是完整 tensor，而当前模型在 TP 下只需要某个 rank 的分片，加载函数会自动按当前 `tp_rank` 切片。

## 命令行示例

GPT2 和 Llama3 示例都通过 `--lora_rank > 0` 启用 LoRA，并在训练结束时按需保存。

### GPT2

```bash
./build/gpt2 \
  --device cuda \
  --input_bin data/train.bin \
  --llmc_filepath data/gpt2_124M.bin \
  --batch_size 4 \
  --sequence_length 64 \
  --num_iteration 10 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --lora_target_modules "c_attn,attn.c_proj" \
  --lora_save_path gpt2_lora.bin
```

GPT2 默认 LoRA target 是 `"c_attn,c_proj"`。

### Llama3

```bash
./build/llama3 \
  --device cuda \
  --input_bin data/train.bin \
  --llmc_filepath data/llama3.bin \
  --batch_size 4 \
  --sequence_length 64 \
  --num_iteration 10 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --lora_target_modules "c_attn,c_proj,c_fc,c_fc2" \
  --lora_save_path llama3_lora.bin
```

Llama3 默认 LoRA target 是 `"c_attn,c_proj,c_fc,c_fc2"`。

### 加载已有 LoRA 权重

```bash
./build/gpt2 \
  --device cuda \
  --input_bin data/train.bin \
  --llmc_filepath data/gpt2_124M.bin \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --lora_target_modules "c_attn,attn.c_proj" \
  --lora_load_path gpt2_lora.bin
```

加载时请保持 `rank` 和 `lora_target_modules` 与保存这份 LoRA 权重时一致。

### 参数表

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--lora_rank` | `0` | LoRA rank；`0` 表示禁用 LoRA |
| `--lora_alpha` | `16.0` | LoRA scaling 分子，实际缩放为 `alpha / rank` |
| `--lora_target_modules` | 模型相关 | 逗号分隔的目标模块后缀 |
| `--lora_load_path` | `""` | 启动时加载 LoRA 权重文件 |
| `--lora_save_path` | `""` | 训练结束后保存 LoRA 权重文件 |

示例入口在启用 LoRA 后会：

1. 构造 `LoRAConfig{rank, alpha, 0.0f, ParseLoRATargetModules(...)}`。
2. 调用 `model = GetLoRAModel(model, lora_config)`。
3. 如果设置 `--lora_load_path`，调用 `LoadLoRAWeights`。
4. 使用 `GetLoRAParameters(model)` 构建优化器参数列表。
5. 如果设置 `--lora_save_path`，训练结束后调用 `SaveLoRAWeights`。

## 目标模块建议

常用配置：

```cpp
// GPT2 attention QKV + attention output projection.
config.target_modules = ParseLoRATargetModules("c_attn,attn.c_proj");

// GPT2/Llama3 覆盖所有名为 c_proj 的层，包括 attention 和 MLP output。
config.target_modules = ParseLoRATargetModules("c_attn,c_proj");

// 包含 MLP 层。
config.target_modules = ParseLoRATargetModules("c_attn,attn.c_proj,c_fc,c_fc2");
```

注意 `c_proj` 是宽匹配，`attn.c_proj` 是更精确匹配。若只想命中 attention output
projection，不要只写 `c_proj`。

## 测试

LoRA 单元测试位于 `tests/lora/test_lora.cc`，覆盖：

- `LoRAConfig::Scaling` 和 target 后缀匹配。
- `LoRALinear` 注入、forward、merge/unmerge。
- `GetLoRAParameters`、参数统计、freeze/unfreeze。
- `LoRAStateDict` 和 `MergeAndUnload`。

构建并运行：

```bash
cmake -S . -B build -DBUILD_TEST=ON -DUSE_CUDA=ON -DUSE_NCCL=ON
cmake --build build -j
ctest --test-dir build -R LoRATest --output-on-failure
```

`scripts/test_config.json` 还包含 `lora` 测试组，覆盖 fp32/bfloat16、LoRA 权重加载，以及 DP、TP、SP、PP 组合。

## 常见注意事项

- 直接用 `GetLoRAModel` 就好；现在没有单独的 `LoRAModel` 类，也没有 `lora_model.h`。
- `LoRAConfig` 先默认创建再填字段；如果想一行构造，需要传完整的 4 个参数。
- 保存和加载传的是具体文件名，比如 `gpt2_lora.bin`，不是目录。
- LoRA 文件只存 `lora_A` / `lora_B`，不会保存基础模型权重。
- TP 运行时加载 LoRA 权重，如果文件里是完整 tensor、当前 rank 只需要分片，加载函数会自动按当前 `tp_rank` 切一段。
- `MergeLoRAWeights` 适合推理前用；如果还要继续训练，先调用 `UnmergeLoRAWeights`。
- `MergeAndUnload` 会把 LoRA 模块变回普通 Linear，之后模型里就没有 `lora_A` / `lora_B` 了。
- 现在一次只支持一套 LoRA adapter，不支持多个 adapter 叠加。

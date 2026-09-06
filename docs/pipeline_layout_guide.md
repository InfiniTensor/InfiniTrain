# Pipeline 并行自定义布局使用说明

本文档描述 InfiniTrain 新增的 Pipeline 自定义布局能力：通过 `--pipeline_layer_partition`
显式指定每个 Pipeline Stage 的 Transformer 层数，并让模型构建、Pipeline 调度与参数加载统一
使用同一份 `PipelineLayout`，避免层归属逻辑在多处重复实现。

## 快速开始

```bash
./build/infini_run \
  --nproc_per_node=4 \
  ./build/gpt2 \
    --device cuda \
    --input_bin data/train.bin \
    --llmc_filepath data/gpt2_124M.bin \
    --pipeline_parallel 4 \
    --pipeline_layer_partition 4,8,6,6
```

上述配置把 24 个 Transformer 层依次划分为：

```text
stage 0: embedding + layers 0-3
stage 1: layers 4-11
stage 2: layers 12-17
stage 3: layers 18-23 + final_norm + lm_head
```

## 参数配置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--pipeline_parallel` | `1` | Pipeline Stage 数量 |
| `--virtual_pipeline_parallel` | `1` | 每个 Stage 的 virtual chunk 数量（vPP） |
| `--pipeline_layer_partition` | `""` | 逗号分隔的各 Stage 层数列表，例如 `4,8,6,6` |

GPT2 与 LLaMA3 示例入口均支持 `--pipeline_layer_partition`，解析后写入全局环境
`GlobalEnv`，模型构建、`PipelineParallel` 包装、调度器与 checkpoint 加载都从同一布局查询层归属。

## 布局语法

- 语法为逗号分隔的正整数列表，例如 `4,8,6,6`。
- 列表长度必须等于 `--pipeline_parallel` 的 Stage 数量。
- 各 Stage 层数之和必须等于模型总层数（GPT2-124M 为 12 层，需先选定层数与 Stage 数匹配的模型）。
- 布局是「连续划分」：`stage i` 拥有编号从 `sum(前 i 项)` 到 `sum(前 i+1 项)` 的连续 Transformer 层。
- Embedding 固定归属第一个 Stage，Final Norm + LM Head 固定归属最后一个 Stage（当前版本不开放单独配置）。

## 默认行为

不传 `--pipeline_layer_partition`（或传空串）时，保持原有自动均匀划分：

```text
layers_per_chunk = total_layers / (num_stages * vpp_size)
remainder        = total_layers % (num_stages * vpp_size)
```

余数按 global chunk 顺序依次多分配一层，vPP 下各 Stage 按
`global_chunk = local_chunk * num_stages + stage` 交错持有多个层范围。因此默认均匀布局与
vPP 完全兼容；自定义布局当前要求 `--virtual_pipeline_parallel 1`（二者不兼容，会报错）。

## 输入输出示例

启动时若 `--pipeline_parallel > 1`，Stage 0 会打印最终布局：

```text
PipelineLayout: num_stages=4, total_layers=24, vpp=1
  stage 0: [0, 4) + embedding
  stage 1: [4, 12)
  stage 2: [12, 18)
  stage 3: [18, 24) + final_norm + lm_head
```

其中 `[start, end)` 表示本 Stage 持有 `start`（含）到 `end`（不含）的 Transformer 层区间；
每个 Stage 在 vPP 下可能打印多个区间。

## 错误排查

以下非法配置会在启动阶段直接 `LOG(FATAL)` 终止，并给出可定位的报错信息：

| 场景 | 触发条件 | 报错关键字 |
| --- | --- | --- |
| 空项或非数字 | `4,,6,6` / `4,8a,6,6` | `not a positive integer` |
| 非正层数 | `4,0,6,6` | `must be positive` |
| Stage 数量不符 | `--pipeline_parallel 4` 但列表只有 3 项 | `entries but pipeline_parallel is` |
| 层数总和错误 | 24 层模型但 `4,8,6,5` | `sums to` |
| 与 vPP 冲突 | 自定义布局 + `--virtual_pipeline_parallel 2` | `incompatible with virtual_pipeline_parallel` |

若报「模型构建与参数加载层归属不一致」，通常是因为某个调用点仍在使用旧的均匀划分：请确认
模型构建（`TransformerModel`）、`PipelineParallel` 包装、两个 checkpoint loader 都改为查询
`PipelineLayout` / `StageInfo`，且 `GetPipelineLayerPartition()` 已正确传入 `InitAllEnv`。

## 测试

单元测试位于 `tests/distributed/test_pipeline_layout.cc`，覆盖默认均匀划分、自定义 `4,8,6,6`、
Embedding/FinalNorm/LMHead 归属、`StageOfLayer`/`OwnsLayer`、vPP 交错、chunk↔stage 映射以及
各类非法配置的死亡断言。

```bash
cmake -S . -B build -DBUILD_TEST=ON
cmake --build build -j
ctest --test-dir build -R 'test_pipeline_layout' --output-on-failure
```

端到端验证需至少 2 个 Pipeline Stage：用相同初始权重分别以单卡/默认布局与自定义布局跑若干
训练迭代，比较前向结果、loss 与梯度在允许误差内一致（fp32 1e-05，bf16 1e-02），并确认训练
过程无通信死锁。

## API 摘要

```cpp
namespace infini_train::nn::parallel {

std::vector<int> ParsePipelineLayerPartition(const std::string &str);

class PipelineLayout {
public:
    static PipelineLayout Create(int total_layers, int num_stages, int vpp_size,
                                 const std::vector<int> &partition = {});
    StageInfo GetStageInfo(int stage_id) const;
    int StageOfLayer(int layer_id) const;
    bool OwnsLayer(int stage_id, int layer_id) const;
    static int StageOfChunk(int global_chunk_id, int num_stages);
    static int LocalChunkIndexOfChunk(int global_chunk_id, int num_stages);
    std::string Describe() const;
};

} // namespace infini_train::nn::parallel
```

- `StageInfo` 包含 `is_first_stage`、`is_last_stage` 与 `layer_ranges_per_chunk`
  （每个 chunk 一个 `(start, end)` 区间）。
- `Create` 是统一布局入口：空 `partition` 走默认均匀划分，否则按显式层数构建并做完整校验。
- `GetStageInfo` 供模型构建 / `PipelineParallel` / checkpoint loader 使用；
  `StageOfChunk` / `LocalChunkIndexOfChunk` 供调度器统一计算 chunk 归属。

# InfiniTrain FlashAttention 接入 - Scrum Story Cards (Iteration 2)

**总预估工时**: 2 人日
**迭代目标**: 修复 LLaMA-3 精度问题，并优化 FlashAttention Kernel 性能。

---

## Story 5: LLaMA-3 FlashAttention 精度修复 (已完成)
**点数**: 1.0 人日
**描述**:
在 Iteration 1 中发现 LLaMA-3 开启 FlashAttention 后 Loss 异常 (14.6 vs 3.5)。需要定位并修复该问题，确保 LLaMA-3 训练精度对齐。

**Acceptance Criteria (AC)**:
- **AC1 [最小复现]**: 编写 C++ 单元测试，模拟 LLaMA-3 的 Attention 输入 (RoPE 后, Causal Mask)，对比 FlashAttention 与标准 Attention 的数值输出，定位误差来源。 (已完成)
- **AC2 [布局修正]**: 检查并修复 LLaMA-3 模型中 Q/K/V 的内存布局（Transpose/Contiguous）与 FlashAttention Kernel 的预期输入是否一致。 (已完成)
- **AC3 [RoPE/Scale]**: 确认 RoPE 旋转后的数据与 Softmax Scale 因子是否正确传递。 (已完成)
- **AC4 [端到端验证]**: 运行 `benchmark.py` LLaMA-3 任务，Baseline 与 FlashAttention 的 Initial Loss 差异小于 `0.1` (即 Flash 也应在 ~3.5-4.0 之间)。 (已完成，Loss 3.51 vs 3.51)

---

## Story 6: FlashAttention Kernel 性能优化 (Tiling)
**点数**: 1.0 人日
**描述**:
当前的 FlashAttention Kernel 为 Naive 实现，性能较差 (0.1x Speedup)。需要引入 Tiling 分块优化与向量化访存，提升计算吞吐。

**Acceptance Criteria (AC)**:
- **AC1 [Tiling优化]**: 在 CUDA Kernel 中实现 Q/K/V 的分块加载 (SRAM 缓存)，减少 HBM 访问。
- **AC2 [向量化访存]**: 使用 `float4` 进行向量化加载与存储。
- **AC3 [性能提升]**: 在 GPT-2 (1024/2048) 基准测试中，FlashAttention 的 Tokens/s 至少达到 Baseline 的 80% 或更高（理想情况应 >1.5x）。

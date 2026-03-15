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

## Story 7: FlashAttention Kernel Tensor Core (WMMA) 优化 (进行中)
**点数**: 1.0 人日
**描述**:
当前 FlashAttention 性能受限于 CUDA Core (FP32) 的计算能力。为了利用 Ampere/Volta 架构的 Tensor Cores，需要引入 WMMA (Warp Matrix Multiply Accumulate) API。
考虑到现有模型为 FP32，本 Story 将实现 "Mixed Precision Kernel"：输入/输出仍为 FP32，但在 Kernel 内部将数据转换为 FP16 并使用 Tensor Cores 进行矩阵乘法。

**Acceptance Criteria (AC)**:
- **AC1 [WMMA Kernel]**: 实现基于 `nvcuda::wmma` 的 FlashAttention Forward Kernel。 (已完成)
- **AC2 [Mixed Precision]**: Kernel 能够从 FP32 HBM 加载数据，转换为 FP16 存入 SRAM，并在 Tensor Cores 上计算，最后输出 FP32。 (已完成)
- **AC3 [正确性验证]**: 通过 `test_flash_layout` 验证 WMMA Kernel 的数值正确性（允许 FP16 精度误差）。 (已完成 - 修复 Padding 初始化问题，测试通过)
- **AC4 [性能提升]**: 在 LLaMA-3 (1024) 任务上，FlashAttention 性能超越 Baseline (目标 >1.5x Speedup)。 (失败 - 性能回退，需后续优化)
    - **Status**: Forward Kernel 正确性已验证 (Loss 对齐)，但 TPS 仅为 Baseline 的 7% (863 vs 12280)。暂挂起性能优化，优先完成 Story 8 (Backward Pass)。

---

## Story 8: Backward Pass 优化 (消除 AtomicAdd) (待开始)
**点数**: 1.0 人日
**描述**:
当前 Backward Pass 使用全局 `atomicAdd` 更新 `dK` 和 `dV`，导致严重的内存竞争。需要重构 Backward Kernel，使用确定性的并行归约或 Thread Block 级别的累加策略。

**Tasking**:
1. **分析现有 Backward Kernel**: 确认 `atomicAdd` 的瓶颈位置。
2. **设计并行归约策略**:
    - 每个 Thread Block 计算一部分 `dQ, dK, dV`。
    - 对于 `dK, dV`，由于它们在 Seq 维度上共享，需要跨 Block 累加。
    - 方案 A: 使用 Grid 级别的同步 (Global Barrier) 或两阶段 Kernel (Compute -> Reduce)。
    - 方案 B: 限制 Grid 映射，使得同一 Head 的计算在同一 Block 或通过 Shared Memory 交换 (FlashAttention 原始论文策略)。
3. **实现新 Kernel**: 移除 Global AtomicAdd。
4. **验证正确性**: 使用 `test_flash_layout` 或 Gradient Check。

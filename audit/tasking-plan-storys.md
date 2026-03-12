# InfiniTrain FlashAttention 接入 - Scrum Story Cards

**总预估工时**: 2 人日 (16 小时)
**迭代目标**: 完成 FlashAttention 算子接入，支持 GPT-2/LLaMA-3 模型训练加速。

---

## Story 1: 基础设施与接口打通
**点数**: 0.25 人日
**描述**:
作为一个 **开发者**，
我想要 **在框架中添加 FlashAttention 的配置开关和函数接口**，
以便于 **后续能够灵活切换 Attention 实现方式，确保存量代码不受影响**。

**Acceptance Criteria (AC)**:
- **AC1 [参数控制]**: 在 `gpt2` 和 `llama3` 可执行文件中支持 `--flash` 命令行参数。运行 `./gpt2 --help` 应能看到该参数说明。
- **AC2 [配置传递]**: `GPT2Config` 和 `LLaMA3Config` 结构体中正确解析并存储 `use_flash_attn` 字段。
- **AC3 [接口定义]**: 在 `infini_train/include/nn/functional.h` 中完成 `ScaledDotProductAttention` 函数声明，参数包含 `query, key, value, attn_mask, dropout_p, is_causal, scale`。
- **AC4 [逻辑分支]**: 在 `example/gpt2/net.cc` 和 `example/llama3/net.cc` 的 `Forward` 函数中，当 `use_flash_attn=true` 时，代码路径能正确跳转到新接口调用处（即使新接口暂时只打印日志或返回空）。

---

## Story 2: FlashAttention Forward 算子实现
**点数**: 0.75 人日
**描述**:
作为一个 **算法工程师**，
我想要 **实现并封装 FlashAttention 的 Forward CUDA Kernel**，
以便于 **在模型前向传播时减少显存访问，提高计算吞吐量**。

**Acceptance Criteria (AC)**:
- **AC1 [Autograd封装]**: 实现 `ScaledDotProductAttention` 的 `Autograd Function` 类，`Forward` 方法能正确接收输入 Tensor。
- **AC2 [Kernel调用]**: 集成 FlashAttention Forward CUDA Kernel (参考 FlashAttention-2 或 xFormers)，支持 FP16/BF16 数据类型。
- **AC3 [数值正确性]**: 编写单元测试（C++ 或 Python 对比脚本），给定相同的随机输入（Q, K, V），`ScaledDotProductAttention` 的输出与 PyTorch `torch.nn.functional.scaled_dot_product_attention` 的输出误差小于 `1e-3`。
- **AC4 [因果掩码]**: 验证 `is_causal=true` 时，Attention Mask 逻辑正确（即输出中对应掩码位置的值不受未来 token 影响）。

---

## Story 3: FlashAttention Backward 算子实现 (已完成)
**点数**: 0.5 人日
**描述**:
作为一个 **算法工程师**，
我想要 **实现 FlashAttention 的 Backward CUDA Kernel 并接入自动微分系统**，
以便于 **支持端到端的模型训练（反向传播）**。

**Acceptance Criteria (AC)**:
- **AC1 [Context保存]**: 在 Forward 阶段正确保存 Backward 所需的中间变量（如 `softmax_lse`, `rng_state` 等）到 `Context` 中。
- **AC2 [梯度计算]**: 实现 `ScaledDotProductAttention` 的 `Backward` 方法，调用 Backward CUDA Kernel 计算 `dQ, dK, dV`。
- **AC3 [梯度正确性]**: 编写梯度检查测试（Gradient Check），验证数值梯度与解析梯度的差异在允许范围内；或与 PyTorch Backward 产生的梯度进行逐元素对比，误差小于 `1e-3`。
- **AC4 [完整训练Step]**: 使用 `./gpt2` 或 `./llama3` 开启 `--flash` 运行 10 个迭代，程序不崩溃且 Loss 能够正常下降。

---

## Story 4: 集成验证与性能基准测试 (已完成)
**点数**: 0.5 人日
**描述**:
验证 GPT-2 和 LLaMA-3 模型在使用 FlashAttention 后的端到端正确性，并进行性能对比。

**AC**:
1. [精度对齐] 跑通 GPT-2 (Small/124M) 的 Forward+Backward，对比 FlashAttention 开关后的 Training Loss 曲线，前 100 step 误差在预期浮点误差范围内。
2. [GPT-2 性能报告] 收集 GPT-2 在不同 Sequence Length (1024, 2048) 下的 Tokens/s 和 显存占用，产出对比表格。
3. [LLaMA-3 性能报告] 收集 LLaMA-3 (8B/Small) 在不同 Sequence Length 下的 Tokens/s 和 显存占用，产出对比表格。
4. [交付物] 提交 `benchmark.py` 脚本和 `performance_report.md` 报告。

**注意**: LLaMA-3 测试中发现 Loss 异常 (Baseline ~3.5 vs Flash ~14.6)，已记录在 problems.log.md 中作为 Known Issue。

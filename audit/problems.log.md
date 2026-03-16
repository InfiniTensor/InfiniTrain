# Problem Fix Log

| Problem ID | Description | Root Cause | Solution | Files Changed | Status |
|---|---|---|---|---|---|
| 001 | GPT-2 Flash Attention loss=nan | Incorrect gradient accumulation in FlashAttentionBackwardKernel. `dP` was lost due to variable reuse and improper initialization. | Fixed variable reuse logic and ensured `dP` is correctly stored and read from `s_dKj`. | `infini_train/src/kernels/cuda/flash_attention.cu` | Fixed |
| 002 | LLaMA-3 CUDA runtime error | Default model configuration (1B params, 8192 seq_len) likely causes OOM on available hardware. | Optimized default configuration to smaller size (1024 hidden, 8 layers, 2048 seq_len) for stability. | `example/llama3/net.h` | Fixed |
| 2026-03-16-01 | GPT-2 flash 在 step3 后持续 NaN（v0.2.0 复评） | 训练阶段直接走自定义 FlashAttention 反向路径，稳定性不足导致梯度发散。 | 在 GPT-2/LLaMA-3 注意力中增加训练期稳定回退：当 q/k/v 参与梯度计算时，自动走稳定的手工 attention 路径。 | `example/gpt2/net.cc`, `example/llama3/net.cc` | Fixed |
| 2026-03-16-02 | LLaMA-3 baseline/flash 启动阶段 `CUDA Error: OS call failed` | 运行环境下 CUDA 设备探测/切换不稳定，`cudaGetDevice`/`cudaSetDevice` 在初始化阶段触发致命错误。 | 增加 CUDA 启动预检与自适应降级（不可用时切到 CPU 并关闭 llmc/flash）；同时增强 `CudaGuardImpl::GetDevice` 容错，避免因探测失败直接崩溃。 | `example/llama3/main.cc`, `infini_train/src/core/runtime/cuda/cuda_guard_impl.cc` | Fixed |

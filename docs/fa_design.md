# Flash Attention 后端设计

## 目标

InfiniTrain 提供与 Megatron 风格一致的 attention backend 参数：

```bash
--attention_backend unfused
--attention_backend flash
```

`unfused` 保持原有的 naive causal attention；`flash` 使用 FlashAttention 2 CUDA kernel
融合 attention score、causal mask、softmax 和 value 聚合，避免生成完整的 `T x T`
attention 中间张量。

本实现采用 native CUDA adapter，不经过 PyTorch extension API：

```text
InfiniTrain Tensor
  -> data_ptr / shape / stride / allocator / CUDA stream
  -> Flash_fwd_params / Flash_bwd_params
  -> run_mha_fwd_ / run_mha_bwd_
  -> FlashAttention CUDA kernel
```

## 源码与依赖

FlashAttention 以 submodule 固定在：

- 路径：`third_party/flash-attention`
- 版本：`v2.7.4.post1`
- commit：`5231d95fe13733fb534c01895f7ea88c6a6c7793`
- CUTLASS 路径：`third_party/flash-attention/csrc/cutlass`
- CUTLASS commit：`c506e16788cb08416a4a57e11a9067beeee29420`

初始化命令：

```bash
git submodule update --init --recursive third_party/flash-attention
```

FlashAttention 接入自身只依赖：

- CUDA toolkit；
- FlashAttention submodule 中的 CUDA 源码；
- FlashAttention 内嵌的 CUTLASS submodule；
- InfiniTrain 自己的 Tensor、allocator、DeviceGuard 和 CUDA stream。

不需要安装 `torch`、`flash-attn` wheel、Python development headers 或其他 attention
运行库。CMake 不执行 `find_package(Torch)`，不编译 `flash_api.cpp`，最终二进制也不链接
`libtorch`、`libc10`、`libtorch_python` 或 `libpython`。InfiniTrain 原有的 CUDA、cuBLAS、
NCCL、gflags、glog、Eigen 依赖不属于本次 attention 接入新增依赖。

上游 kernel 头文件即使关闭 dropout，参数 ABI 仍保留 `at::PhiloxCudaState`，launch
template 仍引用两个 c10 CUDA 检查宏。本实现没有修改 submodule，而是在
`flash_attention_compat` 中提供最小兼容定义：

- Philox state 仅保留两个 64-bit 字段；dropout=0 时不会消费随机状态；
- CUDA 检查宏直接使用 `cudaGetLastError`、`cudaGetErrorString` 和 fail-fast；
- 兼容头不包含或链接任何 ATen/c10 实现。

## 构建接入

`USE_FLASH_ATTENTION=ON` 时创建静态库 `flash_attn_native`。当前只编译 sm80 下训练路径
需要的 8 个实例：

```text
dtype:      fp16, bf16
head_dim:   64, 128
mask:       causal
direction:  forward, backward
```

编译时定义：

```text
FLASHATTENTION_DISABLE_DROPOUT
FLASHATTENTION_DISABLE_ALIBI
FLASHATTENTION_DISABLE_SOFTCAP
FLASHATTENTION_DISABLE_LOCAL
```

这样能显著减少模板实例数量和首次编译时间。后续按实际模型需求扩展，而不是默认编译
上游 Python extension 的全部组合。

如果构建时设置 `USE_FLASH_ATTENTION=OFF`，CMake 会排除 native adapter。此时运行期传入
`--attention_backend=flash` 会在 framework autograd operator 首次执行时明确报错，并提示使用
`-DUSE_FLASH_ATTENTION=ON` 重新构建，不会只得到 dispatcher 的通用 kernel 缺失错误。

TODO：

- 将 CUDA architecture 改为显式 CMake cache 参数并验证 sm89/sm90；
- 按需增加 head_dim 32/96/160/192/256；
- 增加 dropout、local attention、ALiBi、softcap；
- 增加 varlen、KV cache 和 split-KV；
- 增加 deterministic backward。

## Native adapter

上层 autograd 类和 functional API 命名为 `ScaledDotProductAttention`，对应文件为
`scaled_dot_product_attention.h/.cc`。接口根据 Q 与 K/V 的 head 数自动选择 MHA 或原生
GQA/MQA：head 数相等时执行 MHA，Q heads 是 KV heads 的整数倍时执行 GQA/MQA，其他情况
明确报错。当前 API 暂未暴露 mask 参数，backend 固定执行 causal scaled dot-product attention。

入口文件为 `infini_train/src/kernels/cuda/flash_attention.cu`，不 include ATen/c10，
也不创建 `at::Tensor`。

InfiniTrain attention 进入 backend 前的 q/k/v 物理布局是连续 `(B,H,T,D)`。
FlashAttention kernel 按逻辑 `(B,T,H,D)` 访问，因此直接设置元素 stride：

```text
batch_stride = H * T * D
row_stride   = D
head_stride  = T * D
```

无需转置、复制或 Tensor wrapper。输出使用相同 stride，因此物理布局仍为
`(B,H,T,D)`，不改变后续 projection 的接口。

Forward：

1. 检查 CUDA device、Q/K/V shape、compute dtype 和 head_dim。
2. 必要时用 InfiniTrain `Tensor::To` 转成 bf16/fp16。
3. 用 InfiniTrain allocator 创建 output 和 FP32 `softmax_lse`。
4. 填充 `Flash_fwd_params` 并在 InfiniTrain 当前 CUDA stream 调度 causal kernel。
5. 在 autograd context 保存 q/k/v、output 和 `softmax_lse`。

Backward：

1. 创建 dq/dk/dv、FP32 `dsoftmax_sum` 和 `dq_accum` workspace。
2. 填充 `Flash_bwd_params`，调用 non-deterministic causal backward kernel。
3. 原生 GQA/MQA 模式先生成 Q-head 数量的临时 dK/dV，再按 group 求和到原始 KV heads。
4. 必要时将梯度转换回输入原始 dtype。

所有中间内存都由 InfiniTrain `Tensor` 管理，生命周期由 autograd context 或当前调用栈
持有；kernel 直接使用 `CudaStream::cuda_stream()`，不再做 PyTorch stream guard 转换。

## 当前范围与并行策略

当前 flash backend 支持：

- CUDA sm80；
- bf16/fp16 kernel；CLI 训练路径目前开放 bf16；
- causal self-attention；
- head_dim 64/128；
- 默认使用原生 GQA/MQA；
- dropout=0；
- 固定长度 batch；
- non-deterministic backward。

暂不支持 KV cache、外部 mask、generation `start_pos > 0`。Flash backend 默认将原始
KV heads 直接交给 FlashAttention，并根据 Q/KV head 比例自动使用 MHA 或原生 GQA/MQA；
unfused backend 保持先执行 `RepeatKV` 的实现。

backend 选择发生在已有并行布局处理之后：

- Tensor Parallel：每个 rank 只计算 local heads，不改变 TP 通信；
- Sequence Parallel：已有路径先恢复 attention 需要的 sequence layout；
- RoPE：保持在 attention kernel 前执行；
- GQA：unfused 使用 `RepeatKV`，FlashAttention 原生处理 Q/KV head 比例。

扩展能力应继续放在 adapter 参数构造和 CMake kernel 实例列表中，不侵入现有并行策略。

## 构建与验证

验证环境：

- container：`bolunz_infinitrain`
- worktree：`/workspace/Github/InfiniTrain-flash_attn`
- GPU：NVIDIA A800-SXM4-80GB，sm80
- CUDA：13.0

构建命令：

```bash
docker exec bolunz_infinitrain cmake \
  -S /workspace/Github/InfiniTrain-flash_attn \
  -B /workspace/Github/InfiniTrain-flash_attn/build_flash_native_sm80 \
  -DUSE_CUDA=ON -DBUILD_TEST=OFF -DUSE_NCCL=ON \
  -DUSE_FLASH_ATTENTION=ON -DCMAKE_BUILD_TYPE=Release

docker exec bolunz_infinitrain cmake --build \
  /workspace/Github/InfiniTrain-flash_attn/build_flash_native_sm80 \
  --target gpt2 llama3 -j 16
```

`gpt2` 和 `llama3` 均编译通过。对 `gpt2` 执行 `ldd` 和未解析符号检查，未发现
Torch、ATen、c10 或 Python 依赖。

### Loss 对齐

GPT-2 124M，bf16，`B=4,T=128`，3 steps：

| Backend | Losses |
| --- | --- |
| unfused | 9.763003, 9.736726, 9.564169 |
| flash native | 9.739741, 9.735670, 9.571149 |

flash native 三步结果与移除 Torch wrapper 前的 FlashAttention 结果逐项一致。与 unfused
存在小幅 bf16 数值差异，来源是 fused kernel 的计算与归约顺序不同。

### 性能

GPT-2 124M，bf16，`B=4,T=1024`，5 steps，统计 step 3-5：

| Backend | 平均耗时 | 平均吞吐 | Peak Used |
| --- | ---: | ---: | ---: |
| unfused | 124.06 ms | 33,015 tok/s | 7,111 MB |
| flash native | 84.27 ms | 48,626 tok/s | 7,146 MB |

按平均 step time 比值计算，flash native 比 unfused 快约 `47.2%`；耗时降低约 `32.1%`。

### 并行 smoke

GPT-2 124M，bf16，`B=1,T=64`，1 step：

| Config | Backend | Loss |
| --- | --- | ---: |
| TP=2 | flash native | 9.982512 |
| TP=2, SP=2 | flash native | 9.982512 |

两种配置均完成 forward、backward 和 optimizer step，且 loss 一致。

### GQA 对齐

LLaMA 配置使用 `Hq=32,Hkv=8,D=64`，bf16，`B=1,T=64`。原生 GQA 和手工
`RepeatKV` 两条路径均完成 forward、backward 和 optimizer step，首步 loss 都是
`11.761781`：

当前 Flash backend 默认执行原生 GQA；手工 `RepeatKV` 路径用于 unfused backend。

复现命令只需替换最后一个 backend 参数：

```bash
docker exec bolunz_infinitrain \
  /workspace/Github/InfiniTrain-flash_attn/build_flash_native_sm80/gpt2 \
  --input_bin=/tmp/infinitrain_gpt2_tiny_train.bin \
  --llmc_filepath=/workspace/Github/InfiniTrain/torch_compare/gpt2_124M.bin \
  --batch_size=4 --sequence_length=1024 --total_batch_size=4096 \
  --num_iteration=5 --freq_generate_txt=1000 \
  --dtype=bfloat16 --device=cuda --attention_backend=flash
```

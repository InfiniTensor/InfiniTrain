# FlashAttention 2 后端设计

## 目标与边界

InfiniTrain 通过 attention backend 选项在原有 unfused causal attention 和
FlashAttention 2 之间切换：

```bash
--attention_backend=unfused
--attention_backend=flash
```

`unfused` 使用框架算子显式计算 attention score、causal mask、softmax 和 value
聚合；`flash` 调用 FlashAttention 2 的 fused CUDA kernel，不生成完整的 `T x T`
attention 中间张量。

本接入面向训练期、固定长度、causal self-attention。它不是 PyTorch extension，也不以
兼容 `torch.nn.functional.scaled_dot_product_attention` 的完整参数语义为目标。当前
functional API 只表达 Q/K/V 和 scale，mask、dropout、local window、KV cache 等能力尚未
进入接口。

## 分层与调用路径

当前调用链为：

```text
GPT-2 / LLaMA3 CLI --attention_backend
  -> TransformerConfig::flash
  -> CausalSelfAttention::Forward
       -> QKV projection / optional RoPE / MHA or GQA layout handling
       -> nn::function::ScaledDotProductAttention
  -> autograd::ScaledDotProductAttention
       -> Dispatcher availability check
       -> ScaledDotProductAttentionForward / ScaledDotProductAttentionBackward
  -> Flash_fwd_params / Flash_bwd_params
  -> run_mha_fwd_ / run_mha_bwd_
  -> FlashAttention 2 CUDA kernel
```

各层职责如下：

- example main：解析 backend，检查 CLI device/dtype，并把选择写入模型配置；
- transformer module：处理 projection、位置编码、local heads、GQA 和输出 projection；
- functional/autograd：定义可求导接口、选择 compute dtype，并提供统一 availability guard；
- CUDA adapter：校验 kernel contract、管理 workspace/context、填充上游参数并调度 kernel；
- CMake：只实例化当前模型需要的上游 CUDA 模板组合。

## 源码与依赖

FlashAttention 作为 submodule 固定在：

- `third_party/flash-attention`
- tag：`v2.7.4.post1`
- commit：`5231d95fe13733fb534c01895f7ea88c6a6c7793`
- CUTLASS：`third_party/flash-attention/csrc/cutlass`
- CUTLASS commit：`c506e16788cb08416a4a57e11a9067beeee29420`

初始化命令：

```bash
git submodule update --init --recursive third_party/flash-attention
```

本接入不编译上游 `flash_api.cpp`，不创建 `at::Tensor`，也不链接 Torch、ATen、c10、
Python 或 `flash-attn` wheel。运行时只依赖 CUDA、InfiniTrain Tensor/allocator、
DeviceGuard、当前 CUDA stream，以及 submodule 中的 FlashAttention/CUTLASS 源码。

### ATen/c10 兼容头

上游 kernel/launch 头仍残留少量 PyTorch 类型和宏：

- `flash.h` 的参数 ABI 包含 `at::PhiloxCudaState`；
- `flash_fwd_kernel.h` 调用 `at::cuda::philox::unpack()`；
- forward/backward launch template 使用 `C10_CUDA_CHECK` 和
  `C10_CUDA_KERNEL_LAUNCH_CHECK`。

`infini_train/src/kernels/cuda/flash_attention_compat` 提供同路径的最小替代头，让这些
引用在不安装 Torch 的情况下编译：

- Philox state 只保留当前上游头所需的两个 64-bit 字段；
- `unpack()` 只返回 seed/offset；
- CUDA 检查宏基于 CUDA runtime error API 执行 fail-fast。

当前构建禁用 dropout，不会实际消费随机状态。adapter 仍创建并清零一个两元素
`uint64` rng-state buffer，以满足上游参数 ABI。

该兼容层与固定版本的上游内部 header/ABI 耦合，不是稳定接口。升级 FlashAttention 或
未来引入真实 ATen headers 时，必须重新检查类型布局、include 顺序和宏定义冲突，不能把
“无需链接 Torch”等同于“完全没有 ATen/c10 header 形状依赖”。

## 构建接入

### CMake 开关

当前 `USE_FLASH_ATTENTION` 默认值为 `ON`，但只有同时启用 `USE_CUDA` 才会创建 CUDA
backend：

```bash
cmake -S . -B build \
  -DUSE_CUDA=ON \
  -DUSE_NCCL=ON \
  -DUSE_FLASH_ATTENTION=ON \
  -DBUILD_TEST=ON
cmake --build build -j
```

`FLASH_ATTN_SOURCE_DIR` 是一个 CMake cache path，默认指向仓库内 submodule。配置阶段会
检查 `flash.h` 和 CUTLASS 主头是否存在，缺失时直接报错。

开启 backend 时：

1. 创建静态库 `flash_attn_native`；
2. 为 `infini_train_cuda_kernels` 编译并链接 native adapter；
3. 注册 `ScaledDotProductAttentionForward/Backward` CUDA dispatcher kernels。

关闭 backend 时，CMake 从 CUDA source 列表排除 `flash_attention.cu`，不会创建或链接
`flash_attn_native`。functional/autograd 接口仍可编译，但第一次执行 FlashAttention
operator 时会通过 `Dispatcher::HasKernel` 明确提示使用
`-DUSE_FLASH_ATTENTION=ON` 重新构建。

### AOT kernel 实例

`flash_attn_native` 当前固定 `CUDA_ARCHITECTURES=80`，只编译 8 个训练实例：

```text
dtype:      fp16, bf16
head_dim:   64, 128
mask:       causal
direction:  forward, backward
```

同时定义：

```text
FLASHATTENTION_DISABLE_DROPOUT
FLASHATTENTION_DISABLE_ALIBI
FLASHATTENTION_DISABLE_SOFTCAP
FLASHATTENTION_DISABLE_LOCAL
```

这限制了模板数量和编译范围。增加 dtype、head dimension、GPU architecture 或 attention
模式时，必须同步修改：

- `FLASH_ATTN_CUDA_SOURCES` 中的 AOT 实例；
- adapter 的 runtime 校验；
- `RunForward` / `RunBackward` 的 dispatch 分支；
- 相应自动化测试。

InfiniTrain 其他 CUDA kernels 的 architecture 列表更宽，不代表 FlashAttention native
kernel 已支持这些架构；当前 Flash backend 的有效架构边界仍是 sm80。

## API 与 availability

framework 入口为：

```cpp
nn::function::ScaledDotProductAttention(q, k, v, scale)
```

输入 contract：

- Q 使用 `(B, Hq, T, D)`；
- K/V 使用 `(B, Hkv, T, D)`，且 shape 相同；
- Q/K/V 位于同一 CUDA device；
- `Hq % Hkv == 0`；
- `D` 为 64 或 128；
- Q 和 K/V 当前使用相同 batch、sequence length 和 head dimension。

API 没有 `enable_gqa`。`Hq == Hkv` 自动执行 MHA；`Hq > Hkv` 且可整除时自动执行
GQA/MQA；不能整除时在 autograd/operator 边界报错。

availability 不通过公共宏或 `nn` namespace 下的全局布尔值暴露。autograd operator 根据
Q 的 device 构造 dispatcher key，并在 forward 调用前执行 `HasKernel`。这把构建能力
检查放在 framework/operator 边界：example main 不需要包含构建宏，其他 framework
调用者也能得到相同错误。

## dtype 与 autocast

native kernels 支持 fp16 和 bf16。compute dtype 的选择顺序为：

1. Q 已是 fp16/bf16 时直接使用 Q dtype；
2. 否则要求当前 autocast context 启用，且 autocast dtype 为 fp16/bf16；
3. 其他情况 fail-fast。

adapter 使用 `Tensor::To` 把 Q/K/V 和 backward 的 `grad_output` 转成 compute dtype，输出
保持 compute dtype。GPT-2/LLaMA3 当前 CLI 只开放 `--dtype=bfloat16` 的 Flash 路径；
fp16 是 framework/kernel 能力，但不是这两个 example 的已开放训练选项。

BF16 backward 末尾当前会把 dQ/dK/dV 显式提升为 FP32。这是框架 autocast/autograd dtype
语义尚未集中处理前的临时兼容逻辑：forward 的 raw `Tensor::To` 没有建立通用的
cast-backward edge，混合 dtype 梯度也没有在 autograd/accumulation 层统一归一化。未来若
框架层补齐该语义，应删除 adapter 内的特殊 upcast，让 kernel 返回其自然梯度 dtype。

## Transformer 接入

Flash 和 unfused backend 共用 `CausalSelfAttention::Forward`，不再区分
`ForwardStandard` 与 `ForwardWithRoPE`。位置编码由 `PositionEmbeddingType` 决定：

- `kLearnedAbsolute`：模型前段添加 WPE；attention module 创建内部 causal-mask buffer，
  供 unfused fallback 使用；
- `kRoPE`：`ApplyRotaryEmbedding` 在 attention backend 之前处理 Q/K；当前 transformer
  调用者提供 runtime mask。

统一路径中的 QKV 处理为：

1. ColumnParallelLinear 产生 packed QKV；
2. MHA 的 Q/K/V 宽度相等，使用单个 `Split` autograd node；
3. GQA 的 Q 和 K/V 宽度不同，使用三个 `Slice`；
4. RoPE 模型在此后旋转 Q/K；
5. unfused backend 对 K/V 执行 `RepeatKV`；
6. Flash backend 保留原始 KV heads，由 native kernel 根据 `Hq/Hkv` 处理 GQA；
7. Q/K/V 转换到 `(B, H, T, D)` 后进入相应 backend。

`Split` fast path 不改变统一 Forward 的语义。它避免在普通 MHA 中创建三个独立 Slice
autograd nodes；GQA 因为分段宽度不同，仍必须使用 Slice 路径。

### mask 与 start_pos

当前 Flash functional API 没有 mask/start_pos 参数。Transformer Forward 虽然已经解析
这两个输入，但选择 Flash backend 后会忽略它们，并固定设置：

```text
is_causal        = true
window_size_left = -1
window_size_right = 0
seqlen_q         = seqlen_k
```

因此当前 Flash 语义仅适用于从位置 0 开始的标准 causal self-attention。外部自定义 mask、
padding mask、非零 start position、incremental decoding 和 cross-attention 均不受支持。
在这些能力进入 functional API 之前，调用者不能假设传入 mask/start_pos 会影响 Flash
结果。

## Native CUDA adapter

入口文件为 `infini_train/src/kernels/cuda/flash_attention.cu`。它直接包含上游 `flash.h`，
其中的 ATen/c10 引用由前述兼容 include path 满足；adapter 本身不构造任何 PyTorch
对象。

### 布局与 stride

InfiniTrain 进入 adapter 的 Q/K/V 物理布局为连续 `(B, H, T, D)`。FlashAttention kernel
按逻辑 `(B, T, H, D)` 解释，因此 adapter 设置元素 stride：

```text
batch_stride = H * T * D
row_stride   = D
head_stride  = T * D
```

Q 与 K/V 分别使用各自的 head count 计算 batch stride。输出沿用 Q 的 shape/stride，物理
布局仍为 `(B, Hq, T, D)`，无需额外 Tensor wrapper 或布局复制。

### Forward

Forward 执行：

1. 校验 device、shape、head ratio、dtype 和 head dimension；
2. 将 Q/K/V 转为选定 compute dtype；
3. 分配 output、FP32 `softmax_lse` 和零初始化 rng-state；
4. 填充 `Flash_fwd_params`，固定 causal/dropout=0/num_splits=1；
5. 在 InfiniTrain 当前 CUDA stream 上调度 AOT kernel；
6. 把 kernel backward 所需状态保存到 opaque `FlashAttentionContext`。

context 保存转换后的 Q/K/V、detach 后的 output、`softmax_lse`、rng-state、原始 dtype 和
compute dtype。保存 detach output 是为了避免形成
`Function -> flash_ctx -> output -> Function` 引用环。

### Backward

Backward 执行：

1. 将 `grad_output` 转为 compute dtype；
2. 分配 dQ/dK/dV；
3. 分配 FP32 `dsoftmax_sum` 和 `dq_accum` workspace，其中 sequence length 向上取整到 128；
4. 设置 `deterministic=false` 并调度 causal backward；
5. 根据当前 autocast 兼容策略恢复返回梯度 dtype。

MHA 时 kernel 直接写入最终 dK/dV。GQA/MQA 时，上游 backward launch 需要 Q-head shape 的
临时 dK/dV；adapter 随后用自定义 CUDA reduction 按 group 求和回原始 KV-head shape。
reduction 在寄存器中使用 FP32 累加，再写回 fp16/bf16 tensor。该实现会额外占用两个
Q-head shape 临时 buffer，MQA 或长序列下需要关注其峰值内存。

### Stream、device 与生命周期

forward/backward 都使用 `DeviceGuard` 绑定 Q 所在 device，并通过 InfiniTrain
`CudaStream::cuda_stream()` 获取当前 stream。adapter 不创建额外 stream，也不进行
PyTorch stream guard 转换。

所有 tensor 和 workspace 都由 InfiniTrain allocator 管理。forward 后仍需存活的数据由
opaque context 持有，其余 backward 临时量由当前调用栈持有。kernel launch 保持同一
stream 顺序，不依赖额外 host synchronization。

## 当前支持矩阵

| 维度 | 当前状态 |
| --- | --- |
| Device | CUDA |
| GPU architecture | sm80 AOT kernels |
| Kernel dtype | fp16、bf16 |
| GPT-2/LLaMA3 CLI dtype | bf16 |
| Attention type | fixed-length causal self-attention |
| Head dimension | 64、128 |
| Head mapping | MHA、GQA、MQA，要求 `Hq % Hkv == 0` |
| Sequence relation | Q/K/V batch 和 sequence length 相同 |
| Dropout | 0，编译期禁用 |
| Backward | non-deterministic |
| Mask | 仅 kernel 内建 causal mask |
| Position | `start_pos=0` 语义 |
| Unsupported | varlen、padding/custom mask、cross-attention、KV cache、generation、local attention、ALiBi、softcap、split-KV |

## 自动化测试与开发入口

`tests/autograd/test_autograd_scaled_dot_product_attention.cc` 构建为独立 CUDA target
`test_flash_attention_cuda`，仅在 `USE_FLASH_ATTENTION=ON` 时注册。测试覆盖：

- native GQA 与显式展开 KV 的 forward/backward contract；
- packed QKV 经 Slice autograd 回传的梯度；
- native GQA 与 unfused causal reference；
- BF16 backward 的当前 FP32 gradient contract；
- fused BF16 路径相对 BF16/FP32 reference 的误差边界。

运行聚焦测试：

```bash
ctest --test-dir build --output-on-failure -R '^test_flash_attention_cuda$'
ctest --test-dir build --output-on-failure -R '^test_transformer_cuda$'
```

`scripts/test_config.json` 还提供 `flash` tag 的 GPT-2/LLaMA3 端到端训练 cases，用于验证
不同 batch/sequence shape 能完整执行 forward、backward 和 optimizer step。具体运行
结果保存在独立测试日志或报告中。

## 已知技术债与扩展顺序

建议按以下依赖关系扩展：

1. 在 functional API 中明确 mask/start_pos contract，并对不支持的输入 fail-fast；
2. 把 autocast cast-backward 和 mixed-dtype gradient 语义下沉到通用 autograd 基础设施，
   删除 adapter 的 BF16 特殊 upcast；
3. 为 deterministic backward 增加接口、workspace 和 AOT 实例；
4. 优化 GQA backward，避免长期保留 Q-head shape 的 dK/dV 临时 buffer；
5. 按模型需求扩展 head dimension 和 GPU architecture，并保持 CMake/runtime dispatch 同步；
6. 若支持 dropout，引入框架 generator/Philox contract，而不是继续使用零 rng-state；
7. 评估 varlen、padding mask、KV cache、local attention、ALiBi 和 softcap 所需的新参数与
   kernel 实例；
8. 升级 FlashAttention 时优先审计 compat headers，并重新确认最终链接不引入 Torch。

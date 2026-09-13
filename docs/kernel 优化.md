# InfiniTrain Kernel优化
## 1 实验环境

| 项 | 配置 |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB（虚拟机 passthrough，Ampere / sm_80） |
| CUDA / 驱动 | 12.4 / 550.107.02 |
| 工具链 / 构建 | gcc 13.4、cmake 3.30.9；`-DUSE_CUDA=ON -DUSE_NCCL=ON -DNVTX_MODE=ON`，`CMAKE_CUDA_ARCHITECTURES=75;80;90` |
| 模型 | LLaMA3.2-1B（16 层，n_embd 2048，n_head 32，GQA n_kv_head 8，vocab 128256） |
| 精度 | BF16 autocast：Linear/Matmul 走 Tensor Core，master 权重与 Adam 保持 FP32 |
| 训练配置 | batch_size 4 × seq_len 64，total_batch_size 256，5 iter |
| Profiling | Nsight Systems（nsys）+ NVTX，逐 step / 逐阶段标注 |

采集与汇总：

```bash
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=none --cpuctxsw=none \
  --output=nsys/out/llama3_5iter_nvtx --export=sqlite \
  ./build/llama3 --device cuda --dtype bfloat16 \
    --input_bin data/llama3/tiny_shakespeare_train.bin \
    --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
    --num_iteration 5 --batch_size 4 --sequence_length 64 --total_batch_size 256
nsys stats --report cuda_gpu_kern_sum --report nvtx_sum --format table \
  --output nsys/out/llama3_5iter_stats nsys/out/llama3_5iter_nvtx.nsys-rep
```

## 2 性能现状

### 2.1 nsys分析结果

稳态每步 ~113 ms，单 A100 吞吐 **~2260 tok/s**（FP32 ~1424，**1.6×**）。

| step | wall (ms) | Σ kernel (ms) | kernels | GPU util |
|---|---|---|---|---|
| Step_0 | 273.0 | 54.9 | 3483 | 21.1%（warm-up，host-bound） |
| Step_1 | 118.3 | 83.4 | 3569 | 73.1% |
| Step_2 | 114.8 | 76.7 | 3574 | 69.3% |
| Step_3 | 113.0 | 73.6 | 3574 | 67.5% |
| Step_4 | 112.4 | 73.6 | 3574 | 68.1% |

Step_1 Top kernel（Σ = 83.4 ms）：

| # | kernel | n | Σ (ms) | 占比 |
|---|---|---|---|---|
| 1 | `AdamAccumulateGradKernel<float>` | 110 | 31.7 | 38.1% |
| 2 | `CastKernel<bf16,float>`（f32→bf16） | 356 | 9.5 | 11.4% |
| 3 | `ampere_s16816gemm_bf16_128x128_nt` | 49 | 4.6 | 5.5% |
| 4 | `ampere_bf16_s16816gemm_128x256_f2f_tn` | 49 | 4.5 | 5.4% |
| 5 | `BinaryBackwardKernel<float>`(Mul) | 194 | 3.7 | 4.4% |
| – | **全部 GEMM（BF16 Tensor Core）** | 339 | **16.8** | **20.0%** |
| – | **全部 Cast（autocast）** | 661 | **10.7** | **12.8%** |

GPU利用率为73%;

**关键结论**：
1）BF16训练引入了大量的Cast kernel完成bf16和FP32的转换。
2）Adam优化器是所有Kernel中耗时最多的。
3）GPU util 由 FP32 的 ~93% 降至 ~68–73%，整轮转为 **launch/cast 受限**（`cudaLaunchKernel` ~3526 次/步，host 侧 ~27.9 ms）。
4）整个训练过程只使用了一个stream，GPU的kernel和memory操作没有并行。

### 2.2 Timeline分析

按 host 发射归属，每轮step分三个阶段：

| 阶段 | kernel 数 | Σ GPU 时间 |
|---|---|---|
| Forward | 1343 | 23.4 ms（含 339 forward Cast） |
| Backward | 2116 | 28.2 ms（含 290 Cast） |
| Optimizer | 115 | 32.3 ms（FP32 Adam，第一大头） |
| **合计** | **3574** | **83.9 ms** |

按 kernel 家族聚合：

| 家族 | kernel 数 | Σ (ms) | 占比 |
|---|---|---|---|
| Adam（optimizer，fp32） | 115 | 32.3 | 38.5% |
| 其它 elementwise / 结构 | 1749 | 20.8 | 24.8% |
| GEMM（BF16 Tensor Core） | 339 | 16.8 | 20.0% |
| Cast（autocast 新增） | 661 | 10.7 | 12.8% |
| Fill | 710 | 3.3 | 3.9% |

#### 2.2.1 Forward Timeline
Forward 共 **1343 kernels / 23.402 ms**（含 339 个 autocast Cast）。

##### 2.2.1.1 迭代入口（整轮各 1 次）

```
[ 0] +0.693ms   8.80us  EmbeddingFwd·f32   token embedding: (4,64) → (4,64,2048)
[ 1] +0.816ms   5.12us  SliceFwd·f32       freqs_cis 取本轮窗口的 cos/sin
[ 2] +0.900ms   2.82us  TriuFwd·f32        causal mask = Triu(ones(64,64),1)
```

之后进入 **16 层结构相同的 TransformerLayer**（单层 = RMSNorm(ln1) → Attention → 残差 → RMSNorm(ln2) → MLP → 残差），最后接 ln_f → lm_head → CrossEntropy。

##### 2.2.1.2 实测：Layer 0 的 Attention 半区（kernels [3..66]，含 cast）

`+ms` 为相对 Step_1 起点的 host 发射时刻，`us` 为 GPU 执行时间：

```
── RMSNorm ln1（6 kernels，全 fp32）────────────────────────────────
  [ 3] +0.959ms    7.94us  UnaryFwd[Pow]·f32        x²
  [ 4] +0.992ms    6.53us  Mean(Reduce)·f32         mean(x², -1)
  [ 5] +1.024ms    2.46us  UnaryFwd[AddScalar]·f32  + eps
  [ 6] +1.056ms    2.50us  UnaryFwd[Rsqrt]·f32      1/√(·)
  [ 7] +1.089ms   21.82us  BinaryFwd[Mul]·f32       x * rsqrt
  [ 8] +1.132ms   21.47us  BinaryFwd[Mul]·f32       norm * weight

── QKV 投影（2 Cast + 1 bf16 GEMM）─────────────────────────────────
  [ 9] +1.183ms    4.74us  Cast[f32->bf16]          ln1 输出 → bf16（Linear 输入）
  [10] +1.202ms   41.47us  Cast[f32->bf16]          QKV master 权重 → bf16 ★
  [11] +1.259ms   28.32us  ampere_bf16_...f2f_..._tn Linear 2048→3072 (=(H+2·KV)·D)

── 拆分 q/k/v + 取 RoPE cos/sin/even/odd（7 Slice）─────────────────
  [12] +1.325ms   16.38us  SliceFwd·bf16            q = qkv[..., :2048]
  [13] +1.390ms    6.66us  SliceFwd·bf16            k = qkv[..., 2048:2560]
  [14] +1.450ms    6.69us  SliceFwd·bf16            v = qkv[..., 2560:3072]
  [15] +1.531ms    6.27us  SliceFwd·f32             cos = freqs_cis[...,0]
  [16] +1.603ms    5.54us  SliceFwd·f32             sin = freqs_cis[...,1]
  [17] +1.670ms   11.04us  SliceFwd·bf16            q_even = q[..., 0::2]
  [18] +1.734ms   10.82us  SliceFwd·bf16            q_odd  = q[..., 1::2]

── RoPE 作用于 q（fp32 算术：每个 Mul 前先 Cast[bf16->f32] 上转）────
  [19] +1.772ms    3.49us  Cast[bf16->f32]          q_even ↑f32
  [20] +1.792ms   15.81us  BinaryFwd[Mul]·f32       q_even * cos
  [21] +1.828ms    3.39us  Cast[bf16->f32]          q_odd ↑f32
  [22] +1.845ms   15.84us  BinaryFwd[Mul]·f32       q_odd * sin
  [23] +1.878ms    4.22us  BinaryFwdNB[Sub]·f32     left = q_even·cos − q_odd·sin
  [24] +1.915ms    3.36us  Cast[bf16->f32]          q_even ↑f32
  [25] +1.932ms   15.78us  BinaryFwd[Mul]·f32       q_even * sin
  [26] +1.964ms    3.39us  Cast[bf16->f32]          q_odd ↑f32
  [27] +1.982ms   15.81us  BinaryFwd[Mul]·f32       q_odd * cos
  [28] +2.018ms    4.22us  BinaryFwdNB[Add]·f32     right = q_even·sin + q_odd·cos
  [29] +2.070ms    8.29us  StackFwd·f32             stack(left,right) → flatten

── RoPE 作用于 k（2 Slice + 4×[Cast+Mul] + Sub + Add + Stack）──────
  [30] +2.148ms    6.02us  SliceFwd·bf16            k_even
  [31] +2.195ms    5.95us  SliceFwd·bf16            k_odd
  [32] +2.226ms    2.78us  Cast[bf16->f32]
  [33] +2.246ms    7.65us  BinaryFwd[Mul]·f32       k_even * cos
  [34] +2.278ms    2.78us  Cast[bf16->f32]
  [35] +2.295ms    7.65us  BinaryFwd[Mul]·f32       k_odd * sin
  [36] +2.324ms    3.04us  BinaryFwdNB[Sub]·f32     k left
  [37] +2.358ms    2.75us  Cast[bf16->f32]
  [38] +2.375ms    7.62us  BinaryFwd[Mul]·f32       k_even * sin
  [39] +2.406ms    2.82us  Cast[bf16->f32]
  [40] +2.424ms    7.62us  BinaryFwd[Mul]·f32       k_odd * cos
  [41] +2.455ms    3.01us  BinaryFwdNB[Add]·f32     k right
  [42] +2.507ms    4.51us  StackFwd·f32             k stack

── GQA：K/V 8 头复制到 32 头（2 RepeatInterleave）──────────────────
  [43] +2.594ms    9.57us  RepeatInterleaveFwd·f32  k: repeat_interleave(n_rep=4)
  [44] +2.663ms    9.47us  RepeatInterleaveFwd·bf16 v: repeat_interleave(n_rep=4)

── 转置到 (B,H,T,D) 并算 Kᵀ（4×[Fill + Transpose]）─────────────────
  [45] +2.745ms    4.19us  Fill·f32            [46] 19.33us  TransposeFwd·f32   q
  [47] +2.795ms    4.10us  Fill·f32            [48] 18.30us  TransposeFwd·f32   k
  [49] +2.847ms    4.16us  Fill·bf16           [50] 17.86us  TransposeFwd·bf16  v
  [51] +2.901ms    4.22us  Fill·f32            [52] 20.16us  TransposeFwd·f32   kᵀ

── ★ Attention 核心（全 bf16 Tensor Core / WMMA）───────────────────
  [53] +2.934ms    5.02us  Cast[f32->bf16]          q → bf16
  [54] +2.952ms    4.74us  Cast[f32->bf16]          kᵀ → bf16
  [55] +3.011ms    7.26us  cutlass_wmma_bf16_nn     SCORE  att = q · kᵀ   (B·H=128 批)
  [56] +3.048ms    4.93us  UnaryFwd[MulScalar]·bf16 att *= 1/√D (=0.125)
  [57] +3.087ms    2.85us  Cast[f32->bf16]          mask → bf16
  [58] +3.105ms    6.18us  MaskFwd·bf16             masked_fill(causal, -inf)
  [59] +3.141ms   26.14us  SoftmaxFwd·bf16          softmax(att, -1)
  [60] +3.191ms    7.26us  cutlass_wmma_bf16_nn     y = att · v

── 输出整理 + 投影（[Fill+Transpose] + Cast + bf16 GEMM + Cast）─────
  [61] +3.236ms    4.10us  Fill·bf16           [62] 18.02us  TransposeFwd·bf16  y→(B,T,H,D)
  [63] +3.315ms   28.61us  Cast[f32->bf16]          out_proj master 权重 → bf16 ★
  [64] +3.354ms   23.58us  ampere_bf16_...f2f_..._tn OutProj Linear 2048→2048
  [65] +3.420ms    5.02us  Cast[bf16->f32]          attn 输出 → fp32
  [66] +3.438ms    6.43us  BinaryFwdNB[Add]·f32     x = x + attn_out（残差，fp32）
```

##### 2.2.1.3 实测：Layer 0 的 MLP(SwiGLU) 半区（kernels [67..85]）

```
── RMSNorm ln2（6 kernels，全 fp32）[67..72] ───────────────────────
  [67] Pow·f32  [68] Mean·f32  [69] AddScalar·f32  [70] Rsqrt·f32  [71] Mul·f32  [72] Mul·f32

── MLP SwiGLU（每个 Linear = 2 Cast + 1 bf16 GEMM）─────────────────
  [73] +3.711ms    4.80us  Cast[f32->bf16]          c_fc 输入 → bf16
  [74] +3.732ms  102.21us  Cast[f32->bf16]          c_fc 权重(2048×8192) → bf16 ★★
  [75] +3.773ms   72.61us  ampere_bf16_...f2f_..._tn c_fc  Linear 2048→8192  (x1)
  [76] +3.822ms    6.53us  Cast[f32->bf16]          c_fc2 输入 → bf16
  [77] +3.840ms  100.45us  Cast[f32->bf16]          c_fc2 权重(2048×8192) → bf16 ★★
  [78] +3.866ms   72.58us  ampere_bf16_...f2f_..._tn c_fc2 Linear 2048→8192  (x2)
  [79] +3.906ms   14.18us  UnaryFwd[Sigmoid]·bf16   σ(x2)
  [80] +3.939ms   12.51us  BinaryFwdNB[Mul]·bf16    SiLU = x2 * σ(x2)
  [81] +3.968ms   14.66us  BinaryFwdNB[Mul]·bf16    x3 = x1 * SiLU(x2)（门控）
  [82] +3.999ms  102.53us  Cast[f32->bf16]          c_proj 权重(8192×2048) → bf16 ★★
  [83] +4.038ms   89.34us  ampere_bf16_...f2f_..._tn c_proj Linear 8192→2048
  [84] +4.089ms    5.44us  Cast[bf16->f32]          mlp 输出 → fp32
  [85] +4.108ms    6.66us  BinaryFwdNB[Add]·f32     x = x + mlp_out（残差，fp32）
```

> **★★ 首要发现：权重 cast 比 bf16 GEMM 本身还贵。** 

##### 2.2.1.4 迭代收尾（整轮各 1 次）

```
RMSNorm ln_f（6 kernels·f32）   Pow/Mean/AddScalar/Rsqrt/Mul/Mul   —— 第 33 个 RMSNorm
Cast[f32->bf16]                lm_head 输入 + 权重 → bf16
lm_head  ampere_bf16_...f2f_..._tn（128x256，grid 覆盖 vocab=128256）  ~0.70 ms   Linear 2048→128256
CrossEntropyFwd·f32            267.88 us   损失（autocast 保持 fp32）
UnaryFwd[MulScalar]·f32        loss / grad_accum_steps (=1)
```

#### 2.2.2 Backward Timeline
Backward 是 Forward 的逆序 autograd，共 **2116 kernels / 28.237 ms**（host 发射）。结构：

```
CrossEntropyBwd·f32        损失反向（700 us，autocast 保持 fp32）
lm_head 反向               ampere_s16816gemm_bf16_128x256_nn（dW，单 kernel 710 us）+ dX
── 每层逆序（ln_f→…→layer0）──────────────────────────────────────
  残差 Add 反向            BinaryBwdNB[Add]·f32
  MLP 反向                 c_proj/c_fc/c_fc2 各 2 bf16 GEMM（dX+dW）+ 权重 Cast[f32->bf16]
                           门控：BinaryBwdNBVec[Mul]·bf16、UnaryBwd[Sigmoid]·bf16
  RMSNorm ln2 反向         UnaryBwd[Pow/Rsqrt/AddScalar]·f32、GenericReduceBwd·f32、BinaryBwd[Mul]·f32
  残差 Add 反向
  Attention 反向           out_proj 2 GEMM；att·V 反向 cutlass_wmma_bf16_{nt,tn}；
                           SoftmaxBwd·bf16、MaskBwd、MulScalar 反向；Q·Kᵀ 反向 cutlass_wmma_bf16
                           RoPE 反向 StackBwd/SliceBwd/BinaryBwd[Mul]/Sub/Add；RepeatInterleaveBwd
                           qkv 2 bf16 GEMM + Cast
  RMSNorm ln1 反向
  ★ AccumulateGrad         每个产生梯度的张量：grad += rate·g（209 f32 + 16 bf16）
EmbeddingBwd·f32           词嵌入反向（scatter，1.314 ms）
```

Backward kernel Top（Σ GPU 时间，host 发射归属）：

| Kernel | n | Σ | 角色 |
|---|---|---|---|
| `ampere_s16816gemm_bf16_128x128_..._32x5_nt` | 49 | 4.612 ms | Linear 反向 dX/dW（bf16 Tensor Core） |
| `BinaryBwd[Mul]·f32` | 194 | 3.665 ms | RoPE/RMSNorm 乘法反向（fp32） |
| `Fill·f32` | 630 | 2.981 ms | 梯度/输出缓冲清零 |
| `ampere_s16816gemm_bf16_128x128_..._64x3_nn` | 32 | 2.136 ms | Linear 反向 |
| `TransposeFwd·f32` | 80 | 1.548 ms | 反向中的转置 |
| `Cast[f32->bf16]` | 178 | 1.491 ms | 反向 Linear 输入/权重下转 |
| `EmbeddingBwd·f32` | 1 | 1.314 ms | 词嵌入 scatter 反向 |
| `ampere_s16816gemm_bf16_256x128_..._{nt,nn}` | 48 | 2.420 ms | Linear 反向 |
| `AccumulateGrad`（f32+bf16） | 225 | 1.265 ms | autograd 梯度累加 |
| `cutlass_wmma_bf16_..._{nt,tn}` | 64 | 0.508 ms | Attention batched 反向 |

> `Fill` 在反向仍高达 **630 次**（2.981 ms）：每个需累加梯度的张量都要先清零 `.grad`，是「小 kernel 过多、launch 开销大」的主要来源。bf16 计算变快后，这类 launch/访存开销占比反而更突出。

#### 2.2.3 Optimizer Timeline

```
AdamAccumulateGrad·f32  ×115   Σ=32.280 ms   avg=280.69 us
```

- `AdamAccumulateGradKernel<float>` 是**融合的 Adam 更新**：单 kernel 内完成 `m=β1·m+(1-β1)g`、`v=β2·v+(1-β2)g²`、bias-correction、`param -= lr·m̂/(√v̂+eps)`（`accumulate_grad.cu`）。
- **115 ≈ 参数张量数**：每层 7 个（ln1.w、ln2.w、qkv.w、proj.w、c_fc.w、c_fc2.w、c_proj.w）×16 = 112，加 embedding、ln_f、lm_head。


### 2.3 优化项
- **★ 首要优化点——权重 Cast 比 GEMM 还贵**
- **FP32 Adam** 现为第一大头（32.3 ms / 115 次串行发射），可考虑优化器状态降精度或融合。
- **Fill 泛滥**（fwd 80 + bwd 630 = 710 次），合并为更少的大 fill 可降 launch 开销。
- **单 CUDA stream 串行**、无 kernel 并发；**CUDA Graph 或算子融合**（尤其把 Cast 融进 GEMM prologue）收益最大。

## 3 单GPU优化
### 3.1 Cast Kernel优化

Section 2 定位到 Cast 是 BF16 训练的第一大新增开销（Step_1 661 次 / 10.7 ms / 12.8%），且分**两类正交来源**，需分别治理：

| 类别 | 触发点 | 典型场景 | 方向 |
|---|---|---|---|
| ① autocast 边界 cast | `Function::Apply → cast_arg`（`autograd/function.cc`） | Linear/Matmul 前把**权重/激活** f32→bf16 | 下转 f32→bf16 |
| ② kernel 内 PromoteDataTypes | `BinaryForward/Backward`（`elementwise.cu`） | RoPE `q_even(bf16)×cos(f32)`、残差 `x(f32)+attn(bf16)` | 上转 bf16→f32 |

> 关键：`Add/Mul` 不在 `kOpCastPolicyMap` 中、不经 autocast，影子权重管不到；权重 cast 是 autocast 边界行为，融合 elementwise 也管不到。二者**各治一类、互不干扰**。

#### 3.1.1 优化一：影子权重（Shadow Weights）——消除权重 f32→bf16 下转

**为什么不能把权重 cast 直接融进 GEMM**：GEMM 走 cuBLAS `cublasGemmEx`（`common/gemm.cu:60-69`），其 `type_a/type_b` 必须声明 A/B 指针在显存中的**真实 dtype**，cuBLAS 按该类型直接读显存——**不支持「fp32 存储 → bf16 Tensor Core 计算」的内部转换，也没有 cast-in-prologue 的融合能力**。要吃到 bf16 Tensor Core，权重就必须**事先**在显存里是 bf16。三条路：①每步单独 cast（原始做法，权重张量大、单次 ~80–100 us，比 GEMM 还贵）；②CUTLASS mixed-input GEMM 自定义 `PredicatedTileIterator`/`MmaCore`（工程量大、收益不确定，已否决）；③**影子权重**——常驻一份 bf16 副本、在 Adam kernel 内近乎零成本刷新，autocast 直接复用。故选 ③。

> 对比：elementwise 是项目**自有 kernel**，可把上转直接融进 kernel（见 3.1.2）；而 GEMM 是 cuBLAS **闭源算子**、无法融合 → 只能靠影子权重「预物化 + 复用」绕过每步重复 cast。

**方案**：为每个 fp32 master 权重维护一份 **bf16 影子副本**，autocast 需要权重 bf16 时直接复用，跳过 CastKernel。

- **存储 / 注册**：`Adam` 持有 `shadow_weights_`（每参数一个 bf16 Tensor）；`thread_local unordered_map<const Tensor*, shared_ptr<Tensor>> g_shadow_registry` 以 master 裸指针为 key 映射到影子（`optimizer.cc`）。
- **零成本刷新**：Adam 更新与影子刷新**融合进同一 kernel** `AdamAccumulateGradShadowKernel<T,TShadow>`——在写回 fp32 master 的同一次遍历里顺带 `shadow_data[idx] = Cast<TShadow>(param_data[idx])`（`accumulate_grad.cu:206`，向量化版见 `:245`）。Adam 本就 memory-bound，多一次 bf16 store 几乎不增时间，避免「每步单独全量 cast 权重」。
- **autocast 接入**：`cast_arg` 在 `arg->To(target)` 之前先查 `GetShadow(arg.get())`，命中且 dtype 相符则直接返回影子（`autocast.h:122-130`）。


#### 3.1.2 优化二：elementwise 混合输入融合（P0）——消除 bf16→f32 上转

**方案**：把上转**融进 elementwise kernel**——用模板 `<Ta,Tb,Tout>` 在**寄存器内 widen**（`common::cuda::Cast<Tout>`），以 f32 计算，**输出 dtype 保持不变**。因 `PromoteDataTypes(bf16,f32)=f32`，`Tout` 恒为 float。

- **前向**：`BinaryForwardKernelMixed`（广播，走 `BroadcastMeta`）+ `BinaryForwardKernelNoBroadcastMixed`（同形状 grid-stride 快路径）。
- **反向**：`BinaryBackwardKernelNoBroadcastFastMixed` + `BinaryBackwardKernelMixed`（float cub 广播归约版逐行复制，仅把读取改为 `Cast<float>`）。
- **守卫**：仅当 `a_dtype≠b_dtype` 且 `promoted=f32` 且操作数**均连续**时走混合路径，否则回退原 `To()` 路径；混合分支排在 promote 之前，确保 CastKernel 不被启动。**同 dtype 路径完全未改**，零回归。
- **覆盖场景**：RoPE `q_even/q_odd(bf16) × cos/sin(f32)` 广播 Mul（`Slice` 物化为连续张量，广播 meta 安全）、残差 `x(f32)+attn_out(bf16)` 同形状 Add。

#### 3.1.3 优化结果

| 稳态每轮迭代 | 优化前 | 优化后 |
|---|---|---|
| Cast kernel 数 | **661** | **292**（−369，−56%） |
| 迭代时间（wall） | **114.6 ms** | **87.7 ms**（−23%） |
| GPU kernel 时间（Σ） | 76.8 ms | 71.7 ms（−6.7%） |

Cast 减少中，影子权重消除下转 81 个/轮、P0 融合消除上转 288 个/轮。wall 降幅（−23%）远大于 GPU 时间降幅（−6.7%）：该负载 launch 受限，每轮少发射 369 个 kernel 直接压缩了 host 侧开销。

**等价性验证**：`tests/autograd` CUDA 套件 **131/131 通过**（`ExpBackward/0` 当时为既有崩溃、与本次无关；该缺陷已后续单独修复——`autograd::Exp` 缺 `SetupContext` 导致 `Backward` 少传一个实参，现已按 `Tanh` 的写法保存前向 output 并补齐，全量 ctest 现为 327/327 全绿）；5-iter train loss **step1 恒为 `4.898438`、逐位一致 → 前向 bit-exact**，step2–5 的 ~0.001–0.005 波动是反向 `atomicAdd` 归约顺序的固有非确定性（同一二进制连跑 3 次亦自变），非优化引入。

**产物**：`nsys/out/llama3_5iter_nvtx_shadow.{nsys-rep,sqlite}`。

### 3.2 AdamAccumulateGradKernel优化

3.1 把影子刷新融进 Adam 后，`AdamAccumulateGradShadowKernel<float,__nv_bfloat16>` 成为单一最大 kernel：5 iter 共 545 次发射 / 155.9 ms，占全部 GPU kernel 时间的 **约 42%**（优化前后分别为 42.4% / 41.6%），即 Section 2.3 列出的「第一大头」。本节对它做**向量化访存**改造。

#### 3.2.1 瓶颈定位：ncu 证明已吃满 DRAM，而非发射受限

ncu（`llama3_adam_shadow_detailed.ncu-rep`）：

| 指标 | 实测 | 含义 |
|---|---|---|
| DRAM Throughput | **85.6%**（1.33 TB/s，峰值 1.555 TB/s） | 已接近带宽上限 |
| Compute (SM) Throughput | 26.33% | 算力大量闲置 |
| Long Scoreboard stall | 48.5 cycles（占 warp cycles 87.0%） | warp 全在等全局访存 |
| Scheduler No Eligible | 73.44% | 无可发射 warp，非发射瓶颈 |

每元素搬 **30 B**（grad/param/m/v 各读 4 B = 16 B，param/m/v 各写 4 B = 12 B，shadow 写 2 B），算术仅 ~5 FLOP → 算术强度 **0.17 FLOP/B**。

#### 3.2.2 方案：128-bit 向量化访存

- **载体**：进行float4访存：
```c++
VecT grad_vec = *reinterpret_cast<const VecT *>(&grad_data[base]);
VecT param_vec = *reinterpret_cast<const VecT *>(&param_data[base]);
VecT m_vec = *reinterpret_cast<const VecT *>(&m_data[base]);
VecT v_vec = *reinterpret_cast<const VecT *>(&v_data[base]);
```
- **尾部**：`num_elements % VecSize` 的余数由 kernel 内标量循环处理（`:192` / `:252`），对任意长度均正确。

#### 3.2.3 派发守卫：对齐 + SM 数缩放的最小尺寸

向量化并非无条件更优，派发处设两道守卫（`:285-289` 无影子 / `:339-344` 有影子）：

1. **对齐**：grad/param/m/v/shadow 五个指针都须按 `sizeof(T) * VecSize` 对齐，否则宽访存非法 → 回退标量。
2. **最小尺寸**：`num_elements >= sms * threads_per_block * vec_size`，如果尺寸过小而使用向量访存会导致sm无法用满，并且一个线程所占的寄存器和指令膨胀也会影响单线程性能。

#### 3.2.4 优化结果

**LLaMA3.2-1B 5-iter 实测**（同尺寸配对、per-launch 均值，与发射次数无关）：

| 张量规模 n | 发射数 | 优化前 | 优化后 | 加速 | reg/thread | 有效带宽 | 占 A100 峰值 |
|---|---|---|---|---|---|---|---|
| 262,668,288（embedding / lm_head） | 8 | 5958.80 us | 5738.98 us | **1.038x** | 18→46 | 1322→1373 GB/s | 85.0→**88.3%** |
| 16,777,216（MLP up/gate/down） | 232 | 381.56 us | 371.26 us | 1.028x | 18→46 | 1319→1356 GB/s | 84.8→87.2% |
| 6,291,456 | 78 | 145.70 us | 142.42 us | 1.023x | 18→46 | 1295→1325 GB/s | 83.3→85.2% |
| 4,194,304（q/o_proj） | 78 | 98.83 us | 96.65 us | 1.022x | 18→46 | 1273→1302 GB/s | 81.9→83.7% |
| 2,048（RMSNorm 权重） | 160 | 4.20 us | 4.24 us | 0.99x | 18→19 | — | 守卫路由至标量 |



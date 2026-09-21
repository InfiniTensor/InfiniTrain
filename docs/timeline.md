# nsys Timeline 详细分析

> 本文是 `kernel 优化.md` 第 2.2 节的详细数据：阶段与 kernel 家族聚合表，以及逐 kernel 的 timeline 展开（以 Layer 0 为例）。测量口径与基线说明见该文 §2.1。

按 host 发射归属（kernel 经 correlationId 关联到其发射时刻所在的 NVTX 子区间，避开异步偏移），每轮 step 分三个阶段：

| 阶段 | kernel 数 | Σ GPU 时间 | host 窗口 | 特征 |
|---|---|---|---|---|
| Forward | 1343 | 23.50 ms（含 339 Cast） | 62.84 ms | host 窗口是 GPU 时间的 2.7 倍，但其中 27.55 ms（44%）是 3 次 >1 ms 的阻塞调用（含一次 16.30 ms 的 loss DtoH 读回），扣除后真实发射工作约 35 ms，所以这不是单纯的 **host-bound** |
| Backward | 2116 | 28.28 ms（含 322 Cast） | 34.91 ms | host 与 GPU 接近平衡 |
| Optimizer | 115 | 32.29 ms（FP32 Adam，第一大头） | 0.86 ms | 115 个大 kernel 异步发完即返回，GPU 工作拖到后续窗口才排空 |
| **合计** | **3574** | **84.07 ms** | **98.61 ms** | 再加 ZeroGrad/DataUpload/LossReadback 三个不发 kernel 的窗口共 101.89 ms ≈ Step_1 wall 102.08 ms，nsys 下 host 全程无空闲 |

另有三个不发射 kernel 的子区间：ZeroGrad 0.46 ms、DataUpload 0.03 ms、LossReadback 2.79 ms（host 窗口）。六个窗口之和 101.89 ms ≈ Step_1 wall 102.08 ms，说明 nsys 下整轮 host 全程无空闲。但「无空闲」不等于「都在发射」：Forward 那 62.84 ms 里 Σ CUDA API 占 51.24 ms，其中 27.55 ms 集中在 3 次 >1 ms 的阻塞调用上，真正逐次发射 3477 个 API 只花 23.69 ms（见结论 6）。全轮 API 合计 71.29 ms 已被 CUPTI 放大，非 profiling 下会明显缩短，故「Forward host-bound」只是 nsys 下的强结论；跨阶段看，host 有相当比例的时间其实是在等 GPU。

按 kernel 家族聚合：

| 家族 | kernel 数 | Σ (ms) | 占比 |
|---|---|---|---|
| Adam（optimizer，fp32） | 115 | 32.29 | 38.4% |
| 其它 elementwise / 结构 | 1749 | 20.87 | 24.8% |
| GEMM（BF16 Tensor Core） | 339 | 16.85 | 20.0% |
| Cast（autocast 新增） | 661 | 10.74 | 12.8% |
| Fill | 710 | 3.31 | 3.9% |
| **合计** | **3574** | **84.07** | 100% |

其中 Cast 再拆方向：f32→bf16 下转 356 个 / 9.47 ms（权重下转为主，平均 26.6 μs），bf16→f32 上转 305 个 / 1.28 ms（激活上转，平均 4.2 μs）——下转个数只多 17% 但耗时是上转的 **7.4 倍**，这就是 3.1 优先消除权重下转的依据。

#### 2.2.1 Forward Timeline
Forward 共 **1343 kernels / 23.502 ms**（含 339 个 autocast Cast）。

##### 2.2.1.1 迭代入口（整轮各 1 次）

```
[ 0] +0.547ms   8.54us  EmbeddingFwd·f32   token embedding: (4,64) → (4,64,2048)
[ 1] +0.597ms   5.21us  SliceFwd·f32       freqs_cis 取本轮窗口的 cos/sin
[ 2] +0.654ms   2.78us  TriuFwd·f32        causal mask = Triu(ones(64,64),1)
```

之后进入 **16 层结构相同的 TransformerLayer**（单层 = RMSNorm(ln1) → Attention → 残差 → RMSNorm(ln2) → MLP → 残差），最后接 ln_f → lm_head → CrossEntropy。

##### 2.2.1.2 实测：Layer 0 的 Attention 半区（kernels [3..66]，含 cast）

`+ms` 为相对 Step_1 起点的 host 发射时刻，`us` 为 GPU 执行时间：

```
── RMSNorm ln1（6 kernels，全 fp32）────────────────────────────────
  [ 3] +0.679ms    8.03us  UnaryFwd[Pow]·f32        x²
  [ 4] +0.695ms    6.59us  Mean(Reduce)·f32         mean(x², -1)
  [ 5] +0.710ms    2.46us  UnaryFwd[AddScalar]·f32  + eps
  [ 6] +0.724ms    2.53us  UnaryFwd[Rsqrt]·f32      1/√(·)
  [ 7] +0.740ms   22.05us  BinaryFwd[Mul]·f32       x * rsqrt
  [ 8] +0.760ms   21.50us  BinaryFwd[Mul]·f32       norm * weight

── QKV 投影（2 Cast + 1 bf16 GEMM）─────────────────────────────────
  [ 9] +0.779ms    4.80us  Cast[f32->bf16]          ln1 输出 → bf16（Linear 输入）
  [10] +0.789ms   41.50us  Cast[f32->bf16]          QKV master 权重 → bf16 ★
  [11] +0.834ms   28.67us  ampere_bf16_...f2f_..._tn Linear 2048→3072 (=(H+2·KV)·D)

── 拆分 q/k/v + 取 RoPE cos/sin/even/odd（7 Slice）─────────────────
  [12] +0.869ms   16.48us  SliceFwd·bf16            q = qkv[..., :2048]
  [13] +0.903ms    6.72us  SliceFwd·bf16            k = qkv[..., 2048:2560]
  [14] +0.932ms    6.75us  SliceFwd·bf16            v = qkv[..., 2560:3072]
  [15] +0.969ms    6.24us  SliceFwd·f32             cos = freqs_cis[...,0]
  [16] +1.002ms    5.60us  SliceFwd·f32             sin = freqs_cis[...,1]
  [17] +1.033ms   11.23us  SliceFwd·bf16            q_even = q[..., 0::2]
  [18] +1.063ms   11.10us  SliceFwd·bf16            q_odd  = q[..., 1::2]

── RoPE 作用于 q（fp32 算术：每个 Mul 前先 Cast[bf16->f32] 上转）────
  [19] +1.077ms    3.52us  Cast[bf16->f32]          q_even ↑f32
  [20] +1.086ms   15.90us  BinaryFwd[Mul]·f32       q_even * cos
  [21] +1.107ms    3.39us  Cast[bf16->f32]          q_odd ↑f32
  [22] +1.116ms   15.81us  BinaryFwd[Mul]·f32       q_odd * sin
  [23] +1.131ms    4.13us  BinaryFwdNB[Sub]·f32     left = q_even·cos − q_odd·sin
  [24] +1.153ms    3.42us  Cast[bf16->f32]          q_even ↑f32
  [25] +1.162ms   15.87us  BinaryFwd[Mul]·f32       q_even * sin
  [26] +1.176ms    3.36us  Cast[bf16->f32]          q_odd ↑f32
  [27] +1.185ms   15.84us  BinaryFwd[Mul]·f32       q_odd * cos
  [28] +1.200ms    4.16us  BinaryFwdNB[Add]·f32     right = q_even·sin + q_odd·cos
  [29] +1.224ms    8.38us  StackFwd·f32             stack(left,right) → flatten

── RoPE 作用于 k（2 Slice + 4×[Cast+Mul] + Sub + Add + Stack）──────
  [30] +1.261ms    6.01us  SliceFwd·bf16            k_even
  [31] +1.288ms    5.95us  SliceFwd·bf16            k_odd
  [32] +1.301ms    2.78us  Cast[bf16->f32]
  [33] +1.310ms    7.71us  BinaryFwd[Mul]·f32       k_even * cos
  [34] +1.324ms    2.78us  Cast[bf16->f32]
  [35] +1.333ms    7.68us  BinaryFwd[Mul]·f32       k_odd * sin
  [36] +1.345ms    3.07us  BinaryFwdNB[Sub]·f32     k left
  [37] +1.361ms    2.78us  Cast[bf16->f32]
  [38] +1.369ms    7.68us  BinaryFwd[Mul]·f32       k_even * sin
  [39] +1.383ms    2.78us  Cast[bf16->f32]
  [40] +1.392ms    7.68us  BinaryFwd[Mul]·f32       k_odd * cos
  [41] +1.406ms    2.98us  BinaryFwdNB[Add]·f32     k right
  [42] +1.429ms    4.51us  StackFwd·f32             k stack

── GQA：K/V 8 头复制到 32 头（2 RepeatInterleave）──────────────────
  [43] +1.461ms    9.60us  RepeatInterleaveFwd·f32  k: repeat_interleave(n_rep=4)
  [44] +1.487ms    9.50us  RepeatInterleaveFwd·bf16 v: repeat_interleave(n_rep=4)

── 转置到 (B,H,T,D) 并算 Kᵀ（4×[Fill + Transpose]）─────────────────
  [45] +1.516ms    4.19us  Fill·f32            [46] 19.23us  TransposeFwd·f32   q
  [47] +1.543ms    4.13us  Fill·f32            [48] 18.46us  TransposeFwd·f32   k
  [49] +1.568ms    4.16us  Fill·bf16           [50] 18.08us  TransposeFwd·bf16  v
  [51] +1.595ms    4.13us  Fill·f32            [52] 19.97us  TransposeFwd·f32   kᵀ

── ★ Attention 核心（全 bf16 Tensor Core / WMMA）───────────────────
  [53] +1.628ms    4.96us  Cast[f32->bf16]          q → bf16
  [54] +1.639ms    4.80us  Cast[f32->bf16]          kᵀ → bf16
  [55] +1.681ms    7.29us  cutlass_wmma_bf16_nn     SCORE  att = q · kᵀ   (B·H=128 批)
  [56] +1.697ms    4.96us  UnaryFwd[MulScalar]·bf16 att *= 1/√D (=0.125)
  [57] +1.719ms    2.82us  Cast[f32->bf16]          mask → bf16
  [58] +1.729ms    6.34us  MaskFwd·bf16             masked_fill(causal, -inf)
  [59] +1.749ms   26.17us  SoftmaxFwd·bf16          softmax(att, -1)
  [60] +1.773ms    7.17us  cutlass_wmma_bf16_nn     y = att · v

── 输出整理 + 投影（[Fill+Transpose] + Cast + bf16 GEMM + Cast）─────
  [61] +1.794ms    4.10us  Fill·bf16           [62] 18.34us  TransposeFwd·bf16  y→(B,T,H,D)
  [63] +1.824ms   28.48us  Cast[f32->bf16]          out_proj master 权重 → bf16 ★
  [64] +1.844ms   23.93us  ampere_bf16_...f2f_..._tn OutProj Linear 2048→2048
  [65] +1.872ms    4.99us  Cast[bf16->f32]          attn 输出 → fp32
  [66] +1.882ms    6.33us  BinaryFwdNB[Add]·f32     x = x + attn_out（残差，fp32）
```

##### 2.2.1.3 实测：Layer 0 的 MLP(SwiGLU) 半区（kernels [67..85]）

```
── RMSNorm ln2（6 kernels，全 fp32）[67..72] ───────────────────────
  [67] Pow·f32  [68] Mean·f32  [69] AddScalar·f32  [70] Rsqrt·f32  [71] Mul·f32  [72] Mul·f32

── MLP SwiGLU（每个 Linear = 2 Cast + 1 bf16 GEMM）─────────────────
  [73] +1.997ms    4.80us  Cast[f32->bf16]          c_fc 输入 → bf16
  [74] +2.008ms  102.29us  Cast[f32->bf16]          c_fc 权重(2048×8192) → bf16 ★★
  [75] +2.025ms   73.24us  ampere_bf16_...f2f_..._tn c_fc  Linear 2048→8192  (x1)
  [76] +2.044ms    6.59us  Cast[f32->bf16]          c_fc2 输入 → bf16
  [77] +2.054ms  100.57us  Cast[f32->bf16]          c_fc2 权重(2048×8192) → bf16 ★★
  [78] +2.068ms   73.14us  ampere_bf16_...f2f_..._tn c_fc2 Linear 2048→8192  (x2)
  [79] +2.083ms   14.21us  UnaryFwd[Sigmoid]·bf16   σ(x2)
  [80] +2.098ms   12.57us  BinaryFwdNB[Mul]·bf16    SiLU = x2 * σ(x2)
  [81] +2.112ms   14.72us  BinaryFwdNB[Mul]·bf16    x3 = x1 * SiLU(x2)（门控）
  [82] +2.125ms  102.58us  Cast[f32->bf16]          c_proj 权重(8192×2048) → bf16 ★★
  [83] +2.150ms   90.78us  ampere_bf16_...f2f_..._tn c_proj Linear 8192→2048
  [84] +2.171ms    5.34us  Cast[bf16->f32]          mlp 输出 → fp32
  [85] +2.181ms    6.72us  BinaryFwdNB[Add]·f32     x = x + mlp_out（残差，fp32）
```

> **★★ 首要发现：权重 cast 比 bf16 GEMM 本身还贵。**  

##### 2.2.1.4 迭代收尾（整轮各 1 次）

```
[1331..1336] RMSNorm ln_f（6 kernels·f32）   Pow 7.87 / Mean 6.59 / AddScalar 2.40 / Rsqrt 2.50 / Mul 21.76 / Mul 21.60 us   —— 第 33 个 RMSNorm，+46.852ms 起
[1337] +46.927ms    4.93us  Cast f32→bf16            lm_head 输入 → bf16
[1338] +46.937ms 1533.17us  Cast f32→bf16            lm_head 权重 2048×128256 → bf16
                           ★★ 全步最贵的单个 kernel（不只是最贵 Cast），是其 GEMM 698.65us 的 2.19 倍
[1339] +46.955ms  698.65us  ampere_bf16_...f2f_..._tn（128x256，grid 覆盖 vocab=128256）   lm_head Linear 2048→128256
[1340] +46.998ms  197.04us  Cast bf16→f32：logits 上转（autocast 让 CE 走 fp32）
[1341] +47.010ms  268.13us  CrossEntropyFwd·f32   损失
[1342] +63.353ms    2.65us  UnaryFwd MulScalar·f32   loss / grad_accum_steps 为 1

注：[1341] 与 [1342] 之间空了 16.34 ms——正是结论 6 里那次 1024 B loss DtoH 读回造成的 host 阻塞。
Forward 的 GPU 工作在 +47.0 ms 就已全部发出，host 却要等到 +63.3 ms 才能发出最后一个标量 kernel。
```

#### 2.2.2 Backward Timeline
Backward 是 Forward 的逆序 autograd，共 **2116 kernels / 28.276 ms**（host 发射）。结构：

```
CrossEntropyBwd·f32        损失反向（700 us，autocast 保持 fp32）
lm_head 反向               ampere_s16816gemm_bf16_128x256_nn（dW，单 kernel 686 us）+ dX
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
EmbeddingBwd·f32           词嵌入反向（scatter，1.313 ms）
```

Backward kernel Top（Σ GPU 时间，host 发射归属）：

| Kernel | n | Σ | 角色 |
|---|---|---|---|
| `ampere_s16816gemm_bf16_128x128_..._32x5_nt` | 49 | 4.635 ms | Linear 反向 dX/dW（bf16 Tensor Core） |
| `BinaryBwd[Mul]·f32` | 194 | 3.674 ms | RoPE/RMSNorm 乘法反向（fp32） |
| `Fill·f32` | 630 | 2.982 ms | 梯度/输出缓冲清零 |
| `ampere_s16816gemm_bf16_128x128_..._64x3_nn` | 32 | 2.137 ms | Linear 反向 |
| `TransposeFwd·f32` | 80 | 1.551 ms | 反向中的转置 |
| `Cast[f32->bf16]` | 178 | 1.492 ms | 反向 Linear 输入/权重下转 |
| `EmbeddingBwd·f32` | 1 | 1.313 ms | 词嵌入 scatter 反向 |
| `ampere_s16816gemm_bf16_256x128_..._{nt,nn}` | 48 | 2.428 ms | Linear 反向 |
| `AccumulateGrad`（f32+bf16） | 225 | 1.267 ms | autograd 梯度累加 |
| `cutlass_wmma_bf16_..._{nt,tn}` | 64 | 0.506 ms | Attention batched 反向 |

> `Fill` 在反向仍高达 **630 次**（2.982 ms）：每个需累加梯度的张量都要先清零 `.grad`，是「小 kernel 过多、launch 开销大」的主要来源。bf16 计算变快后，这类 launch/访存开销占比反而更突出。

#### 2.2.3 Optimizer Timeline

```
AdamAccumulateGrad·f32  ×115   Σ=32.292 ms   avg=280.80 us
```

- `AdamAccumulateGradKernel<float>` 是**融合的 Adam 更新**：单 kernel 内完成 `m=β1·m+(1-β1)g`、`v=β2·v+(1-β2)g²`、bias-correction、`param -= lr·m̂/(√v̂+eps)`（`accumulate_grad.cu`）。
- **115 ≈ 参数张量数**：每层 7 个（ln1.w、ln2.w、qkv.w、proj.w、c_fc.w、c_fc2.w、c_proj.w）×16 = 112，加 embedding、ln_f、lm_head。

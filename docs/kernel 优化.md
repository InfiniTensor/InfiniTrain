# InfiniTrain Kernel优化
## 1 实验环境

| 项 | 配置 |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB（虚拟机 passthrough，Ampere / sm_80） |
| CUDA / 驱动 | 12.4 / 550.107.02 |
| 工具链 | gcc 13.4、cmake 3.30.9、nsys 2023.4.4 |
| 构建（全文统一） | `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -DBUILD_TEST=OFF -DNVTX_MODE=ON -DUSE_CUDA=ON -DUSE_NCCL=ON` |
| 模型 | LLaMA3.2-1B（16 层，n_embd 2048，n_head 32，GQA n_kv_head 8，vocab 128256） |
| 精度 | BF16 autocast：Linear/Matmul 走 Tensor Core，master 权重与 Adam 保持 FP32 |
| 训练配置 | batch_size 4 × seq_len 64，total_batch_size 256；Section 2 采集 5 iter，3.2.4 对照 10 iter ×3 次 |
| Profiling | Nsight Systems（nsys）+ NVTX，逐 step / 逐阶段标注 |

Section 2 的 trace 用基线 commit `9946886` 的 `build/llama3`——与 3.2.4 三配置对照里的 **A 配置是同一个二进制**，所以本章的基线与第三章的收益可以直接对齐，不存在构建口径差。

采集与汇总：

```bash
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=none --cpuctxsw=none \
  --output=nsys/out/llama3_5iter_nvtx_release --export=sqlite --force-overwrite=true \
  ./build/llama3 --device cuda --dtype bfloat16 \
    --input_bin data/llama3/tiny_shakespeare_train.bin \
    --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
    --num_iteration 5 --batch_size 4 --sequence_length 64 --total_batch_size 256
# 注意：--export=sqlite 生成的 .sqlite 会比 .nsys-rep 旧，stats 必须带 --force-export=true
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report nvtx_sum --format table \
  --force-overwrite=true --force-export=true \
  --output nsys/out/llama3_5iter_release_stats nsys/out/llama3_5iter_nvtx_release.nsys-rep
# 本章表格的来源：逐步窗口/GPU busy 用 analyze_steps.py，阶段归属与逐 kernel timeline 用 extract_timeline.py
python3 nsys/out/analyze_steps.py     nsys/out/llama3_5iter_nvtx_release.sqlite
python3 nsys/out/extract_timeline.py  nsys/out/llama3_5iter_nvtx_release.sqlite
```

## 2 性能现状

### 2.1 nsys分析结果

非 profiling 的稳态每轮 **82.05 ms**，单 A100 吞吐 **3122 tok/s**。nsys 采集下稳态每轮 94.80 ms——两者的差是 CUPTI 开销，见下方口径说明；

| step | wall (ms) | Σ kernel (ms) | Σ memcpy (ms) | kernels | GPU busy (ms) | GPU util |
|---|---|---|---|---|---|---|
| Step_0 | 228.9 | 54.14 | 3.18 | 3478 | 57.3 | 25.0%（warm-up，host-bound） |
| Step_1 | 102.1 | 83.74 | 3.53 | 3573 | 87.3 | 85.5% |
| Step_2 | 96.4 | 77.84 | 3.11 | 3574 | 80.9 | 84.0% |
| Step_3 | 89.9 | 73.79 | 2.40 | 3574 | 76.2 | 84.7% |
| Step_4 | 90.8 | 73.34 | 2.56 | 3573 | 75.9 | 83.6% |

Step_1 Top kernel（分母 = 该窗口 Σ kernel 83.74 ms）：

| # | kernel | n | Σ (ms) | 占比 |
|---|---|---|---|---|
| 1 | `AdamAccumulateGradKernel<float>` | 114 | 31.96 | 38.2% |
| 2 | `CastKernel<bf16,float>`（f32→bf16） | 356 | 9.47 | 11.3% |
| 3 | `ampere_s16816gemm_bf16_128x128_nt` | 49 | 4.64 | 5.5% |
| 4 | `ampere_bf16_s16816gemm_128x256_f2f_tn` | 49 | 4.51 | 5.4% |
| 5 | `BinaryBackwardKernel<float>`(Mul) | 194 | 3.67 | 4.4% |
| 6 | `FillKernel<float>` | 678 | 3.18 | 3.8% |
| – | **全部 GEMM（BF16 Tensor Core）** | 339 | **16.85** | **20.0%** |
| – | **全部 Cast（autocast）** | 661 | **10.74** | **12.8%** |


nsys 下稳态 GPU util 为 84–85%；非 profiling 下约 **93%**。

> **测量口径**：本次 trace 由基线 commit `9946886` 的 **Release** 构建采集。需要区分两类量：
![nsys timeline](image.png)

**关键结论**：

1）BF16 训练引入了大量 Cast kernel 完成 bf16↔FP32 转换：**661 个/轮、10.74 ms、占 kernel 时间 12.8%**；其中 f32→bf16 下转 356 个就占 9.47 ms（以权重下转为主，见 2.2.1.3 的 ★★）。

2）Adam 优化器是所有 kernel 中耗时最多的：**115 次发射、32.29 ms、占 38.4%**，单一 kernel 家族接近全部 GPU 时间的四成,其中lm_head和embedding的参数优化，单个kernel就占了5ms左右的时间。

3）整个训练过程只使用了一个 stream。

4）FillKernel占据的时间也比较长，可以尝试用异步memset替换。

5）CrossEntropyForward API占据了大量的host时间完成D2H的memcpy_async。当然并不是copy本身需要很长的时间，这个异步DMA会排在一个异步执行队列中，他要等前面的执行完成才能执行。并且CrossEntrypyForward后续mean计算放在了CPU中，他依赖于这次D2H拷贝的数据。也就是说CrossEntropy引入了一个同步点。

6）SliceForward Kernel中有5次异步H2D memory copy，其中有几次时间很长推测原因为队列耗尽，

### 2.2 Timeline分析

参考timeline.md


### 2.3 优化项
- **cast kernel泛滥**
- **FP32 Adam 降低内存带宽要求**
- **Fill 泛滥**
- **单 CUDA stream 串行**
- **Forward 末的 1024 B loss DtoH 读回**：GPU 侧仅 3.46 us，却让 host 阻塞 16.30 ms 等整条队列排空（四个构建都测到 12.2–16.3 ms）。改 pinned memory 异步读回、或只在需要打日志的轮次读，可直接消掉这段阻塞。
- **每轮 1515 次微小 HtoD**（合计仅 76 KB、平均 50.4 B/次）：标量是逐个拷上 GPU 的，可合并为一次批量拷贝。

## 3 单GPU优化
### 3.1 Cast Kernel优化

Section 2 定位到 Cast 是 BF16 训练的第二大开销（Step_1 661 次 / 10.7 ms / 12.8%），且分**两类正交来源**，需分别治理：

#### 3.1.1 优化一：影子权重（Shadow Weights）——消除gemm前的cast kernel

有两个方案：1）将cast kernel和后续的gemm kernel进行融合
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
| GPU kernel 时间（Σ，nsys，Step_3–4 稳态窗口） | 73.56 ms | 68.65 ms（−6.7%，−4.91 ms） |
| wall 稳态 step5–10（Release，无 profiling） | **82.05 ms** | **77.13 ms（−6.0%）** |

Cast 减少中，影子权重消除下转 81 个/轮、P0 融合消除上转 288 个/轮。wall 降幅（−6.0%，4.92 ms）与 GPU kernel 时间降幅（−6.7%，4.91 ms）几乎完全相等：省下的主要就是被消除的 CastKernel 自身的 GPU 时间；每轮少发射 369 个 kernel 对 host 侧的压缩在无 profiling 环境下只占很小一部分。

> **修正**：本节此前记录的 `114.6 → 87.7 ms（−23%）` 取自两份 nsys 的 NVTX 稳态窗口，把 26.95 ms 的差值整体当成了收益，实际高估 **5.2×**。其中只有 5.18 ms 属于 cast 优化，另外 21.77 ms 是测量偏差：
>
> - **19.83 ms**：基线那份 nsys 用旧 `build/llama3` 采集，其 `CMAKE_BUILD_TYPE` 为空（host 侧无 `-O3`/`-DNDEBUG`）。非 profiling 下这只值 6.67 ms（同一 commit `9946886` 以 Release 重建后，稳态 88.72 → 82.05 ms）；但在 nsys 下，同一 commit 的 `-O0` trace 稳态窗口是 114.63 ms、Release trace 是 94.80 ms，即值 **19.83 ms**（3.0 倍）。`-O0` 与 CUPTI 是「相乘」而非「相加」：host 越慢，逐调用插桩越落在关键路径上。
> - **1.94 ms**：CUPTI 逐调用插桩，而基线每轮比优化后多 ~351 次 launch，开销因此不对称。用同一二进制的 nsys 与非 nsys 运行按 step 配对（steps 2–5，各 3 次取逐步中位数再求窗口均值）实测：nsys 给基线叠加 +10.02 ms（+11.8%），给 cast 版叠加 +8.08 ms（+10.2%），折合基线 ≈2.0 μs/次 API（每步 3478 次 `cudaLaunchKernel` 与 1515 次 `cudaMemcpyAsync`）；两者之差 1.94 ms 才是差异化的 CUPTI 开销。
>
> 三分量 5.18＋1.94＋19.83＝26.95 ms，与两份 trace 稳态窗口之差 114.63 − 87.68 精确闭合；非 profiling 配对下 A 84.78 → B 79.60 ms，真实 cast 收益 5.18 ms（与上表 step5–10 口径的 4.92 ms 同量级，差异来自取样窗口与中位数取法）。
>
> 结论：**跨配置比较必须同构建类型、同 profiling 状态**；nsys 的 NVTX 窗口只适合做单份 trace 内部的 kernel 构成分析，不适合直接相减当收益。受控对照见 3.2.4。

**等价性验证**：`tests/autograd` CUDA 套件 **131/131 通过**（`ExpBackward/0` 当时为既有崩溃、与本次无关；该缺陷已后续单独修复——`autograd::Exp` 缺 `SetupContext` 导致 `Backward` 少传一个实参，现已按 `Tanh` 的写法保存前向 output 并补齐，全量 ctest 现为 327/327 全绿）；5-iter train loss **step1 恒为 `4.898438`、逐位一致 → 前向 bit-exact**，step2–5 的 ~0.001–0.005 波动是反向 `atomicAdd` 归约顺序的固有非确定性（同一二进制连跑 3 次亦自变），非优化引入。

**产物**：`nsys/out/llama3_5iter_nvtx_shadow.{nsys-rep,sqlite}`。

### 3.2 AdamAccumulateGradKernel优化

3.1 把影子刷新融进 Adam 后，`AdamAccumulateGradShadowKernel<float,__nv_bfloat16>` 成为单一最大 kernel：5 iter 共 545 次发射 / 154.05 ms，占全部 GPU kernel 时间的 **约 42%**（adam 向量化前后分别为 41.9% / 41.6%——B 的占比高于第二章基线 A 的 37.5%，是因为 cast 优化削掉了分母里的 369 个 Cast kernel），即 Section 2.3 列出的「第一大头」。本节对它做**向量化访存**改造。

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

**LLaMA3.2-1B 5-iter 实测**（同尺寸配对、per-launch 均值，与发射次数无关；「发射数」列取 C 配置，B 的同尺寸组依次为 8/228/77/76/156、合计 545）：

| 张量规模 n | 发射数 | 优化前 | 优化后 | 加速 | reg/thread | 有效带宽 | 占 A100 峰值 |
|---|---|---|---|---|---|---|---|
| 262,668,288（embedding / lm_head） | 8 | 5958.80 us | 5738.98 us | **1.038x** | 18→46 | 1322→1373 GB/s | 85.0→**88.3%** |
| 16,777,216（MLP up/gate/down） | 232 | 381.56 us | 371.26 us | 1.028x | 18→46 | 1319→1356 GB/s | 84.8→87.2% |
| 6,291,456 | 78 | 145.70 us | 142.42 us | 1.023x | 18→46 | 1295→1325 GB/s | 83.3→85.2% |
| 4,194,304（q/o_proj） | 78 | 98.83 us | 96.65 us | 1.022x | 18→46 | 1273→1302 GB/s | 81.9→83.7% |
| 2,048（RMSNorm 权重） | 160 | 4.20 us | 4.24 us | 0.99x | 18→19 | — | 守卫路由至标量 |

**端到端稳态时延：三配置受控对照**

三个配置 checkout 到同一 commit 链的三个点，用**完全相同**的构建 flags 各自全新构建，在**无 profiling** 环境下每配置跑 10 轮 × 重复 3 次。统计法：每步先取 3 次运行的中位数，再对窗口求均值（对离群步稳健）。

| 配置 | commit | 构建目录 | peak used | step5–10 稳态 | 吞吐量 | step1 warm-up |
|---|---|---|---|---|---|---|
| A 基线（无优化） | `9946886` | `build` | 23143 MB | **82.05 ms** | 3122 tok/s | 185.9 ms |
| B ＋cast 优化 | `0e2dc18` | `build_cast` | 26001 MB | **77.13 ms** | 3321 tok/s | 212.2 ms |
| C ＋adam 向量化 | `25c3bd8` | `build_adam` | 26001 MB | **75.69 ms** | 3378 tok/s | 214.9 ms |

| 增量 | 稳态时延 | 降幅 | 吞吐量 | 折合每轮 |
|---|---|---|---|---|
| A→B（cast 优化） | 82.05 → 77.13 ms | **−5.99%** | +6.4% | −4.92 ms |
| B→C（adam 向量化） | 77.13 → 75.69 ms | **−1.86%** | +1.7% | −1.44 ms |
| A→C（合计） | 82.05 → 75.69 ms | **−7.74%** | +8.2% | −6.36 ms |

窗口敏感性（A→B / B→C / A→C）：step3–10 `−6.42% / −0.56% / −6.94%`；step4–10 `−5.97% / −2.01% / −7.86%`；step5–10 `−5.99% / −1.86% / −7.74%`。两种统计法（逐步中位数 / 先平均每次运行的窗口再汇总）在 step5–10 上给出 82.05/77.13/75.69 与 81.95/77.09/75.62，差异 <0.2%。

> **可信度**：3 次重复的运行间极差（step5–10 窗口）为 A ±1.02%、B **±0.41%**、C ±0.90%；B 最稳（step4–10 窗口三次运行仅差 0.04%）。A→B 的 6.0% 远超噪声，可直接采信。B→C 的 1.86%（−1.44 ms）量级接近 C 自身的噪声带（±0.9% ≈ ±0.7 ms），且随窗口在 0.56%–2.01% 间摆动（含未完全热身的 step3 时最低），故只能作为量级结论；它与 nsys 的独立测量同向、但量级更小：B→C 的 Adam 自身 Σ 154.05 → 151.37 ms（5 iter），折合 −0.54 ms/轮、占 B 稳态 wall 的 0.69%；全 kernel Σ 367.45 → 363.68 ms，折合 −0.75 ms/轮。即 GPU 侧只解释了 wall 降幅 −1.44 ms 的一半左右，另一半落在噪声带内。

> **代价**：影子权重使 peak used 23143 → 26001 MB（**+2858 MB，+12.4%**），首轮 warm-up 185.9 → 212.2 ms（一次性构建 bf16 副本）。adam 向量化不改变显存占用。

> **数值等价性**：三配置 step1 loss 均为 `4.898438`，逐位一致。step2 起该训练**本身即非确定**——同一二进制连跑 3 次，基线 step2 为 4.548160 / 4.543328 / 4.542815（跨度 0.005）；各配置逐步的运行间跨度为 A ≤0.0060、B ≤0.0052、C ≤0.0073，而跨配置差值 |B−A| ≤0.0067、|C−B| ≤0.0083 与之同量级且区间重叠（如 step2 的 C run2 = 4.544229 落在 B 的 4.543237–4.544785 之内）。因此 loss 对照既不能证明也不能否证位级等价；向量化的位级等价由共享 `AdamUpdateElement` 与显式 FMA 收缩在代码层面保证。

**复现**（两个已踩到的坑：① `nvcc` 可能不在 PATH；② **首次配置若因缺 nvcc 而失败，会在构建目录留下 `CMAKE_CUDA_FLAGS_RELEASE` 为空的残缺 cache，后续成功的配置不会重新初始化该条目，导致所有 `.cu` 不带 `-O3`/`-DNDEBUG` 编译而 host 侧 `.cc` 仍带——一个很难发现的混合态**。用 `--fresh` 避开，并在构建前校验）：

```bash
export PATH=/usr/local/cuda/bin:$PATH
for cfg in 9946886:build 0e2dc18:build_cast 25c3bd8:build_adam; do
  git checkout "${cfg%%:*}"
  cmake --fresh -S . -B "${cfg##*:}" -DUSE_CUDA=ON -DUSE_NCCL=ON -DNVTX_MODE=ON \
        -DBUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
  # 必须校验，否则可能在未优化的 CUDA 代码上得出错误结论
  grep -q '^CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3 -DNDEBUG' "${cfg##*:}/CMakeCache.txt" || { echo "flags 异常"; break; }
  cmake --build "${cfg##*:}" -j16
  for i in 1 2 3; do
    "./${cfg##*:}/llama3" --device cuda --dtype bfloat16 \
      --input_bin data/llama3/tiny_shakespeare_train.bin \
      --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
      --num_iteration 10 --batch_size 4 --sequence_length 64 --total_batch_size 256
  done
done
```

### 3.3 FillKernel 优化

Section 2.2 显示 Fill 家族每轮 **710 次 / 3.31 ms / 3.9%**，平均 **3.6–4.2 μs/次**——单次 GPU 执行时间几乎全部是 launch/setup，是典型的 **launch-overhead 主导型**开销，与带宽或算力无关。治理分三档：

- **档 1（P0，3.3.1–3.3.4）**：删除**冗余** Fill——即前置的 `Fill(0)` 后续 kernel 会覆盖全部输出元素，Fill 是死代码。
- **档 2（P1，3.3.5–3.3.6）**：`Tensor::Fill(0.0)` 在 CUDA 后端特化为 `cudaMemsetAsync`，走 DMA 引擎，不占 SM，脱离 kernel launch 路径。
- **档 3（P2，未做）**：ZeroGrad arena 化，把 115 次每步 Fill 合并为 1 次大 memset。

#### 3.3.1 P0 判据

与仓库里 matmul/linear/stack/softmax/reduction/cross_entropy 已有的 `// No Fill(0) needed: ...` 注释同一判据：

> **kernel grid 覆盖 output 全部元素、且每个元素恰好被写入一次（无 atomicAdd、无 early-return 遗漏）**，则前置的 `Fill(0)` 是死代码，可以删除。

按此判据 sweep 全仓库 `Fill(0.0)` 调用点，[transform.cu](../infini_train/src/kernels/cuda/transform.cu) 中 **6 处**满足；其他文件（slice/split/gather/embedding/layernorm/elementwise 广播路径）的 Fill 都是 scatter 或 atomicAdd 起点，**保留**，属档 2/3 范畴。

#### 3.3.2 P0 方案：删除 transform.cu 6 处 Fill

| # | 函数 | 位置 | kernel 覆盖证据 |
|---|---|---|---|
| 1 | `TrilBackward` | transform.cu:96 | `TrilBackwardKernel` L62–76：每 idx 无条件写 `grad_output[idx]` 或 `T(0)`，grid 覆盖 `[0, rows*cols)` |
| 2 | `TriuBackward` | transform.cu:183 | `TriuBackwardKernel` L150–164：同上（else 分支写 `T(0)`） |
| 3 | `TransposeForward` | transform.cu:275 | `TransposeForwardKernel` L194–222：`output[idx] = input[in_flat_idx]`，`num_blocks = ceil(num_elements/256)` 全覆盖，无 atomicAdd |
| 4 | `MaskBackward` (rows) | transform.cu:441 | `MaskLeadsBackwardKernel` L410–415：每 i 无条件写 `mask ? T(0) : grad_output[i]`，`rows*inner = grad_output->NumElements()` |
| 5 | `MaskBackward` (tail) | transform.cu:455 | `MaskBackwardKernel` L402–407：同上，`batch_size*mask_size = grad_output->NumElements()` |
| 6 | `RepeatInterleaveBackward` | transform.cu:568 | `RepeatInterleaveBackwardKernel` L520–540：**gather-reduce 而非 scatter**，每 thread 独占一个 `grad_input[idx]`，寄存器内累加 `repeat` 个 grad_output 后单点写 |

删除后在原位置留一句 `// No Fill(0) needed: ...` 说明判据（与仓库已有风格一致），防止后人回退。

#### 3.3.3 P0 消除数量估算

按 doc 2.2.1.2 / 2.2.2 的每层 kernel 频次 × 16 层推算：

| 场景 | 每 step 消除的 Fill 数 |
|---|---|
| Forward `TransposeForward`（每层 5 × 16 层） | 80 |
| Backward `TransposeBackward`（= `TransposeForward`，每层 5 × 16 层） | 80 |
| Backward `RepeatInterleaveBackward`（每层 2 × 16 层，GQA K/V） | 32 |
| Backward `MaskBackward`（每层 1 × 16 层，causal mask） | 16 |
| Backward `TriuBackward`（causal mask 反向，整轮 1 次） | 1 |
| **合计** | **209**（占 doc 2.3 中 710 的 **29.4%**） |

其中 Forward 那 80 次**恰好等于** doc 2.3 里 "Fill 泛滥（fwd 80 + bwd 630）" 的 fwd 全部——即 P0 把 forward 侧 Fill 清零。

#### 3.3.4 P0 优化结果

**编译**：`build_new`（BUILD_TEST=ON）+ `build_adam`（HEAD `25c3bd8`）双目录 CUDA kernel 重编 + 全项目链接通过，无警告。

**单元测试**：

- 定向 transform 套件（`TransposeForward/Backward`、`Tril`、`Triu`、`Mask`、`RepeatInterleave`）：**14/14 pass**
- 全量 CUDA autograd：**263 pass / 1 pre-existing skip**（`BFloat16MulBroadcastBackwardLargeBlock`，与本次无关）

**数值等价性**：`build_new/llama3` 5-iter × 3 次运行，step1 loss 恒为 **`4.898438`**（逐位一致，与 doc 3.2.4 中 A/B/C 三配置 step1 完全相同）→ **前向 bit-exact**，直接验证 `TransposeForward` 的 Fill 删除无副作用。step2 三次运行分别 `4.544808 / 4.547627 / 4.543972`，跨度 0.0037，落在 doc 3.2.4 记录的 C 配置自变区间（≤0.0073）之内 → 未引入系统偏差。

**端到端稳态时延**（`build_new/llama3`，无 profiling，10 iter × 3 次，每步取中位数再对窗口求均值，与 doc 3.2.4 同口径）：

| 配置 | commit | 构建目录 | step5–10 稳态 | 运行间极差 |
|---|---|---|---|---|
| C ＋adam 向量化（基线） | `25c3bd8` | `build_adam` | **75.69 ms** | ±0.90% |
| D ＋P0 Fill 删除 | `25c3bd8` + 未提交改动 | `build_new` | **75.41 ms** | ±0.72% |

D 的窗口敏感性：step3–10 **75.91 ms** / step4–10 **75.51 ms** / step5–10 **75.41 ms**（C 基线的 step3–10 / step4–10 绝对值在 doc 3.2.4 中未直接记录，仅给出相对降幅，故不列对照）。

| 增量 | 稳态时延 | 降幅 | 折合每轮 |
|---|---|---|---|
| C→D（P0 Fill 删除） | 75.69 → 75.41 ms | **−0.37%** | −0.28 ms |

> **可信度**：−0.28 ms 落在 D 自身噪声带（±0.72% ≈ ±0.54 ms）之内，只能作为**量级结论**，不能作为强信号。这与理论预期基本一致：209 次 Fill × 4 μs/次 ≈ **0.84 ms** GPU 时间被消除，但非 profiling 环境下 host 侧发射早已与 GPU 执行流水化，Fill 的 GPU 时间大部分被 launch gap 吸收，wall 上只留下小于理论值的一半；另一部分被运行间噪声淹没。nsys 下（CUPTI 逐调用放大）收益会更明显——doc 3.1.3 修正块里已经量化过：基线每轮 3478 次 `cudaLaunchKernel` 在 nsys 下值 ~2.0 μs/次，P0 消除 209 次 launch 折合约 **0.42 ms/step 的 host 侧 CUPTI 开销**，加上 GPU 侧 0.84 ms，nsys 稳态窗口预计降 **~1.3 ms/step**（待复测）。

> **主要收益不在 wall**：P0 的核心价值是**清理死代码 + 减少 launch 数**，为后续档 2（`cudaMemsetAsync`）和档 4（CUDA Graph）铺路——Graph 捕获对 launch 数敏感，越少越好；且 forward 侧 Fill 清零后，timeline 上 `[Fill + Transpose]` 成对模式消失，可读性提升。

**产物**：`build_new/llama3`（含 P0 改动）；未生成新 nsys trace（待档 2 完成后一起复测）。

#### 3.3.5 P1 方案：Fill(0) 特化为 `cudaMemsetAsync`

P0 只消除了 **transform.cu 内满足全覆盖判据**的 6 处 Fill（209 次/step）。剩下的 ~501 次/step 都是真需要零起点的场景：ZeroGrad（115）、LayerNormBwd grad_weight/grad_bias（66，atomicAdd）、EmbeddingBwd（1，atomicAdd scatter）、Slice/Split/Gather Bwd（~144，部分写入）、BinaryBwd 广播路径 grad_a/grad_b（~256，atomicAdd 广播归约）。这些不能删，但可以**换一条更便宜的路径**。

**方案**：在 [fill.cu](../infini_train/src/kernels/cuda/fill.cu) 的 CUDA `Fill` 入口处加一条快路径：当且仅当

1. `Scalar` 的存储位模式**全零**（`kBool/kUInt64`: `u == 0`；`kInt64`: `i == 0`；`kDouble`: `memcpy` 后 `bits == 0`）；
2. tensor `IsContiguous()`（当前实现恒为 true，预留接口给未来的 strided view）；
3. `kDataTypeToSize` 命中（覆盖全部 13 个 dtype 枚举）；

三条同时成立，则调 `cudaMemsetAsync(data_ptr, 0, num_elements * dtype_size, stream)` 并 `return`，不再进 `DispatchCudaFunc` / `FillKernel<<<...>>>`；任一条不成立则**回退**到原有 `FillKernel` 路径（包括 `-0.0` 这种数值等于 0 但符号位为 1 的边界情况，以及非零标量如 `Fill(1.0)` / `Fill(-inf)`）。

**为什么位零判据是关键**：`cudaMemsetAsync` 只能写**字节模式**，不能写任意标量。对 InfiniTrain 支持的所有 dtype（bool / intN / uintN / fp16 / bf16 / fp32 / fp64），**+0 的位模式恰好是全零字节**，所以 memset(0) 与 FillKernel(+0) 位级等价。-0.0 的位模式是 `0x8000...`，memset(0) 会把它写成 +0.0——数值相等但位不同，为避免改变下游任何依赖位模式的代码（如 hash、bit-exact 对拍），主动拒绝 -0.0 走 memset。

**收益机制**：

- **不占 SM**：memset 走 copy engine / DMA，与相邻的 compute kernel 不争抢 SM 资源；理论上可与前一个不相关的 compute kernel **重叠执行**（同一 stream 内仍串行，但不同 engine 的 setup 开销更低）。
- **脱离 kernel launch 路径**：nsys 下 `cudaLaunchKernel` 被 CUPTI 逐调用插桩（doc 3.1.3 修正块量化为 ~2.0 μs/次），而 `cudaMemsetAsync` 是另一条 API，插桩开销更小；在非 profiling 环境下也少一次 driver-side launch 调度。
- **GPU 侧时间**：小 buffer 的 memset 通常 1–2 μs，比 FillKernel 的 3–4 μs 略低（少了 kernel setup / grid launch）；大 buffer 上 memset 走 DMA 带宽与 FillKernel 走 SM 写带宽相近，差异不大。

**未覆盖的路径**：`Scalar` 非零（如 `Fill(1.0)`、`Fill(-inf)`）、非连续 tensor（当前不存在）、dtype 不在 `kDataTypeToSize`（当前不存在）——均回退到 `FillKernel`，语义与原来完全一致。

#### 3.3.6 P1 优化结果

**编译**：`build_new` 重编 `infini_train_cuda_kernels` + `llama3` + `test_autograd_cuda` 通过，无警告。

**单元测试**：

- 全量 CUDA autograd：**263 pass / 1 pre-existing skip**（与 P0 完全一致，未引入回归）
- `test_tensor_cuda`：**28/28 pass**（含 Tensor 生命周期与基础操作，覆盖 Fill 路径）

**数值等价性**：`build_new/llama3` 10-iter × 3 次运行，step1 loss 恒为 **`4.898438`**（与 A/B/C/D 逐位一致）→ **前向 bit-exact**。step2 三次分别 `4.545731 / 4.545866 / 4.546757`，跨度 **0.001**，比 P0 的 0.0037 更窄，也远小于 doc 3.2.4 记录的 C 自变区间（≤0.0073）→ memset 路径与 FillKernel 路径位级等价，未引入任何系统偏差。

**端到端稳态时延**（`build_new/llama3`，无 profiling，10 iter × 3 次，每步取中位数再对窗口求均值）：

| 配置 | commit | 构建目录 | step5–10 稳态 | 运行间极差 |
|---|---|---|---|---|
| C ＋adam 向量化（基线） | `25c3bd8` | `build_adam` | **75.69 ms** | ±0.90% |
| D ＋P0 Fill 删除 | `a696321` | `build_new` | **75.41 ms** | ±0.72% |
| **E ＋P1 memsetAsync** | `a696321` + 未提交改动 | `build_new` | **75.18 ms** | **±0.39%** |

E 的窗口敏感性：step3–10 **75.93 ms** / step4–10 **75.17 ms** / step5–10 **75.18 ms**。

| 增量 | 稳态时延 | 降幅 | 折合每轮 |
|---|---|---|---|
| C→D（P0） | 75.69 → 75.41 ms | −0.37% | −0.28 ms |
| **D→E（P1）** | 75.41 → 75.18 ms | **−0.31%** | **−0.23 ms** |
| **C→E（P0+P1 合计）** | 75.69 → 75.18 ms | **−0.67%** | **−0.51 ms** |

> **可信度**：E 的运行间极差 **±0.39%** 是三配置里最窄的（C ±0.90%、D ±0.72%），说明 memset 路径比 kernel 路径**更稳定**——DMA engine 的调度抖动小于 SM kernel launch。D→E 的 −0.23 ms 仍落在 E 自身噪声带（±0.29 ms）边缘，但 C→E 的 −0.51 ms 已经**超出** E 的噪声带，可作为量级结论采信。
>
> **为什么 wall 收益远小于理论 GPU 收益**：P1 覆盖的 ~501 次 Fill × ~4 μs ≈ **2.0 ms** GPU 时间被转成 memset，但 wall 只降 0.23 ms。原因与 P0 相同：非 profiling 环境下 host 发射早已与 GPU 执行流水化，Fill 的 GPU 时间大部分被 launch gap 吸收；memset 虽然不占 SM，但仍占用 stream 时间，在单 stream 串行架构（doc 2.1 结论 3）下无法与 compute kernel 重叠。**真正的收益要在 nsys 下才能放大**：按 doc 3.1.3 的 CUPTI 量化（~2.0 μs/次 `cudaLaunchKernel`），P1 消除 ~501 次 kernel launch 折合约 **1.0 ms/step 的 host 侧 CUPTI 开销**，加上 GPU 侧 ~1.0 ms（memset 比 FillKernel 每次省 ~2 μs × 501），nsys 稳态窗口预计降 **~2.0 ms/step**（待复测）。
>
> **架构价值**：P1 的核心收益不在 wall，而在**把 Fill(0) 从 kernel 家族里摘出去**。这为后续优化铺路：① 档 4 CUDA Graph 捕获时，memset 节点比 kernel 节点更轻；② 多 stream 并行时，memset 走 copy engine 可与 compute stream 真正重叠；③ nsys timeline 上 `Fill·f32` 条目消失，可读性大幅提升。

**产物**：`build_new/llama3`（含 P0 + P1 改动）；未生成新 nsys trace（待后续统一复测）。
### 3.4 其他kernel级优化
#### 3.4.1 slice Kernel：元数据改走 kernel parameter space

Section 2.1 结论 6 指出 `SliceForward` 每次有 5 次微小 H2D 拷贝，其中几次因命令队列耗尽而耗时很长。每 step 约 **290 次 Slice**（forward ~145 + backward ~145），每次拷 5 个长度 = `num_dims`（典型 3–4）的 int64 数组，合计 **~1450 次 H2D + ~290 次 `cudaMallocAsync`/`cudaFreeAsync`**，正是 2.3 里「每轮 1515 次微小 HtoD」的主因。

**方案**：把 5 个元数据数组打包进定长 POD 结构 `SliceMeta`，经 kernel parameter space 按值传入，彻底消除 `cudaMallocAsync` + 5×`cudaMemcpyAsync` + `cudaFreeAsync` 整条序列。

```cpp
constexpr int kMaxDims = 8;
struct SliceMeta {
    int64_t new_dims[kMaxDims], starts[kMaxDims], steps[kMaxDims];
    int64_t in_strides[kMaxDims], out_strides[kMaxDims];
};
// SliceForwardKernel<<<...>>>(in, out, meta, num_dims, total_elements);  // meta 按值 → constant cache
```

- **为何不能用 `std::vector`**：kernel 参数须 trivially copyable，`vector` 内部是指向 host 堆的裸指针，按值传入后 GPU 解引用的是非法设备地址；定长数组是 POD，`sizeof(SliceMeta)=320 B` ≪ sm_80 的 4 KB 参数上限。
- **守卫**：host 端加 `CHECK_LE(num_dims, kMaxDims)`，防 `std::copy` 写越界。
- **附带收益**：元数据访问从 global memory（经 L2）升级为 constant cache，所有 thread 读同一地址走 zero-conflict broadcast，延迟更低。

**验证**：`test_autograd_cuda` **263 pass / 1 pre-existing skip**、`test_tensor_cuda`、`test_transformer_cuda` 全通过；E/F 共 6 次运行 step1 loss 恒为 **`4.898438`**（逐位一致）→ 前向 bit-exact。

**端到端稳态时延**（同会话内 E/F 各 10 iter × 3 次，每步取中位数再对窗口求均值，与 3.2.4 同口径；E 由暂存 slice 改动重编得到，非引用旧值）：

| 配置 | 构建目录 | step5–10 稳态 | 运行间极差 |
|---|---|---|---|
| E ＋P0+P1（基线） | `build_new`（stash slice） | **74.90 ms** | ±2.39% |
| F ＋Slice 参数化 | `build_new` | **71.83 ms** | ±0.87% |

| 增量 | 稳态时延 | 降幅 | 折合每轮 |
|---|---|---|---|
| E→F（Slice 参数化） | 74.90 → 71.83 ms | **−4.09%** | −3.07 ms |

F 的窗口敏感性：step3–10 **73.28 ms** / step4–10 **72.32 ms** / step5–10 **71.83 ms**。

> **可信度**：−3.07 ms 远超 F 自身噪声带（±0.87% ≈ ±0.62 ms），也超出 E 的噪声带（±2.39% ≈ ±1.79 ms），可作强信号采信。与 P0/P1 的 wall 收益（各 −0.2~0.5 ms）相比本次显著更大，原因是 slice 消除的不只是 kernel launch，还有 **290 次 `cudaMallocAsync`/`cudaFreeAsync`**（async mempool 分配器带锁，host 开销远高于普通 launch）与 ~1450 次 memcpy API；且 F 的噪声带收窄到 E 的约 1/3，印证结论 6 里「队列耗尽导致的长尾阻塞」被消除。

> **产物**：`build_new/llama3`（含 P0+P1+Slice 参数化）；未生成新 nsys trace（待统一复测）。该模式可推广到 `transform.cu` 的 `TransposeForward`（3 数组 × ndim 同样可并入 kernel 参数）。

### 3.5 RMSNorm Kernel 融合

Section 2.1 结论 4 与 2.2.1.2 的 timeline 给出了 RMSNorm 的开销画像：16 层 × 2 个 RMSNorm + 收尾 ln_f 共 **33 个实例**，每实例由 Pow → Mean → AddScalar → Rsqrt → Mul → Mul **六个独立 kernel** 组成（其中两次 Mul 各 ~22 μs，是单实例内的大头），forward 侧合计 **198 个 kernel/step**；backward 侧再把 Pow/Rsqrt/AddScalar 反向、GenericReduceBwd、BinaryBwd[Mul] 等展开近十个节点。本节把整个 RMSNorm 融合为单 Function：forward、backward 各一个 kernel——这也是 Section 2.3「CUDA Graph 或算子融合」方向的第一炮。

#### 3.5.1 方案：autograd::RMSNorm + 融合 kernel（对齐 LayerNorm 模式）

- **Function 层**：新增 `autograd::RMSNorm`（kType = `RMSNormFunction`）。Forward 经 `Dispatcher` 调 host wrapper `RMSNormForward`，一次返回 `{output, rstd}`；`rstd` 标记 `MarkNonDifferentiable` 并连同 `{input, weight, rstd}` 存入反向上下文。Backward 调 `RMSNormBackward` 返回 `{grad_input, grad_weight}`。
- **数学**：forward 保持与 composite **相同的两步舍入顺序**——先 `n = x·rstd` 再 `y = n·w`，不合并为 `x·(rstd·w)`（避免 FMA 结合律改变舍入行为）；backward 用 `S1 = Σ g·w·x`、`K = (S1/H)·rstd`、`dx = rstd·(g·w − n·K)`、`dw = Σ g·n`（跨 token 块 `atomicAdd`）。
- **rank-agnostic**：与 composite 路径（Mean(-1) 支持任意秩）保持同等一般性——`embed_dim = dims.back()`、`rows = NumElements()/embed_dim`、`rstd` 形状取前 n−1 维；非连续输入按惯例回退 `Contiguous()`。
- **grad_weight 清零**：权重梯度跨 token 块累加，起点必须在 launch 前清零——放 host 侧 `grad_weight->Fill(0.0)`（同 stream 先于 backward kernel，由 stream 顺序提供全局 happens-before）。**反例（已实测否决）**：kernel 内 per-block `grad_weight[blockIdx.x] = 0` + `__syncthreads()` **不能**替代——`__syncthreads` 只是块内屏障，而 `grad_weight` 的累加跨全部 block；最小复现实验（`build_new/grad_weight_clear_race.cu`）在三种规模（行数 <、=、> embed_dim）下 200/200 全部出错，其中行数 > embed_dim 时每次越界写显存（llama3 的 B·S 远大于 embed_dim，即真实场景必现）。
- **autocast**：注册 `"RMSNorm" → kFP32`（与其 fp32 残差流语义一致）；顺带修复既有大小写 bug——`kFP32Ops` 表中的 `"Layernorm"` 与 `GetBaseOpName("LayerNormFunction")` 返回的 `"LayerNorm"` 永不匹配，LayerNorm 的 kFP32 策略此前实际从未生效，本次一并改为 `"LayerNorm"`。
- **CPU fallback**：`kernels/cpu/rmsnorm.cc` 同步实现（rank-agnostic、fp32），CPU 测试实例可跑。

#### 3.5.2 优化结果

**单元测试**：`test_autograd_cuda` 全量 **270/270 通过**（含新增用例：forward/backward 数值断言、2D 输入秩无关性、autocast bf16→fp32 集成、rstd 非可微）；`test_transformer_cuda` 全绿（RMSNorm / LLaMA3Model 端到端）。

**数值等价性**：step1 loss **4.898438 → 4.896993**——融合改变了归约顺序（块内 BlockReduce vs composite 的 Mean kernel），不再 bit-exact，差 0.0014（相对 3×10⁻⁴）为预期量级；step2 起各配置的差值落在训练固有自变区间（同一二进制 3 次运行跨度 ≤0.0075）内，未引入系统偏差。

**端到端稳态时延**（同会话 A/B 配对：`git stash` 暂存融合改动 → reconfigure + 重建基线 → 10 iter × 3 次 → 恢复改动重建并经 sha256 校验；每步取中位数再对窗口均值，与 3.2.4 同口径）：

| 配置 | commit | 构建目录 | step5–10 稳态 | 运行间极差 |
|---|---|---|---|---|
| G 基线（无融合） | `4548686` | `build_new` | **70.36 ms** | ±0.33% |
| H ＋RMSNorm 融合 | 本次提交 | `build_new` | **66.41 ms** | ±0.37% |

| 增量 | 稳态时延 | 降幅 | 折合每轮 | 吞吐量 |
|---|---|---|---|---|
| G→H（RMSNorm 融合） | 70.36 → 66.41 ms | **−5.61%** | −3.95 ms | 3638 → 3855 tok/s（+6.0%） |

窗口敏感性（G→H）：step3–10 71.04 → 66.56（−6.31%）；step4–10 70.41 → 66.44（−5.64%）；warm-up step1 190.52 → 181.23 ms。

> **基线口径**：G 为当前 HEAD `4548686`，在 3.4.1 的 F 配置（71.83 ms）之上另含两项未入档的中间提交（CrossEntropy 设备端归约、grad_a Fill 删除）；本节对照全部为同会话配对，不引用历史绝对值。恢复改动后的独立冒烟复核 66.36 ms。
>
> **可信度**：−3.95 ms 约为 H 自身噪声带（±0.37% ≈ ±0.25 ms）的 16 倍，属强信号。收益大于此前估算（1–2 ms）的原因主要在 backward：composite 侧近十个节点的反向链 × 33 实例被一并消除，而估算时只计了 forward 的舍入往返。

**产物**：`build_new/run_base_{1,2,3}.log`、`run_fused_{1,2,3}.log`（配对原始日志）、`parse_wall_log.py`（解析脚本）；nsys 复测（逐 kernel 验证 launch 数下降）待后续统一进行。

### 3.6 Embedding 稀疏梯度（Sparse Row）优化

Section 2.2.2/2.2.3 记录的 embedding 画像（`EmbeddingBwd·f32` 1.3 ms、大参数 Adam 5.7 ms）在 3.2 向量化后仍是每步最大的单体浪费：`tok_embeddings` 权重 [128256, 2048] fp32 = **1.05 GB**，而每步实际命中仅 **256 个 token 行（0.2%）**。3.3.3 的判据把 `EmbeddingBwd` 的 Fill 归入「真需要零起点、保留」的档 2/3 场景——本节把「零起点」从每步整表缩小到每步只清命中行，连同反向与 Adam 一并稀疏化。

三项浪费（同会话 nsys 实测，`nsys/out/llama3_5iter_sparse_base`，bf16，每步）：① 整表 memset 1.05 GB / **680 µs**（backward 前 `Fill(0)`，3.3.5 特化后形态）；② `EmbeddingBackwardKernel` 全表 scan **1090 µs**（max 1335 µs）；③ `tok_embeddings` dense vectorized Adam 单 kernel **5.74 ms**。

#### 3.6.1 方案：持久 grad buffer + 行清单 + 稀疏 Adam

- **持久 grad buffer 不变量**：`EmbeddingBackward` 的 `grad_weight` 改为跨步常驻，维持不变量「除行清单记录的命中行外恒零」。首次 backward 一次性 `Fill(0)`（走 3.3.5 的 memset 快路径）；此后 `ZeroGrad` 只清零上一步的 **~256 行（≈2 MB）** 并递增 `generation` 使去重 stamp 失效。返回的梯度是 buffer 的**零拷贝视图**（`make_shared<Tensor>(*buffer, 0, dims)`）。
- **行去重 claim**：backward kernel 改 2D grid，`(blockIdx.y==0, threadIdx.x==0)` 的线程经 `stamp[vocab]`（int32，按 `generation` 区分批次）`atomicCAS` 争抢写入 `row_list`（capacity=vocab，天然无溢出）；命中数 `count` 常驻**设备端**，host 全程零同步（消除 2.3 类 D2H 风险）。
- **稀疏 Adam**：新增 `AdamSparseRows(Shadow)Kernel`，grid-strided 遍历 `row_list[0..*count)`，只对命中行更新 param/m/v（及 shadow），行内复刻既有 128-bit 向量化（`dim%4==0` + 对齐守卫，否则标量回退）。
- **兼容机制**（正确性兜底）：① **alias 防双计**——`accumulate.cc` 检测待累加梯度与视图同址（DataPtr 相等）时跳过；② **poison + flush**——出现非 alias 的 dense 写入（未来 `lm_head` 真 weight-tying、DDP bucket 摊平）时标记 `poisoned` 永久回退 dense Adam，同时清行保证每个 micro-batch 只贡献 delta；③ **dtype 回退**——`grad_output` 与 weight dtype 不一致时回退 dense 路径。

#### 3.6.2 正确性验证

- **单元测试**：全量 **331 项 0 失败**（autograd / optimizer / module / transformer CUDA 套件）。
- **数值等价性**（本次 3×3 配对，10 iter）：step1 `4.896993`、step2 `4.544800` 在全部 6 次运行**逐位一致**（首步更新与 dense LazyAdam 数学等价）；step3 差 4.3×10⁻⁵、step4 差 1.2×10⁻³，至 step5 起差值（3–9×10⁻³）落入**运行间固有自变带**（基线自身 step6 起三次运行分叉至 ≤6×10⁻³，新代码 step5 起分叉，量级相同）。差异来源：① LazyAdam 语义（未命中行冻结 vs dense 用陈旧动量继续更新）；② 原子累加顺序（同一二进制重跑亦抖动）。

#### 3.6.3 优化结果

**端到端稳态时延**（同会话配对：`git stash` 切基线重编，各 10 iter × 3 次，每步取中位数再对窗口均值，与 3.2.4 同口径）：

| 配置 | commit | 构建目录 | step5–10 稳态 | 运行间极差 |
|---|---|---|---|---|
| I 基线（HEAD，含 3.1–3.5 全部优化） | `195b0fb` | `build_new`（stash 本次改动） | **66.46 ms** | ±0.49% |
| J ＋Embedding 稀疏梯度 | 未提交改动 | `build_new` | **59.27 ms** | ±0.91% |

| 增量 | 稳态时延 | 降幅 | 折合每轮 | 吞吐量 |
|---|---|---|---|---|
| I→J（Embedding 稀疏梯度） | 66.46 → 59.27 ms | **−10.82%** | −7.19 ms | 3859 → 4342 tok/s（+12.5%） |

窗口敏感性（I→J）：step3–10 67.51 → 60.76（−10.0%）；step4–10 66.63 → 59.93（−10.1%）；warm-up step1 186.5 → 198.6 ms（含首次持久 buffer 分配与 kernel 首启，运行间极差较大）。

> **可信度**：−7.19 ms ≈ J 自身噪声带（±0.91% ≈ ±0.54 ms）的 13 倍，属强信号；基线 10 步内始终维持 66.26–66.72 ms 平台（无时钟漂移）。J 的 step2–3（66.2–66.6 ms）恰等于基线平台、step4 起进入 59 ms 稳态——与 3.4.1/3.5.2 一致的「收益从上一步末尾开始兑现」流水填充现象。

**GPU 侧 kernel 账**（同会话 nsys 对照，`llama3_5iter_sparse_base` vs `llama3_5iter_sparse`，均 5 iter bf16）：

| 项目（每步） | 基线 | 新代码 | Δ |
|---|---|---|---|
| 1.05 GB memset | **5 次**（678–683 µs/次） | **0 次**（仅首步 1 次初始化） | −680 µs |
| embedding backward | `EmbeddingBackwardKernel` 1090 µs avg（max 1335 µs） | `EmbeddingBackwardSparseKernel` **7.1 µs avg**（max 11.1 µs） | −1083 µs |
| 大参数 Adam | vectorized 346 次 Σ137.2 ms，含 **8 个 5.74 ms 大 call**（tok_emb + lm_head 各 1/步） | vectorized 343 次 Σ114.5 ms，大 call 仅 **4 个（lm_head）**；tok_emb 改走 `AdamSparseRowsShadow` **14 µs/步** | −5727 µs |
| 新增清行 | — | `SparseRowClearRows` 3.6 µs + `SparseRowResetCount` 1.5 µs | +5 µs |
| **合计** | — | — | **≈ −7.48 ms** |

> **分解核对**：两次采集小 kernel 部分几乎重合（Σ 91.3 vs 91.6 ms），vectorized Adam 总量差 22.7 ms 与「大 call 少 4 个 × 5.74 ms」吻合（两次采集截断深度不同，大 call 计数 8 vs 4 同源）。nsys 口径 −7.48 ms 与 wall −7.19 ms 一致。
>
> **显存**：峰值 used 25999 → 26008 MB（stamp 0.5 MB + row_list 0.5 MB 常驻；1.05 GB buffer 基线本就每步分配同尺寸，峰值口径不变）。

#### 3.6.4 边界与后续

- **残留 dense Adam 5.74 ms/步是 `lm_head`**：当前 `tok_embeddings` 与 `lm_head` 为两个独立 fp32 参数（`transformer.cc` 的 weight-tying TODO 未实现），且 `lm_head` 梯度经 softmax 本质稠密（全词表），不在本优化射程；未来实现真 tying 后 poison 机制自动保证正确性。
- **LazyAdam 语义为有意取舍**：未命中行完全冻结（保留 m/v），与 dense 「陈旧动量继续更新」不同；如需严格对齐可加低频 dense 兜底。

**产物**：`build_new/run_bsl_{1,2,3}.log`、`run_sparse_{1,2,3}.log`（10 iter 配对原始日志）；nsys `nsys/out/llama3_5iter_sparse_base.*`（基线对照）、`nsys/out/llama3_5iter_sparse.*`（本次改动，stats 输出见 `build_new/run_llama3_5iter_sparse_nsys.log`）；查询脚本 `build_new/query3_nsys.py`、`query_memset.py`；实现为本轮 7 个改动文件 + 新增 `infini_train/include/sparse_row_grad.h`。

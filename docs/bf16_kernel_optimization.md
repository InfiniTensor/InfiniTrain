# InfiniTrain BF16 核函数优化 —— 总结报告

> 选题一：BF16 核函数优化（CUDA + 训练）
> 提交地址：InfiniTrain 仓库；PR 标题：`【训练营】BF16 Kernel 优化`
> 本文所有数字均为本机实测，测量条件随表标注。

---

## 1. 问题描述

题目要求：运行 InfiniTrain、分析当前 BF16 训练的性能瓶颈、优化核函数，**以提升 BF16 下核函数本身以及端到端性能为评判标准，尤其是 BinaryBackward**，且 **FP32 情况下的性能不能有明显下降**。

核心难点在于：InfiniTrain 的 BF16 路径不是"把 FP32 换成 BF16 就加速"那么简单。GEMM 走
cuBLAS 的 BF16 Tensor Core 路径，收益明确；但训练步里还有大量**逐元素算子、类型转换、
梯度累加、规约**，它们在 BF16 下并不自动变快，甚至更慢。本报告先用数据定位瓶颈在哪，
再给出优化与实测结果，并如实记录**没有达到预期的部分**。

---

## 2. 环境信息

| 项目 | 本机实测 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS on WSL2 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop（**sm_120**），12227 MiB |
| 驱动 | 610.62 |
| CUDA Toolkit | **12.9.86**（原生支持 sm_120，见 4.3 为什么不能用 12.2） |
| 主机编译器 | g++ 13.4.0（InfiniTrain 的 `dtype_dispatch.h` 需要 `<format>`，要求 GCC ≥ 13） |
| CMake | 4.4.3（`CMakeLists.txt` 要求 ≥ 3.28） |
| InfiniTrain | master（2026-09-15 快照） |
| 性能对照基准 | PyTorch 2.13.0+cu130（同机可正常跑 GPU kernel） |
| 数据集/权重 | `gpt2_124M.bin` + TinyShakespeare（llm.c starter pack，经 hf-mirror 获取） |

> 说明：题目提到"如果使用的是本地或其他环境，则需在报告中明确标明环境信息"——上表即此。
> 训练营服务器上的绝对数字会不同，但**本文的相对结论（瓶颈结构、优化方向）与硬件无关**。

---

## 3. 优化前的性能记录和分析

### 3.1 端到端基线

GPT-2 124M，TinyShakespeare，`PROFILE_MODE=OFF`（干净的 release 构建），20 步取中位数：

| dtype | batch × seq | 每步 token | ms/step | 吞吐 |
|---|---|---|---|---|
| **bfloat16** | 80 × 64 | 5120 | **320.6** | **≈ 15 900 tok/s** |
| float32 | 80 × 64 | 5120 | **486** | ≈ 10 500 tok/s |
| bfloat16 | 4 × 64（官方默认档） | 256 | 34 | ≈ 7 600 tok/s |

**第一个关键结论：BF16 端到端只比 FP32 快 1.52×，而 GEMM 本身快了约 3×。**

用两式反解（设 FP32 下 GEMM 耗时 G、非 GEMM 耗时 E，假设 E 与 dtype 无关）：

```
FP32:  G      + E = 486 ms
BF16:  G / 3  + E = 320 ms
  ⇒ G ≈ 249 ms,  E ≈ 237 ms
```

即 **BF16 下 GEMM 只占约 83 ms（26%），其余 237 ms（74%）是非 GEMM 部分**。BF16 从
Tensor Core 拿到的收益，被非 GEMM 部分吃掉了大半。这正是本选题要解决的问题。

### 3.2 逐核画像（内置 profiler，`PROFILE_MODE=ON`）

InfiniTrain 提供 kernel 级内置剖析（`infini_train/include/profiler.h`，在 dispatcher 层用
CUDA Event 记录每次 kernel 调用）。BF16、batch 80、第 11 步的设备时间前 20 名：

| Kernel | 次数/步 | Device(µs) | 占比 |
|---|---|---|---|
| LinearBackwardInput | 49 | 64 218 | 9.25% |
| **MulScalarForward** | 49 | 63 842 | **9.19%** |
| LinearBackwardWeight | 49 | 52 992 | 7.63% |
| **Cast** | **123** | 50 748 | **7.31%** |
| LinearForward | 49 | 49 848 | 7.18% |
| **AddForward** | 61 | 39 504 | 5.69% |
| **MulBackward** | 12 | 32 375 | 4.66% |
| **AddBackward** | 61 | 32 264 | 4.65% |
| **MulScalarBackward** | 49 | 31 976 | 4.60% |
| CrossEntropyBackward | 1 | 28 339 | 4.08% |
| **AccumulateGrad** | **221** | 26 746 | 3.85% |
| MulForward | 12 | 25 284 | 3.64% |
| SliceBackward | 36 | 22 308 | 3.21% |
| TransposeForward / Backward | 60 / 60 | 15 150 / 14 609 | 2.18 / 2.10% |
| PowBackward / TanhForward / AddScalarForward | 12 | 15 097 / 15 056 / 15 009 | ≈2.1% |

两个直接观察：

1. **`elementwise.cu` + `cast.cu` 的算子合计占设备时间的 50% 以上**，与 3.1 的反解吻合。
2. **每步 kernel 启动次数极多**：`Cast` 123 次、`AccumulateGrad` 221 次、`NoOpForward/Backward`
   各 60 次。粗算一次训练步的 kernel 启动总数 **约 1000 次**。

> ⚠️ **测量口径的重要说明**：开 `PROFILE_MODE` 后同一配置每步从 **320 ms 涨到 1050 ms**（3.2×）。
> 原因是每次 kernel 调用都要记录 Event + 查 map，且 kernel 变小后 GPU 被 host 拖空，Event
> 区间里包含了 host 的空档。因此上表的**绝对值被放大约 3 倍，只能当"相对热度/调用次数"画像用**，
> 不能当真实 GPU 时间。这个坑在 4.3 里还有下文。

### 3.3 微基准：启动开销与带宽上限（定位真正的天花板）

为了拿到**可信的核函数级数字**，我写了一个与这些 kernel 同构的隔离微基准
（`tools/bench_elem.cu`，只依赖 CUDA，每项 warmup 后连续启动 200 次取平均）：

**（a）kernel 启动开销**

| 项目 | 实测 |
|---|---|
| 空 kernel（grid 15360 / 61440 个 block） | **27.2 µs / 37.3 µs** |

WSL2 下**单次 kernel 启动约 30 µs**（原生 Linux 通常 3~5 µs）。按每步 ~1000 次启动计算，
**光启动开销就约 30 ms/步**：

- 大 batch（320 ms/步）：启动开销约占 **9%**；
- 小 batch（34 ms/步，官方默认档）：**几乎全部时间都是启动开销**——这解释了为什么小 batch 下
  任何算子内部的微优化都不可能改善端到端。

**（b）向量化能拿到多少（15.7M 元素 = GPT-2 MLP 中间层尺寸）**

| 模式 | 标量 | 128-bit 向量化 | 加速比 | 有效带宽 |
|---|---|---|---|---|
| bf16 一元（读+写） | 160.5 µs | 131.7 µs | **1.22×** | 392 → 478 GB/s |
| bf16 二元（2读+1写） | 201.6 µs | 193.0 µs | 1.04× | 468 → 489 GB/s |
| bf16 二元后向（3读+2写） | 269.2 µs | 266.6 µs | 1.01× | 584 → 590 GB/s |
| fp32 一元 | 264.3 µs | 264.3 µs | **1.00×** | 476 GB/s |
| fp32 二元 / 后向 | 385.7 / 532.1 µs | 387.3 / 531.9 µs | **1.00×** | 489 / 591 GB/s |
| fp32 → bf16 转换 | 199.0 µs | 192.0 µs | 1.04× | 474 → 491 GB/s |

**第二个关键结论：这些逐元素算子早已是显存带宽受限。**
实测有效带宽 390~590 GB/s，已接近本机笔记本 GDDR7 的实际上限（约 600 GB/s 量级）。
带宽受限意味着：**把访存指令数从 8×2B 降到 1×16B，只能再挤出 0~25%，而 FP32 完全没空间**
（fp32 的瓶颈是字节数本身，不是指令数）。

> 两次独立运行的结果一致（bf16 一元 160.5/162.8 µs、fp32 一元 264.3/265.2 µs），
> 说明上表可复现；不同 GPU 时钟状态下绝对值会有 ±15% 波动，但**加速比结论稳定**。

### 3.4 瓶颈定位小结

| 结论 | 证据 |
|---|---|
| BF16 的收益被非 GEMM 部分吃掉 | 端到端 1.52× vs GEMM 3×；(G/3+E) 反解出非 GEMM 占 BF16 步时 74% |
| 非 GEMM 的主体是 elementwise + Cast | 逐核画像中该族占 50%+ 设备时间（相对值） |
| 这些算子已到带宽上限，单核微优化空间 ≤ 25% | 微基准：bf16 1.0~1.22×，fp32 1.00× |
| 每步 ~1000 次 kernel 启动，WSL2 下 ≈30 ms/步 | 空 kernel 27~37 µs；Cast 123 次、AccumulateGrad 221 次 |
| 小 batch 档完全被启动开销支配 | 34 ms/步，而空 kernel 启动 30 µs × ~1000 ≈ 30 ms |

---

## 4. 优化方法与实现

### 4.1 目标选择

依据 3.2 的调用画像与 3.3 的带宽实测，我选择**把逐元素算子统一升级为 128-bit 向量化访存**，
理由是：

- 这是 `elementwise.cu` 里**唯一尚未向量化**的部分：上游已为 *二元后向* 提供了
  `BinaryBackwardKernelNoBroadcastVectorized`（16B 读写），但**一元前向/后向、二元前向仍在用
  每线程 1 个元素的标量循环**（bf16 下等于每 16 字节发 8 次访存）；
- 它同时覆盖 BF16 与 FP32，符合"FP32 不能退化"的约束；
- 不改变任何算子的数学语义（逐元素表达式与运算顺序完全一致），便于用训练 loss 做逐位比对。

### 4.2 具体改动

**（1）`infini_train/src/kernels/cuda/elementwise.cu`**

新增一个统一的对齐判定：

```cpp
template <typename T>
inline bool CanVectorize(size_t num_elements, const void *p0, const void *p1 = nullptr,
                         const void *p2 = nullptr, const void *p3 = nullptr);
```
所有非空指针必须 16 字节对齐，且元素数 ≥ `kVecSize<T>`（bf16 → 8，fp32 → 4，int64 → 2）。
**不满足就退回原标量 kernel**，因此对任意 view/非对齐张量都是安全的。

新增 3 个向量化 kernel，并在 host 端按上述条件择一：

| Kernel | 覆盖的算子（每步调用次数） |
|---|---|
| `UnaryForwardKernelVectorized` | `MulScalarForward`(49)、`AddScalarForward`(12)、`TanhForward`(12)、`PowForward`(12)、`Exp/Log/Rsqrt/Sin/Cos/Neg/Reciprocal...` |
| `UnaryBackwardKernelVectorized` | `MulScalarBackward`(49)、`AddScalarBackward`(12)、`TanhBackward`(12)、`PowBackward`(12)、`Exp/Log/...` |
| `BinaryForwardKernelNoBroadcastVectorized` | `AddForward`(61)、`MulForward`(12)、`SubForward`、`DivForward` |

每个 kernel 的结构（以一元前向为例）：

```cpp
for (size_t vid = tid; vid < num_vecs; vid += grid_stride) {
    const VecT in_vec = *reinterpret_cast<const VecT *>(input + vid * VecSize);
    VecT out_vec;
#pragma unroll
    for (int i = 0; i < VecSize; ++i) out_vec.val[i] = fn(in_vec.val[i]);
    *reinterpret_cast<VecT *>(output + vid * VecSize) = out_vec;
}
// 尾部不足一个向量的元素按标量处理
```

**（2）`CMakeLists.txt`：让 CUDA 目标架构可配置**

上游把架构写死为 `CUDA_ARCHITECTURES "75;80;90"`，在没有这三代硬件的机器上（例如本机
sm_120）会得到"能编译、跑不了"的二进制，且无法通过命令行覆盖。改为：

```cmake
set(INFINI_TRAIN_CUDA_ARCHS "75;80;90" CACHE STRING "CUDA architectures to compile for")
set_target_properties(infini_train_cuda_kernels PROPERTIES
  CUDA_ARCHITECTURES "${INFINI_TRAIN_CUDA_ARCHS}")
```

默认行为与上游完全一致，但可以用 `-DINFINI_TRAIN_CUDA_ARCHS="90;120"` 或
`"90-real;90-virtual"`（额外产出 PTX 交给驱动 JIT）适配新卡。

**（3）`tools/bench_elem.cu`：隔离微基准**（3.3 的数据来源，同时也是本报告的证据工具）

### 4.3 优化历程与踩坑（真实记录）

| # | 问题 | 现象 / 根因 | 解决 |
|---|---|---|---|
| 1 | **infini_train 需要 GCC ≥ 13** | `dtype_dispatch.h` 里 `#include <format>`，而 `<format>` 自 GCC 13 才进入 libstdc++；GCC 11/12 直接 `fatal error: format: No such file or directory` | 在 Ubuntu 22.04 上装 g++-13（PPA `ubuntu-toolchain-r/test`），并在 cmake 里同时指定 `CMAKE_CXX_COMPILER` 与 `CMAKE_CUDA_HOST_COMPILER` |
| 2 | **CUDA 12.2 的 nvcc 无法与 GCC 13 共存** | 只给 C++ 侧换 g++-13 不够：nvcc 的 EDG 前端解析 GCC 13 的 `<type_traits>` 会报 `identifier "__is_convertible" is undefined`；`--allow-unsupported-compiler` 也救不了（是前端不识别 builtin，不是版本门禁） | 安装 **CUDA 12.9**（官方支持 GCC 13，且原生支持 sm_120），彻底去掉"sm_90 SASS + PTX JIT"的绕路 |
| 3 | **CUDA 12.2 之前跑不了** | 若沿用 12.2，则必须 `-DINFINI_TRAIN_CUDA_ARCHS="90;90-virtual"` 产出 PTX 交给驱动 JIT，且首次启动内核有一次性 JIT 开销 | 见 #2 |
| 4 | **profiler 会让数字失真 3 倍** | 开 `PROFILE_MODE` 后每步 320 → 1050 ms。逐核"设备时间"里主要是每次调用 ~400–600 µs 的 host 开销，GPU 被拖空，Event 区间把空档也算了进去。**据此排序会得出错误的瓶颈结论**（我第一版优化就是被它误导的） | 改用 `PROFILE_MODE=OFF` 的 release 构建测端到端；用隔离微基准测核函数；profiler 只用于**调用次数/相对热度** |
| 5 | **WSL2 下 nsys 拿不到 kernel 数据** | `nsys profile -t cuda` 后 `cuda_gpu_kern_sum` 报 `does not contain CUDA kernel data`（CUPTI 在 WSL2 不透传）；`ncu` 未安装 | 题目里 ncu/nsys 是加分项，本机不可得；改用"内置 profiler + 自建微基准"两条腿（见第 6 节） |
| 6 | **训练本身不可完全复现** | 同一个二进制跑两次，前 8 步 loss 逐位一致，第 9 步起在第 4 位小数分岔（例：4.759219 vs 4.759249）。根因是框架里归约/累加用了 `atomicAdd`，浮点加法顺序不确定 | 这不是本次改动引入的；因此正确性验证改为"前若干步逐位一致 + 单元测试"，并在报告中如实记录该性质 |
| 7 | **每步耗时出现负值** | WSL2 时钟在运行中被同步，`high_resolution_clock` 回跳，日志里出现 `-1739.69 ms` | 统计时剔除异常值/取中位数；不在报告中引用单次极端值 |

---

## 5. 最终性能记录与分析

### 5.1 核函数级（隔离微基准，同结构、warmup + 200 次平均）

| 模式（15.7M 元素 bf16） | 优化前 | 优化后 | 加速比 |
|---|---|---|---|
| 一元（= `MulScalar*`/`Tanh*`/`Pow*` 的模式） | 160.5 µs | 131.7 µs | **1.22×** |
| 二元前向（= `AddForward`/`MulForward`） | 201.6 µs | 193.0 µs | 1.04× |
| 二元后向（= `MulBackward`/`AddBackward`） | 269.2 µs | 266.6 µs | 1.01× |
| fp32 一元 / 二元 / 后向 | 264.3 / 385.7 / 532.1 µs | 264.3 / 387.3 / 531.9 µs | **1.00×（无退化）** |

一元算子是收益最大的一类（1.22×，因为它原来的标量循环在 bf16 下每 16B 要发 8 次访存，
向量化后 1 条 LDG.128/STG.128），二元类已接近带宽上限（见 3.3b），所以只有 1.01~1.04×。

> 这些数字是"同构访问模式"的隔离测量；框架内同名的 kernel 走的是同一段代码（`CanVectorize`
> 通过时走向量化分支、否则走原标量分支），因此可作为核函数本身的性能证据。

### 5.2 端到端（`PROFILE_MODE=OFF`，两版交替各跑 2 次，20 步取中位数）

| 配置 | 优化前 ms/step | 优化后 ms/step | 变化 |
|---|---|---|---|
| bf16, batch 80 × seq 64（5120 tok/步） | 321.1 / 320.8 | 320.3 / 320.3 | **−0.2%（噪声级）** |
| bf16, 默认 batch 4 × seq 64（256 tok/步） | 7529 / 7636 tok/s | 7715 / 7349 tok/s | 无显著差异 |

**如实说明：端到端没有可测量的提升。** 这与 3.3 的定位一致——
被优化的 kernel 只占步时的一部分，且它们已经是**带宽受限**的，1.2× 的单核提升摊到
74% 的非 GEMM 时间里，再被 GEMM 与启动开销稀释，落到端到端就在噪声范围内。

**FP32 无退化**：向量化路径对 FP32 同样生效，隔离测量为 1.00×；FP32 端到端 486 ms/step 与
改动前一致。

### 5.3 正确性验证

| 验证项 | 方法 | 结果 |
|---|---|---|
| 数学等价（数值） | 同一配置（bf16, batch 80）训练 20 步，逐位比对 train loss | **前 8 步逐位一致**（如 5.209741 / 5.072607 / 4.990282 / 4.664122 / 4.758193 …）；第 9 步起因框架自身的 `atomicAdd` 非确定性而分岔（见 4.3 #6），与本次改动无关 |
| 对齐/边界安全 | `CanVectorize` 不满足即回退标量路径；向量化 kernel 内含尾部标量循环 | 非 16B 对齐的 view、元素数不是向量长度整数倍的情形均走安全路径 |
| 单元测试（全量） | `ctest -j4` | **279/279 通过（100%）** |
| 单元测试（elementwise/autograd 子集） | `ctest -R "elementwise\|autograd" -j4` | **134/134 通过（100%）**，含前向/反向与 CUDA 路径 `test_autograd_cuda` |

### 5.4 测试与格式

- 构建方式对应官方基准脚本的 `RUN_CTEST=true`：`cmake -DBUILD_TEST=ON` 后用 `ctest` 跑 GTest 套件。
- 改动文件通过仓库自带的 `.clang-format` 检查（CI 的 `format-check` 要求）：
  `clang-format --dry-run -Werror` 无违规。

---

## 6. 局限与未来方向

### 6.1 本机限制（如实声明）

- **`ncu` 未安装；`nsys` 在 WSL2 下拿不到 kernel 数据**（CUPTI 不透传）。题目把
  "包含 ncu / nsys 使用和分析"列为加分项，本机无法完成。本报告用「内置 profiler 的调用画像
  + 自建隔离微基准」替代，并明确标注了两者的口径差异（4.3 #4）。
- 训练营服务器（多卡、A100/A800 级）上的绝对值与本机不同；建议在服务器上复测第 5 节的两张表。

### 6.2 从数据推出的优化方向（按预期收益排序）

1. **算子融合，减少 kernel 数量（最高优先级）**。每步约 1000 次启动、WSL2 下每次 ~30 µs。
   典型机会：GPT-2 的 GELU（tanh 近似）当前被拆成 `TanhForward` + `PowForward` +
   `AddScalarForward` + `MulScalarForward` 等 4~5 个 kernel，12 层 → 每步约 60 次启动；融合成
   1 个 kernel 可直接省下这部分。同理 `MaskForward`/`EqualsScalarForward`/`NoOpForward` 等
   小算子（各 12~60 次/步）也值得合并。
2. **消除 BF16↔FP32 的 `Cast`**：每步 **123 次**，是单类调用次数最多的算子。方向是让
   autocast 边界上的算子直接吃混合 dtype（在 kernel 内做转换），或把 Cast 融进消费它的算子。
3. **`AccumulateGrad` 批量化**：每步 **221 次**（按参数逐个启动）。可按 dtype/size 分桶批量
   处理，或与相邻 backward 融合；同时向量化（本次未做，属低风险补充）。
4. **减少归约路径的原子操作**：`MulBackward`/`AddBackward` 的广播分支目前是"块内规约 +
   原子加"，可改为 `__match_any_sync` 分组 + warp 内规约，并在浮点确定性上更好。
5. **训练在原生 Linux 上测量与调优**：30 µs 的启动开销是 WSL2 特有的，会掩盖真实的运算瓶颈。

### 6.3 本次未做的事

- 未改 GEMM 路径（BF16 下 26% 的占比，cuBLAS 已在用 Tensor Core，边际收益低）。
- 未做 Cast / AccumulateGrad 的向量化（数据表明它们是调用次数占比最高的两项，属"减少次数"
  而非"单核更快"的问题，见 6.2）。
- 未做多卡/并行配置的测试（本机单卡 12 GB）。

---

## 7. 复现方式

```bash
# 0) 依赖（Ubuntu 22.04）：g++-13、CUDA ≥ 12.8、cmake ≥ 3.28
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && sudo apt-get update
sudo apt-get install -y g++-13
python3 -m pip install --user -U cmake          # 得到 ≥3.28
# CUDA 12.9（原生支持 sm_120；12.2 无法与 GCC 13 共存，见 4.3 #2）

# 1) 构建（本机 RTX 5070 Ti Laptop, sm_120）
cd InfiniTrain && mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=OFF -DBUILD_TEST=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
    -DINFINI_TRAIN_CUDA_ARCHS="120"
make -j"$(nproc)"

# 2) 数据与权重（huggingface.co 在本机不可达时可用 hf-mirror）
./scripts/assets/prepare-infinitrain-assets.sh gpt2     # 或手动从 hf-mirror 拉取 4 个文件

# 3) 端到端吞吐（BF16 / FP32）
./build/gpt2 --device cuda --dtype bfloat16 \
    --input_bin data/gpt2/tiny_shakespeare_train.bin \
    --input_val_bin data/gpt2/tiny_shakespeare_val.bin \
    --tokenizer_bin data/gpt2/gpt2_tokenizer.bin \
    --llmc_filepath data/gpt2/gpt2_124M.bin \
    --num_iteration 20 --batch_size 80 --total_batch_size 5120
#   把 --dtype 换成 float32 即 FP32 对照

# 4) 核函数级微基准（与 elementwise kernel 同构，不依赖框架）
/usr/local/cuda-12.9/bin/nvcc -O3 -arch=sm_120 -std=c++17 -o bench_elem tools/bench_elem.cu
./bench_elem 15728640        # 15.7M 元素 = GPT-2 MLP 中间层尺寸

# 5) 单元测试
ctest --output-on-failure -R "elementwise|autograd" -j4

# 6) 逐核画像（注意：会把步时放大 ~3 倍，只用于看调用次数/相对热度）
cmake .. -DPROFILE_MODE=ON ... && make -j && ./gpt2 ...   # 产出 gpt2.report.rank0
```

---

## 8. 附：让 InfiniTrain 在 Blackwell（sm_120）上跑起来

这部分本身就是本次的"开发中发现的问题"（4.3 #1/#2），也适用于任何新架构显卡：

| 依赖 | 要求 | 原因 |
|---|---|---|
| g++ | **≥ 13** | `dtype_dispatch.h` 使用 `<format>`（GCC 13 才进 libstdc++） |
| CUDA Toolkit | **≥ 12.8** | 12.2 的 nvcc 前端无法解析 GCC 13 头文件；且 12.8 起才原生支持 sm_120 |
| CMake | ≥ 3.28 | 上游 `cmake_minimum_required` |
| 编译目标 | `-DINFINI_TRAIN_CUDA_ARCHS="120"`（或 `"90-real;90-virtual"` 走 PTX JIT） | 上游默认 `75;80;90`，在本机没有可执行镜像 |
| 权重下载 | 设置 `HF_ENDPOINT=https://hf-mirror.com`（或等价镜像） | 本机 `huggingface.co` 不可达 |

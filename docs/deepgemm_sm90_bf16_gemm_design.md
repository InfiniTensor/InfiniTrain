# DeepGEMM `sm90_bf16_gemm` Kernel Detailed Design

面向 Hopper（SM90，H100/H800）的 BF16 GEMM 内核详细设计。本文覆盖：warp specialization 与 pipeline 组织、多级 tiling、SMEM 排布、**寄存器累加器**分片、warpgroup MMA（`wgmma.mma_async`）调用方式、生产者-消费者 mbarrier 同步协议、TMA 指令（含 SM90 multicast）的发射与完成语义，以及若干容易被忽略但对正确性/性能关键的设计点。

> 本文与 [deepgemm_sm100_bf16_gemm_design.md](./deepgemm_sm100_bf16_gemm_design.md) 是姊妹篇。两者共享同一套 host 启发式框架、同一个 persistent 调度器、同一份 TMA 封装，但 **device 侧的流水组织因架构差异而完全不同**：SM90 用 warpgroup MMA + 寄存器累加器，SM100 用单线程 UMMA + Tensor Memory。凡涉及两代差异处，本文都会显式对照，并在 §1.4 给出一张总表。

## 0. Code Index

| 层次 | 文件 | 职责 |
| --- | --- | --- |
| Device 主体 | [sm90_bf16_gemm.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh) | kernel 本体：SMEM 布局、TMA/math 两类 warpgroup、流水推进、epilogue |
| WGMMA 描述符 | [mma/sm90.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/mma/sm90.cuh) | `BF16MMASelector`、`GmmaDescriptor` 构造、SBO/LBO 推导、K 方向描述符推进 |
| wgmma PTX | [ptx/wgmma.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/wgmma.cuh) | `wgmma.fence/commit_group/wait_group`、累加器操作数 fence |
| TMA load | [common/tma_copy.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/tma_copy.cuh) | swizzle-atom 循环、1CTA / SM90-multicast / SM100-2SM 分支 |
| LD/ST & STSM | [ptx/ld_st.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/ld_st.cuh) | `SM90_U32x2_STSM_N`、`st_shared`、`mapa_shared` |
| 调度器 | [scheduler/gemm.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh) | persistent 块分配、L2 swizzle、SM90 multicast 合法性、grouped/batched 索引 |
| Epilogue transform | [epilogue/transform.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/epilogue/transform.cuh) | `EpilogueIdentity` / `EpilogueHeadSplits` 的 N 索引映射 |
| 通用工具 | [common/utils.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/utils.cuh)<br>[common/math.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/math.cuh) | `PatternVisitor`、编译期循环展开、`ceil_div`/`align`、`cast_into_bf16_and_pack` |
| Cluster 同步 | [comm/barrier.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh) | `cluster_sync_with_relaxed_arrive` |
| Host JIT | [impls/sm90_bf16_gemm.hpp](../third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) | 模板实参拼装、TMA descriptor 构造、launch（5 个入口 API） |
| Host 启发式 | [heuristics/sm90.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/sm90.hpp)<br>[heuristics/common.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/common.hpp)<br>[heuristics/utils.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/utils.hpp)<br>[heuristics/config.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/config.hpp) | BLOCK_M/N/K、cluster、swizzle、stage 数、线程数推导、L1/L2 周期打分 |
| TMA desc 构造 | [impls/runtime_utils.hpp](../third_party/DeepGEMM/csrc/jit_kernels/impls/runtime_utils.hpp) | `make_tma_{a,b,cd,3d}_desc`、`get_compiled_dim`、swizzle→tensormap 映射 |

> **注**：本仓库的 `third_party/DeepGEMM/third-party/cutlass` 子模块为空目录，CUTLASS/CuTe 头文件未随仓库落地。文中涉及 `cute::GmmaDescriptor` 的精确位域、`cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS::fma` 展开出的 `wgmma.mma_async` PTX 文本、`cutlass::arch::warpgroup_reg_{alloc,dealloc}` 的 `setmaxnreg` 指令、`SM90_TMA_LOAD_MULTICAST_2D` 的 `cp.async.bulk.tensor` 修饰串等，均标注为「依据 CUTLASS 约定 / 由本仓库调用方式反推」，不做逐字断言。

---

## 1. 设计总览

### 1.1 一句话概括

一个 **persistent + warp-specialized + TMA 全异步、但 MMA 半同步** 的两段流水线内核。数据通路为：

```
GMEM ──(cp.async.bulk.tensor / TMA，可 multicast)──► SMEM(A,B ring)
                                                       │
                                                       ▼  wgmma.mma_async.sync.aligned.m64nNk16（整 warpgroup 协同）
                                              RF(累加器 accum[]，分散在 128 线程)
                                                       │
                                                       ▼  stmatrix.x2.b16 (bf16) / st.shared.v2.f32 (fp32)
                                                 SMEM(C/D 单缓冲)
                                                       │
                                                       ▼  cp.async.bulk.tensor store / reduce.add
                                                     GMEM(D)
```

与 SM100 最大的不同在于中段：**SM90 没有 Tensor Memory，累加器从头到尾住在 math warpgroup 128 个线程的寄存器里**。这带来两个连锁后果：

1. `wgmma.mma_async` 是 **warpgroup 级协同指令**，128 个线程必须全部参与发射（不是 SM100 的单 lane 控制指令）——因为每个线程都要「拿着」自己那一份累加器寄存器。
2. **发射 MMA 的 warpgroup、持有累加器的 warpgroup、做 epilogue 的 warpgroup 是同一批线程**。没有 SM100 那种「MMA warp / epilogue warp 分离 + TMEM 双缓冲」的重叠，累加器的生命周期把一个输出块的 compute 与 epilogue 串在一起。

A/B 的 GMEM→SMEM 搬运仍然全异步（TMA + mbarrier），这一段与 SM100 同构，可以超前 MMA 多达 `kNumStages` 个 k_block。

### 1.2 框图（单个 CTA 内）

以典型配置 `BLOCK_M=128`（⇒ `kNumMathThreads=256`，2 个 math warpgroup）为例，共 384 线程 / 12 warp：

```
                ┌──────────────────── kNumTMAThreads(128) + kNumMathThreads(256) = 384 threads ───────────────────┐
                │                                                                                                 │
  math warp-group 0 (warp0..3, 128 thr)      math warp-group 1 (warp4..7, 128 thr)      TMA warp-group (warp8..11) │
 ┌───────────────────────────────────┐    ┌───────────────────────────────────┐    ┌──────────────────────────────┐│
 │ WGMMA + EPILOGUE                   │    │ WGMMA + EPILOGUE                   │    │ w8: prefetch tensormap (1 lane)││
 │  rows 0..63   (math_wg_idx=0)      │    │  rows 64..127 (math_wg_idx=1)      │    │ w9: init barriers   (1 lane) ││
 │  accum[64] in RF                   │    │  accum[64] in RF                   │    │ w10: TMA LOAD k-loop(1 lane) ││
 │  reg_alloc<224>                    │    │  reg_alloc<224>                    │    │ w11: 空转                    ││
 └──────┬──────────────────▲──────────┘    └──────┬──────────────────▲──────────┘    │ reg_dealloc<48>              ││
        │ wait full[s]      │ wait full[s]         │                  │              └──────┬──────────────▲────────┘│
        ▼                   │                      ▼                  │              wait    │ empty[s]     │          │
   ┌────────────────────────┴──────────────────────────────────────────┴──────────────────── ▼ ─────────────┴───────┐  │
   │                          SMEM:  D(单缓冲) │ A ring(kNumStages) │ B ring(kNumStages) │ full/empty barriers        │  │
   └───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
                └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **math warpgroup**（`warp_idx < kNumMathThreads/32`）：既发 `wgmma`、又持有累加器、又做 epilogue（STSM/st.shared → SMEM → TMA store）。`BLOCK_M ≤ 64` 时只有 1 个 math warpgroup（128 线程），`BLOCK_M > 64` 时有 2 个（256 线程），每个 warpgroup 负责 `WGMMA::M = 64` 行。
- **TMA warpgroup**（`warp_idx ≥ kNumMathThreads/32`，恒 128 线程 = 4 warp）：只有第三个 warp（w10）的单个 lane 真正发 TMA；w8  prefetch descriptor、w9 初始化 barrier、w11 空转。选第三个 warp 的原因见 §3.1 注释——`BLOCK_M==32` 时 warp0/1 可能正忙于 WGMMA。

注意 w10 发 TMA 用的是 **单个 elected lane**（`cute::elect_one_sync()`），这与 SM100 一致：TMA 本身是单线程指令。真正与 SM100 不同的是 **MMA 侧**：SM90 的 math warpgroup 是 128 个线程一起发 `wgmma`，而 SM100 是 1 个 lane 发 `tcgen05.mma`。

### 1.3 关键设计选择

| 设计点 | 取值 | 理由 |
| --- | --- | --- |
| 编译方式 | 每个 (shape, config) 组合 JIT 生成一份全量模板特化的 `.cu` 编到 cubin | BLOCK_*、swizzle、stage 数、甚至 N/K 本身都成为编译期常量 → 内层 K 循环完全展开、描述符偏移常量折叠 |
| Grid | `gridDim.x = num_sms`，`__launch_bounds__(kNumTMAThreads+kNumMathThreads, 1)` | persistent kernel，1 CTA/SM；配合近满额 SMEM，物理上排除 2 CTA 共 SM |
| Cluster | 1 或 2（`cluster_m×cluster_n ≤ 2`） | 用 **TMA multicast** 让 pair 内两个 CTA 共享一份 A 或 B，省 L2/GMEM 带宽（**注意：不省 SMEM，见 §7.2**） |
| MMA 指令 | `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16`（`_SS` 变体） | Hopper warpgroup MMA；A/B 均来自 SMEM descriptor，累加器在 **寄存器** |
| `WGMMA::M` | 恒为 **64** | Hopper wgmma 的 M 原子固定 64（一个 warpgroup 4 warp × 16 行）；`BLOCK_M ∈ {16,32}` 时也照发 M=64（见 §7.5） |
| `WGMMA::K` | 恒为 **16** | BF16 的 wgmma K 原子固定 16 |
| `WGMMA::N` | `= BLOCK_N ∈ {8,16,…,256}` | 由 `BF16MMASelector<BLOCK_N>` 选出对应的 `MMA_64xNx16` |
| 累加器 | 寄存器数组 `accum[kNumAccum × waves]`，`kNumAccum = 64×BLOCK_N/128 = BLOCK_N/2` | 没有 TMEM；每线程持 `BLOCK_N/2` 个 FP32（×wave 数） |
| C/D 缓冲 | SMEM **单缓冲**（`SMEM_D_SIZE`） | 靠 `tma_store_wait<0>()` 串行复用；不像 SM100 双缓冲重叠 |
| A/B 环 | 尽可能多的 stage（host 端按 SMEM 预算反解，上限 **16**） | 掩盖 HBM 延迟；stage ≥ 10 且 NT-Normal-单 math warpgroup 时再做「stage 合并」把 `BLOCK_K` 放大（见 §7.4） |
| 寄存器再分配 | TMA warpgroup `dealloc<48>`，math warpgroup `alloc<224 或 248>` | 累加器吃寄存器，把 TMA warpgroup 的配额让给 math warpgroup |
| PDL | `cudaGridDependencySynchronize()` 放在 barrier init + cluster sync **之后** | barrier 初始化、descriptor prefetch 与前驱 kernel 的尾巴重叠 |
| 不支持 | swap-AB、Tensor Core 利用率控制、tail-K 专用分支 | SM90 路径明确断言 `swap_ab==0`；tail-K 靠 TMA 零填充自然处理（见 §7.6） |

### 1.4 SM90 ↔ SM100 对照总表

这张表是理解本 kernel 的钥匙——很多 SM90 的「为什么这么写」只有在与 SM100 对照时才清晰。

| 维度 | SM90（本文） | SM100（姊妹篇） |
| --- | --- | --- |
| Tensor Core 指令 | `wgmma.mma_async.m64nNk16` | `tcgen05.mma.cta_group::{1,2}.kind::f16` |
| 发射宽度 | **整 warpgroup（128 线程）协同** | **单线程**（ elected lane） |
| A 操作数 | SMEM descriptor（`_SS`） | SMEM descriptor（`_SS`） |
| B 操作数 | SMEM descriptor | SMEM descriptor |
| **D 累加器** | **寄存器**（分散在 128 线程，每线程 `BLOCK_N/2` 个 f32） | **TMEM**（独立 256 KB 存储，128 行 × 512 列） |
| MMA 的 M | 固定 64 | `128 × kNumMulticast`（可达 256） |
| 为什么需要那么多线程 | 累加器每个元素都得有个线程「拿着」 | 没人需要「拿着」任何东西 |
| compute 与 epilogue | **同一批 math 线程串行**（累加器占用寄存器整个 k-loop） | MMA warp 与 epilogue warp **分离**，TMEM 双缓冲重叠 |
| cluster 协作 | TMA **multicast**（复制操作数，省带宽不省 SMEM，两 CTA 各算各的块） | **2-CTA UMMA**（切分操作数，省带宽也省 SMEM，两 SM tensor core 合算一块） |
| C/D SMEM | 单缓冲 | 双缓冲（`kNumTMAStoreStages=2`） |
| 最大 stage 数 | 16 | 32 |
| 描述符 per-stage 存储 | 单个 `a_desc_lo`/`b_desc_lo`，每 stage 加常量步长 | 32 个 lane 各存一个 stage 的描述符低位，`__shfl` 取用 |
| barrier 种类 | 2 类（`full` / `empty`） | 5 类（`full`/`empty`/`tmem_full`/`tmem_empty`/`tensor_core_full`） |
| swap-AB | 不支持（`DG_HOST_ASSERT(swap_ab==0)`） | 支持（MoE 小 M 场景） |
| 寄存器再配置 | 有（`setmaxnreg`） | 无（TMEM 不吃寄存器） |
| **线程填充率是否是有效指标** | **是**——math warpgroup 的 128/256 线程稳态都在算或搬 | **否**——256 线程稳态仅约 34 活跃，须看 tensor pipe 指标 |

最后一行尤其重要：SM90 的 `wgmma` 要 128 线程是**存储约束**（累加器在寄存器）而非算力约束，但这些线程在 k-loop 里确实都参与发射、在 epilogue 里确实都参与搬运，所以「活跃线程数」对 SM90 是有意义的健康指标；而对 SM100 用同一指标会得出完全错误的「低效」结论。评估 SM90 时仍可结合 `sm__pipe_tensor_cycles_active`，但 `sm__warps_active` 不再像 SM100 那样具有误导性。

---

## 2. Host 侧：JIT 特化与配置推导

Kernel 的全部行为由 22 个模板实参决定，它们在 [sm90_bf16_gemm.hpp](../third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) 的 `generate_impl()` 里被格式化成一段只包含 `__instantiate_kernel()` 的 `.cu` 源码，再交给 nvcc 编成 cubin。

### 2.1 模板实参来源

device 侧模板签名（[sm90_bf16_gemm.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh) 第 28–38 行）：

```cpp
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages_,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t>
```

各实参在 host 侧的来源（`generate_impl` 第 50–65 行）：

```
kMajorA / kMajorB        ← to_string(gemm_desc.major_a/b)：由 a/b 的 stride 推断
SHAPE_M / SHAPE_N / SHAPE_K ← get_compiled_dim(dim, 'm'/'n'/'k', compiled_dims)：
                            在 compiled_dims 里 → 填真实值；否则填 0（= 运行期参数）
kNumGroups               ← gemm_desc.num_groups
BLOCK_M / BLOCK_N / BLOCK_K_ ← gemm_config.layout.{block_m, block_n, block_k}
kSwizzle{A,B,D}Mode      ← gemm_config.storage_config.swizzle_{a,b,cd}_mode
kNumStages_              ← gemm_config.pipeline_config.num_stages
kNumTMAThreads           ← launch_config.num_tma_threads（恒 128）
kNumMathThreads          ← launch_config.num_math_threads（block_m≤64 ? 128 : 256）
kNumTMAMulticast         ← layout.get_cluster_size()（= cluster_m × cluster_n）
kIsTMAMulticastOnA       ← (layout.cluster_n > 1)
kNumSMs                  ← launch_config.num_sms（= gridDim.x）
kGemmType / kWithAccumulation / cd_dtype_t ← gemm_desc
```

与 SM100 的 26 个实参相比，SM90 **少了** `kSwapAB`、`kEnsureZeroPadding`、`kKAlignment`、`kTensorCoreUtilControl`、`kNumNonEpilogueThreads/kNumEpilogueThreads` 这一批——因为 SM90 路径不支持 swap-AB、没有 tensor core 利用率旋钮、epilogue 与 math 是同一批线程。**多了** `kNumTMAThreads/kNumMathThreads`（SM100 用的是 `kNumNonEpilogueThreads/kNumEpilogueThreads`，两套线程划分模型不同）。

Python 侧 `compiled_dims` 默认 `"nk"`（grouped contiguous/masked 也是 `"nk"`，两个 batched einsum 变体是 `"mn"`）。因此典型场景下 `SHAPE_M == 0`、`SHAPE_N`/`SHAPE_K` 为编译期常量，kernel 内第 68–70 行的覆写：

```cpp
shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
```

会把 N/K 变成常量。注意 SM90 **没有** SM100 那个 `kMayHaveTailKBlock` 的编译期分支——即便 K 是常量，SM90 也不生成专门的 tail-K 代码，而是靠 TMA 的 OOB 零填充把不足 `BLOCK_K` 的尾巴补 0（见 §7.6）。这是 SM90 相对 SM100 少掉的一整块复杂度。

### 2.2 Layout 候选枚举

`SM90ArchSpec::get_layout_candidates()`（[heuristics/sm90.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/sm90.hpp) 第 16–118 行）：

- **`block_k` 恒定**：`128 / element_size(BF16=2) = 64`。即 `BLOCK_K_ == 64`，物理含义是一个 swizzle atom 的 K 方向字节数固定 128 B。
- **`block_m` 候选**（按 GemmType 分流）：
  - Normal / Batched / KGroupedContiguous：基础 `{64, 128}`；`m ≤ 16` 追加 `16`，`m ≤ 32` 追加 `32`（注释：*smaller block M can avoid TMA L2 OOB bound*）；若输出非 FP32 再追加 `256`（*BF16 output GEMM supports 256*）。
  - MGroupedContiguous{,WithPsumLayout}：强制 `= get_mk_alignment_for_contiguous_layout()`。
  - MGroupedMasked：`{64, 128}`。
- **`block_n` 候选**：步长 `lcm(16, block_n_multiple_of)`，从 `step` 到 `end` 枚举；`end` 按 kernel 类型收紧（NoSF→256、1D2D→192、1D1D→160，注释：*Register spills*）。BF16 GEMM 走 NoSF，故 `block_n ∈ {16,32,…,256}`。
- **`disable_multicast`**：k-grouped 且 `num_groups > 4`，或 Batched → 禁用 multicast（`cluster` 只枚举到 1）。
- **逐条过滤**（第 71–110 行）：
  - `cluster_m × cluster_n > 2`、`num_sms % cluster_size != 0` → 跳过。
  - `block_m > 128 && block_n > 128` → 跳过（*for enough registers, at least one dim less than 128*）——这是寄存器容量约束，因为累加器吃 `block_n/2 × waves` 个寄存器。
  - masked / psum 布局要求 `ceil_div(n, block_n) % cluster_size == 0`（multicast 合法性）。
  - `swizzle_a_mode % 64 != 0 || swizzle_b_mode % 64 != 0` → 跳过（*32B's performance is low*）。
  - **stage 数下限**：`num_stages < 3`，或（`block_m×block_n < 128×192` 且 `num_stages < 4`）→ 跳过（*To hide TMA latency*）。
- **打分 `compare()`**：SM90 用的是一个**解析带宽模型**（`get_layout_info`，第 201–238 行），而非 SM100 的「wave 优先」字典序：

```cpp
num_bytes_l2_ab = expected_k * (block_m/cluster_n + block_n/cluster_m) * elem_ab;   // multicast 省 L2
num_bytes_l1_ab = expected_k * (block_m + block_n) * elem_ab;
num_bytes_l1_tc = expected_k * (max(64, block_m) + block_n) * elem_ab + block_m*block_n*elem_cd;
num_l2_cycles   = (num_bytes_l2_ab + num_bytes_l1_l2_cd) * num_blocks / l2_bw_per_cycle;
num_l1_cycles   = (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) * num_blocks / l1_bw_per_cycle;
num_cycles      = max(num_l1_cycles, num_l2_cycles) / wave_efficiency;
// compare: a.num_cycles < b.num_cycles
```

即 SM90 显式建模 L1（128 B/cycle/SM）与 L2（`min(64×num_sms, 8e6/1.3e3)` B/cycle）带宽，取二者瓶颈周期、再除以 wave 效率作为代价。`num_bytes_l2_ab` 里的 `block_m/cluster_n + block_n/cluster_m` 正是 **multicast 把某一维的 L2 流量对半**的体现。若只有 1 个 wave（`num_waves ≤ 1`），multicast 直接被判定为无穷大代价（*Disable multicasting if only one wave exists*）——单 wave 时 multicast 没有跨块复用，只剩开销。

### 2.3 StorageConfig

```cpp
constexpr int wgmma_m = 64;
DG_HOST_ASSERT(layout.swap_ab == 0);            // SM90 不支持 swap-AB
load_block_m  = layout.block_m;                 // 注意：不除 cluster！
load_block_n  = layout.block_n;
store_block_m = (kernel_type == 1D1D) ? 64 : layout.block_m;   // BF16 GEMM 走 block_m
store_block_n = layout.block_n;

swizzle_mode_a  = get_swizzle_mode(major_a == K ? block_k : load_block_m, sizeof(a));
swizzle_mode_b  = get_swizzle_mode(major_b == K ? block_k : load_block_n, sizeof(b));
swizzle_mode_cd = (cd_dtype != float) ? get_swizzle_mode(store_block_n, sizeof(cd)) : 0;
```

两个与 SM100 的关键差异：

1. **`load_block_m/n` 不除以 cluster**。SM100 是 `load_block_m = block_m / cluster_n`（2-CTA UMMA 各存一半）；SM90 是 `load_block_m = block_m`（multicast 各存**整份**）。这正是 §1.4「multicast 省带宽不省 SMEM」的根源，详见 §7.2。
2. **FP32 输出时 `swizzle_mode_cd = 0`**（*We only enable swizzling for non-FP32 outputs*）。因为 FP32 走 `st.shared.v2.f32` 而非 STSM，且 `TMA_D_BLOCK_N` 在 `kSwizzleDMode==0` 时退化为整个 `BLOCK_N`（单条 TMA store）。BF16 输出时 `swizzle_cd = get_swizzle_mode(block_n, 2)`，`block_n ≥ 64` 时恒为 128。

`get_swizzle_mode()`（[heuristics/utils.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/utils.hpp)）从 `{128,64,32,16}` 里挑第一个能整除 `block_size × elem_size` 的值。BF16 + `block_k=64` → 128 B，恒为 `swizzle=128`。

### 2.4 PipelineConfig（SMEM 预算 → stage 数）

```cpp
constexpr int kNumMaxStages = 16;                                    // SM100 是 32
const int smem_cd       = align(block_m * block_n * elemsize_cd, 1024);   // 单缓冲！
const int smem_barriers = kNumMaxStages * 8 * 2;                     // = 256 B（只有 full+empty 两组）
const int smem_a_per_stage = load_block_m * block_k * elemsize_a;
const int smem_b_per_stage = load_block_n * block_k * elemsize_b;
// BF16 GEMM 无 SF、无 extra tensormap
const int smem_extra     = smem_cd + smem_barriers;
const int smem_per_stage = smem_a_per_stage + smem_b_per_stage;
const int num_stages = min((smem_capacity - smem_extra) / smem_per_stage, kNumMaxStages);
smem_size = smem_extra + num_stages * smem_per_stage;
```

`smem_capacity = 232448`（227 KB，H100 上限）。三处与 SM100 的差异精确对应 device 布局（§5）：

- **`smem_cd` 单缓冲**：SM100 是 `× 2`（双缓冲），SM90 没有 `× 2`。对应 device 侧只有一个 `smem_d`。
- **`smem_barriers = 16×8×2`**：只有 `full` + `empty` 两组 barrier，每组按 `kNumMaxStages=16` 预留。SM100 是 `32×8×3 + 2×8×2 + 8`（三组 + tmem 两组 + tensor_core），BF16 那里第三组是死占位。SM90 干净得多——没有 TMEM，自然没有 `tmem_full/tmem_empty`。
- **`kNumMaxStages = 16`**：上限只有 SM100 的一半。原因不是 lane 寄存器（SM90 不像 SM100 那样用 32 个 lane 存描述符），而是纯粹的经验/SMEM 预算取舍。

### 2.5 LaunchConfig（线程数模型）

```cpp
const int num_tma_threads  = 128;                          // 恒 1 个 warpgroup
const int num_math_threads = layout.block_m <= 64 ? 128 : 256;   // 1 或 2 个 warpgroup
return { num_sms, cluster_size, num_tma_threads + num_math_threads,
         num_tma_threads, num_math_threads, 0, 0 };         // 后两项 SM90 无意义
```

于是总线程数 ∈ `{256, 384}`：

| `BLOCK_M` | math warpgroup 数 | `kNumMathThreads` | 总线程 | warp 总数 | math warp | TMA warp |
| --- | --- | --- | --- | --- | --- | --- |
| 16 / 32 / 64 | 1 | 128 | 256 | 8 | w0–w3 | w4–w7 |
| 128 / 256 | 2 | 256 | 384 | 12 | w0–w7 | w8–w11 |

`num_math_threads` 由 `BLOCK_M` 决定的本质：每个 math warpgroup 覆盖 `WGMMA::M = 64` 行，`BLOCK_M > 64` 就需要第二个 warpgroup。`BLOCK_M = 256` 时仍是 2 个 warpgroup，但要做 2 个「wave」（§6.2），而不是 4 个 warpgroup。

`__launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)` 的第二个参数 `1` 声明每 SM 最多 1 个 block——与近满额 SMEM 一起，物理上排除 2 CTA 共 SM。

### 2.6 TMA descriptor

三个 descriptor 都以 `__grid_constant__ cute::TmaDescriptor` 按值传参（128 B 常量内存，避免走 GMEM）。构造见 [runtime_utils.hpp](../third_party/DeepGEMM/csrc/jit_kernels/impls/runtime_utils.hpp)：

| descriptor | gmem (inner, outer) | smem box (inner, outer) | 备注 |
| --- | --- | --- | --- |
| A | K-major: `(k, m×G)`；MN-major: `(m×G, k)` | K-major: `(block_k→64, block_m)`；MN-major: `(block_m, block_k)` | `num_groups > 1` 时强制 K-major；box 内维被 swizzle 覆写为 `swizzle/elem` |
| B | K-major: `(k, n)`；MN-major: `(n, k)` | 同上，`block_n` 换 `load_block_n` | `num_groups` 只作用在外维：`gmem_outer × num_groups` |
| C/D | `(n, m×G)` | `(store_block_n→swizzle/elem, store_block_m)` | D 必须 N-major；FP32 时 swizzle=0，box 内维=`block_n` |

公共属性：`CU_TENSOR_MAP_INTERLEAVE_NONE`、`CU_TENSOR_MAP_L2_PROMOTION_L2_256B`、`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`（越界元素零填充——这是 tail-K 与 M/N 非整除时结果仍正确的硬件保证，见 §7.6）、swizzle 由 `mode_into_tensor_map_swizzle()` 映射到 `CU_TENSOR_MAP_SWIZZLE_{NONE,32B,64B,128B}`。

Batched（`sm90_bf16_bhr_hdr_bhd` / `bhd_hdr_bhr`）走 `make_tma_3d_desc()`，第三维是 head，device 侧 `kIsBatchedMM`（= `kGemmType == Batched`）打开 `SM90_TMA_LOAD_3D` 分支。SM90 一共暴露 5 个 host 入口：`sm90_bf16_gemm`（Normal）、`sm90_m_grouped_bf16_gemm_contiguous`、`sm90_bf16_m_grouped_gemm_masked`、`sm90_bf16_k_grouped_gemm`、两个 batched einsum。

### 2.7 Launch 属性

`LaunchArgs` 携带 `num_sms / num_threads / smem_size / cluster_size`，最终经 `launch_kernel` → `cuLaunchKernelEx`：

1. `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size)` —— 动态 SMEM 远超 48 KB 静态上限，必须显式抬。
2. `cluster_size > 1` → `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = {cluster_size, 1, 1}`。
3. `enable_pdl` → `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 1`。默认关闭，需 `deep_gemm.set_pdl(True)` 打开（与 SM100 共用 `DeviceRuntime`）。
4. 编译 flag：`--gpu-architecture=sm_90a`（wgmma 与 TMA multicast 都需要 `a` 后缀的架构特性），`-O3 --expt-relaxed-constexpr --expt-extended-lambda`，产物 `-cubin`。kernel 内 `#if __CUDA_ARCH__ >= 900` 之外的分支只留 `DG_DEVICE_ASSERT(false and "This kernel only support sm_90a")`（第 386–389 行）。

---

## 3. 线程组织与 warp 角色

### 3.1 角色表（以 `BLOCK_M=128` ⇒ 384 线程 / 12 warp 为例）

| warp | 条件 | 角色 | 实际活跃线程 |
| --- | --- | --- | --- |
| 0–3 | `warp_idx < kNumMathThreads/32` | **math warpgroup 0**：发 `wgmma`（rows 0–63）、持累加器、做 epilogue | 128（全 warpgroup 协同发 wgmma） |
| 4–7 | 同上（`kNumMathThreads=256` 时） | **math warpgroup 1**：rows 64–127 | 128 |
| `kNumMathThreads/32`（=8） | `warp_idx == kNumMathThreads/32 && elect_one_sync()` | **prefetch** 三个 TMA descriptor（prologue 一次） | 1 lane |
| `+1`（=9） | `warp_idx == kNumMathThreads/32+1 && elect_one_sync()` | **init barriers**（prologue 一次）+ `fence_barrier_init` | 1 lane |
| `+2`（=10） | `warp_idx == kNumMathThreads/32+2 && elect_one_sync()` | **TMA load**：persistent 遍历所有块，每块跑完整 K 循环，发 A/B 的 TMA | **1 lane**（TMA 单线程指令） |
| `+3`（=11） | — | **空转** | 0 |

`BLOCK_M ≤ 64`（`kNumMathThreads=128`）时，math warpgroup 只有 w0–w3，TMA warpgroup 是 w4–w7，角色一一对应下移。

源码第 152–154 行的注释解释了为什么 TMA load 用**第三个** warp 而非第一个：

```cpp
// NOTES: only one thread (or warp) will be used
// We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`
if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
```

这里的「warp 0/1」指的是 math warpgroup 内部的相对编号——当 `BLOCK_M=32 < WGMMA::M=64` 时，只有前 2 个 math warp（w0/w1）参与 WGMMA store（§3.4），把 TMA 的三个 prologue 角色错开到 math warpgroup 之后、且集中在同一个 TMA warpgroup 内，避免与 math warp 争用。

### 3.2 寄存器再配置（`setmaxnreg`）

Hopper 引入的 warpgroup 级寄存器再分配，本 kernel 用它把寄存器从「不怎么用寄存器的 TMA warpgroup」挪给「累加器吃寄存器的 math warpgroup」：

```cpp
// 第 128–129 行：编译期常量
constexpr uint32_t kNumTMARegisters  = 48;
constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

// TMA warpgroup 分支（第 150 行）
cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();   // setmaxnreg.dec.sync.aligned.u32 48

// math warpgroup 分支（第 210 行）
cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();    // setmaxnreg.inc.sync.aligned.u32 224/248
```

`kNumMathRegisters` 的取值是被 **64K 寄存器/SM 的总预算**反解出来的（`setmaxnreg` 要求 8 的倍数、范围 `[24, 256]`）：

| 配置 | 总线程 | math 配额 | TMA 配额 | 合计 | 是否 ≤ 65536 |
| --- | --- | --- | --- | --- | --- |
| 1 math wg（`BLOCK_M≤64`） | 256 | `128 × 248 = 31744` | `128 × 48 = 6144` | 37888 | ✓（还有余量，但 248 已接近单线程上限 255） |
| 2 math wg（`BLOCK_M>64`） | 384 | `256 × 224 = 57344` | `128 × 48 = 6144` | 63488 | ✓（贴着上限，故 math 只能给到 224 而非 248） |

这解释了「为什么 2 个 math warpgroup 时每线程寄存器反而更少（224 < 248）」：线程总数从 256 涨到 384，为了塞进同一个 64K 寄存器文件，单线程配额必须下调。这也从侧面印证了 §2.2 那条 `block_m>128 && block_n>128 → 跳过` 的过滤——累加器 `block_n/2 × waves` 个寄存器 + 描述符 + 地址，在 224/248 的预算里放不下两个都大的维度。

**SM100 没有这一步**：累加器在 TMEM 不吃寄存器，无需再分配。寄存器再配置是 SM90「累加器在寄存器」这一根本约束的直接衍生。

### 3.3 为什么 WGMMA 需要 128 个线程

`wgmma.mma_async.sync.aligned` 是 **warpgroup 级协同指令**：`.sync.aligned` 要求 warpgroup 内全部 128 线程收敛执行，每个线程贡献自己那一份 A（若 `_RS`）/持有自己那一份 D 累加器寄存器。本 kernel 用 `_SS` 变体（A、B 都来自 SMEM descriptor），所以线程不参与提供 A/B 数据，但**必须全部到场**，因为：

- 一条 `m64nNk16` 的 D 累加器是 `64 × N` 个 FP32，硬件把它按固定图案**分散到 128 个线程的寄存器**里，每线程恰好 `64×N/128 = N/2 = kNumAccum` 个。
- `WGMMA::wgmma(desc_a, desc_b, shifted_accum, 1)` 的 `shifted_accum` 就是本线程持有的那 `kNumAccum` 个寄存器的指针；`call_fma_impl` 用 `cute::make_index_sequence<N_/2>` 把它们逐个展开成 `MMA::fma(desc_a, desc_b, d[0], d[1], …, scale)` 的操作数。

对照 SM100：`tcgen05.mma` 的 D 在 TMEM，没有任何线程需要「拿着」累加器，于是发射退化成单 lane 的控制指令。**SM90 的 128 线程是被累加器的存储位置逼出来的，与算力无关**——但和 SM100 不同的是，这些线程在 epilogue 阶段确实要干活（把寄存器里的累加器搬进 SMEM），所以它们在整个输出块生命周期里都是「有事做」的，不是空转。

### 3.4 math warpgroup 数量、wave 与 store 线程

三个编译期量共同决定了 math 侧的形状（第 223–230 行）：

```cpp
constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;   // ≤64→BLOCK_M；>64→128
DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};                    // 每线程的累加器

constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);        // = WAVE_BLOCK_M × 2
const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;
```

| `BLOCK_M` | math wg 数 | `WAVE_BLOCK_M` | wave 数 `BLOCK_M/WAVE_BLOCK_M` | 每线程 accum | `kNumWGMMAStoreThreads` | 参与 store 的 warp |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 1 | 16 | 1 | `(16/2)×1=8`… 见下注 | 32 | w0（`BLOCK_M<64`，只前 1 warp） |
| 32 | 1 | 32 | 1 | 16 | 64 | w0–w1 |
| 64 | 1 | 64 | 1 | 32 | 128 | w0–w3（`BLOCK_M≥64` 全 store） |
| 128 | 2 | 128 | 1 | 64 | 256 | 全 8 warp |
| 256 | 2 | 128 | 2 | 128 | 256 | 全 8 warp |

> 注：每线程 accum 数 = `kNumAccum × waves = (BLOCK_N/2) × waves`，上表按 `BLOCK_N=64` 举例（`kNumAccum=32`）；实际值随 `BLOCK_N` 线性变化，`BLOCK_N=256` 时每线程可高达 `128 × waves` 个 FP32，这正是 §2.2 寄存器过滤的由来。

关键区分：

- **wave**（`local_idx`）：一个 math warpgroup（覆盖 64 行）需要跑几轮才能覆盖 `WAVE_BLOCK_M`。`BLOCK_M=256` 时 `WAVE_BLOCK_M=128`、2 个 warpgroup各覆盖 64 行合起来 128 行 = 1 个 wave，需要 2 个 wave 才够 256 行。wave 之间累加器数组 `accum` 分段（`shifted_accum = accum + kNumAccum × local_idx`）。
- **`do_wgmma_store`**：`BLOCK_M ≥ 64` 时全部 math warp 都参与 epilogue；`BLOCK_M < 64`（16/32）时，因为 WGMMA 按 M=64 发射、但只有前 `BLOCK_M` 行有效，只有前 `kNumWGMMAStoreThreads/32` 个 warp 的结果需要写回，其余 warp `continue` 跳过（第 289–290 行）。

`a_desc` 在构造时就用 `math_wg_idx * WGMMA::M` 把每个 warpgroup 定位到自己那 64 行（第 217 行），wave 内再用 `local_idx * WAVE_BLOCK_M` 推进（第 262–263 行）。两者叠加：warpgroup `g`、wave `w` 覆盖的全局行区间是 `[g×64 + w×WAVE_BLOCK_M, +64)`。

---

## 4. Persistent 调度器

SM90 与 SM100 **共用同一个** `sched::Scheduler`（[scheduler/gemm.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh)），差异只在模板实参与少数 `#if __CUDA_ARCH__ < 1000` 的 SM90 专属分支。

### 4.1 复制式状态机，不是共享工作队列

第 135–136 行在**角色分派之前**构造 `scheduler`：

```cpp
auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups,
                                  kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(
    shape_m, shape_n, shape_k, grouped_layout);
```

它是个**寄存器里的值对象**，每个线程各持一份私有副本。TMA warp（w10）与每个 math warpgroup 各自独立调用 `get_next_block()`，靠 `next_block_idx = (++current_iter) * kNumSMs + blockIdx.x`（第 198 行）这一条纯算术式子得到**完全相同**的块序列：

```
iter 0 → blockIdx.x
iter 1 → blockIdx.x + kNumSMs
iter 2 → blockIdx.x + 2*kNumSMs
...
```

因此：

- **零原子操作、零全局 ticket 计数器**，调度开销是几条整数指令；
- TMA warp 与 math warpgroup 天然锁步，无需为「现在在处理哪一块」建立任何额外通信；
- 与 SM100 不同的是，SM90 **不需要**用 `current_iter` 去推导 TMEM 累加器缓冲的 stage/phase（没有 TMEM）。SM90 里 math warpgroup 与 TMA warp 的耦合只通过 A/B 环的 `full`/`empty` barrier，累加器是每块就地 `{0}` 重置的寄存器数组。

代价同样是负载不能动态窃取：尾波的空闲 CTA 只能干等。但 SM90 的打分模型（§2.2）用解析带宽模型 + `wave_efficiency` 提前量化了这件事。

### 4.2 L2 swizzle 分组

`get_swizzled_block_idx()`（第 117–153 行）把线性的 `block_idx` 重映射成 `(m_block_idx, n_block_idx)`，目的是让**同时在飞的 kNumSMs 个 CTA 尽量共享 L2 里的 A/B**。这段逻辑 SM90/SM100 完全一致：

```cpp
kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<...>();   // 编译期，∈ {8, 16}
primary_num_blocks   = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
group_idx            = block_idx / num_blocks_per_group;
first_block_idx      = group_idx * kNum1DBlocksPerGroup;
in_group_idx         = block_idx % num_blocks_per_group;
num_blocks_in_group  = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

// kIsMulticastOnA == false（在 M 上分组，组内 M 变化最快）
m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
n_block_idx =                  in_group_idx / num_blocks_in_group;
```

组大小选 `{8, 16}` 中最小化 L2 工作集者（`get_num_1d_blocks_per_group`，第 14–26 行）。`DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0)` 保证一个 cluster 的两个 CTA 不跨组边界。

### 4.3 SM90 专属：multicast 的动态关闭

这是 SM90 调度器与 SM100 最实质的分歧。SM100 的 2-CTA UMMA **不能动态关闭**（硬件成对发射），只能靠 host 侧整除性过滤；而 **SM90 的 TMA multicast 可以在运行期逐块关闭**，因为发不发 multicast load 只是 TMA warp 的一个分支选择，两个 CTA 始终各算各的块。为此调度器提供了两个 SM90-only 方法：

**① `is_tma_multicast_valid(m_block_idx)`**（第 290–307 行）：

```cpp
if (num_blocks_in_group == 1) return false;            // 组内只剩 1 块，无 peer 可 multicast
if constexpr (Normal / Masked / KGrouped / Batched / MGroupedPsum) return true;
else /* MGroupedContiguous */ {
    if constexpr (kIsMulticastOnA) return true;
    else return grouped_layout[m_block_idx*BLOCK_M] == grouped_layout[(m_block_idx^1)*BLOCK_M];  // peer 同组才能共享 B
}
```

TMA warp 在第 161 行读它，决定 `num_tma_multicast_a/b` 是取 `kNumTMAMulticast` 还是退回 `1`（第 162–163 行）。对 m-grouped contiguous，若相邻两块（`m_block_idx` 与 `m_block_idx ^ 1`）属于**不同 group**，它们的 B 就不是同一份，multicast 非法 → 退回单播。

**② `is_peer_cta_alive`**（第 281–283 行，仅 Normal 路径设置）：

```cpp
is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 or      // N 恒对齐（常量短路）
                    num_m_blocks % kNumMulticast == 0 or      // M 恒对齐（常量短路）
                    (next_block_idx ^ 1) < num_blocks;        // peer CTA 的块仍在界内
```

它服务于 `empty_barrier_arrive` 的目标 CTA 选择（§10.3）：当 cluster 的 peer CTA 因为落在矩阵边界外而没有有效块时，本 CTA 的 math warp 不能把 empty 信号远程投递给「不存在的 peer」，否则会写到一个没人等待、甚至已释放的 barrier 上。`is_peer_cta_alive == false` 时，两个 lane 都投递给本 CTA（`target_cta = block_rank_in_cluster()`）。

第 132–142 行还有一段 `#if __CUDA_ARCH__ < 1000` 的「修正不对齐的 TMA multicast」，注释点明：*for SM90 only, as SM90 can dynamically disable TMA multicast while SM100 uses 2-CTA, which can not be dynamically disabled*。当 `num_blocks_in_group` 是奇数时，它把最后一个落单的块单独成组（`num_blocks_in_group = 1`），从而让 `is_tma_multicast_valid` 对它返回 false、退回单播。

### 4.4 GemmType 变体

| GemmType | `num_blocks` | 额外状态 | 说明 |
| --- | --- | --- | --- |
| `Normal` | `num_m_blocks × num_n_blocks` | — | `get_global_idx` 退化为 `block_idx × block_size`；设置 `is_peer_cta_alive` |
| `Batched` | 同上，`× kNumGroups` | `current_group_idx` 作 batch_idx | 不走 swizzle，按 `kIsMulticastOnA` 决定 m/n 谁变化快；TMA 走 3D；host 侧禁用 multicast |
| `MGroupedContiguous` | 同上 | `grouped_layout[m]` = 每行所属 group | B 的外维加 `group × shape_dim` 偏移；multicast 需 peer 同组 |
| `MGroupedMasked` | 逐 group 累加 | `current_m_cumsum` | 边扫边把 `next_block_idx` 落到对应 group，`num_m_blocks` 每 group 重算 |
| `MGroupedContiguousWithPsumLayout` | 逐 group 累加 | `last_psum_m`/`current_psum_m`/`current_m_block_cumsum` | group 边界按 psum 偏移切分，`m_block_idx += last_psum_m / BLOCK_M` |
| `KGroupedContiguous{,WithPsumLayout}` | 同上 | `current_shape_k`/`current_k_cumsum`/`current_k_start,end` | 每 group 的 K 长度不同 → `num_total_k_blocks` 逐块变化；要求 A/B 都 MN-major |

对本 kernel 最关键的约束在第 177 行：

```cpp
DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous
                 or kMajorA == cute::UMMA::Major::K, "Invalid major");
```

即所有 m-grouped 变体的 A 必须 K-major（group 偏移加在外维上）。host 侧对应 `DG_HOST_ASSERT(major_a == K)`。

### 4.5 跨块连续的流水线状态

第 139–146 行：

```cpp
uint32_t stage_idx = 0, phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++ k_block_idx;
    // Flip phases only if reach the next first stage
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;      // 只在回绕到 stage 0 时翻转相位
};
```

`stage_idx`/`phase` 声明在**块循环之外**，意味着 A/B 环**跨输出块连续运转**：TMA warp 可以在 math warpgroup 还在算第 i 块最后一个 k_block 时，就开始往刚被释放的 stage 里灌第 i+1 块的 k=0 数据。块边界上没有「排空-重启」的开销，这是 persistent kernel 相较 grid-per-tile 的核心优势。

`advance_pipeline` 同时被 TMA warp（第 167 行 `for` 的递增式）和 math warpgroup（第 244 行）使用，两边独立维护但推进规则一致，因此 `stage_idx`/`phase` 序列天然对齐。注意 SM90 的 `phase` 翻转写成 `phase ^= stage_idx == 0`（先更新 stage_idx 再判断是否回绕到 0），与 SM100 的 `stage_idx = (stage_idx+1) % kNumStages; phase ^= stage_idx == 0` 语义等价，只是三元写法不同。

与 SM100 的一个显著区别：SM100 的 C/D 环也是跨块连续的（`tma_stage_idx` 双缓冲）；SM90 的 D 是**单缓冲**，跨块复用同一个 `smem_d`，靠 epilogue 开头的 `tma_store_wait<0>()` 串行化（§11.1）。所以 SM90 的「跨块连续」只体现在 A/B 环，D 侧是块间串行的。

---

## 5. 共享内存布局

### 5.1 线性布局

```cpp
extern __shared__ __align__(1024) uint8_t smem_buffer[];   // 1024 B 对齐，服务于 swizzle-128B
```

三个区段尺寸都是编译期常量（第 73–75 行）：

```cpp
static constexpr uint32_t SMEM_D_SIZE           = constexpr_align(BLOCK_M * BLOCK_N * sizeof(cd_dtype_t), 1024u);
static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);
```

注意 A/B 用的是 **`BLOCK_M`/`BLOCK_N` 而非 SM100 的 `LOAD_BLOCK_M/N`**——因为 SM90 的 multicast 不切分 SMEM（§7.2），每个 CTA 都存整份 A、整份 B，所以「load block」与「block」是同一个量，源码里干脆不引入 `LOAD_*` 记号。

`utils::PatternVisitor`（[common/utils.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/utils.cuh) 第 11–22 行）是个零开销的「下标 → 指针」闭包包装器（`operator[](i)` 直接调 lambda），用它替代指针数组，避免在 SMEM/寄存器里存 stage 指针表：

| 区段 | 起址 | 大小 | 访问器 |
| --- | --- | --- | --- |
| **D（单缓冲）** | `smem_buffer + 0` | `SMEM_D_SIZE` | `smem_d`（裸指针，非 ring） |
| A ring | `smem_buffer + SMEM_D_SIZE` | `kNumStages * SMEM_A_SIZE_PER_STAGE` | `smem_a[i]` |
| B ring | `+ kNumStages * SMEM_A_SIZE_PER_STAGE` | `kNumStages * SMEM_B_SIZE_PER_STAGE` | `smem_b[i]` |
| Barriers | `+ kNumStages * SMEM_B_SIZE_PER_STAGE` | 见 §5.2 | `full_barriers[i]` / `empty_barriers[i]` |

```cpp
auto smem_d = reinterpret_cast<cd_dtype_t*>(smem_buffer);
auto smem_a = PatternVisitor([&](uint32_t i){ return (bf16*)(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE); });
auto smem_b = PatternVisitor([&](uint32_t i){ return (bf16*)(smem_buffer + SMEM_D_SIZE
                                    + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE); });
```

三个 `DG_STATIC_ASSERT(... % 1024 == 0)`（第 95–96 行）保证每个区段起点都 1024 B 对齐——这是 swizzle-128B 的硬件要求（一个 swizzle atom 是 8 行 × 128 B = 1 KB，若基址不按 1 KB 对齐，TMA 写入的 swizzle 图案与 WGMMA 描述符解读的图案会错位）。`SMEM_D_SIZE` 显式 `constexpr_align(…, 1024)`，A/B 因 `BLOCK_* × 64 × 2` 天然是 1024 的倍数（`BLOCK_K=64`、bf16=2B ⇒ 每行 128 B，8 行 1 KB）。

**与 SM100 的三处结构差异**：

1. **D 在最前、且单缓冲**。SM100 是 `C/D ring`（`× kNumTMAStoreStages=2`）在最前；SM90 只有一个 `smem_d`，没有 ring 下标。
2. **没有 TMEM 基址槽**。SM100 在 barriers 之后还要放一个 4 B 的 `tmem_ptr_in_smem`（`tcgen05.alloc` 的结果）；SM90 无 TMEM，barriers 之后就是末尾。
3. **barriers 只有两组**（§5.2），SM100 有五组 + 一个空洞。

### 5.2 Barrier 区（干净的两组）

以 `Barrier`（= `cutlass::arch::ClusterTransactionBarrier`，8 B）为单位，`barrier_start_ptr` 起：

```cpp
auto barrier_start_ptr = (Barrier*)(smem_buffer + SMEM_D_SIZE
                          + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
auto full_barriers  = PatternVisitor([=](uint32_t i){ return barrier_start_ptr + i; });
auto empty_barriers = PatternVisitor([=](uint32_t i){ return barrier_start_ptr + kNumStages + i; });
```

| 索引区间 | 名称 | 个数 | `init()` 计数 |
| --- | --- | --- | --- |
| `[0, S)` | `full_barriers` | `kNumStages` | `1` |
| `[S, 2S)` | `empty_barriers` | `kNumStages` | `kNumTMAMulticast × kNumMathThreads/32` |

（`S = kNumStages`）

host 侧 `smem_barriers = kNumMaxStages * 8 * 2 = 256 B`（§2.4）与这里的 `2 × kNumStages` 个 Barrier 精确呼应。**没有 SM100 那个 `kNumStages` 大小的空洞**——SM100 的索引按「每 stage 三组 barrier」排布（第三组是 FP8/FP4 的 with-SF full barriers，BF16 下空占位），SM90 只有 full + empty 两组，索引连续，不留死区。

初始化由 `warp_idx == kNumMathThreads/32 + 1`（TMA warpgroup 的第二个 warp）的单个 lane 完成（第 113–122 行），随后：

```cpp
cutlass::arch::fence_barrier_init();     // fence.mbarrier_init.release.cluster
```

注释写明目的：*Make initialized barrier visible in async proxy*。mbarrier 会被 TMA 这个**异步代理**访问（`arrive_and_expect_tx` 的 tx 计数由 TMA 硬件回填），普通的 `__syncthreads()` 不足以建立 generic proxy → async proxy 的可见性，必须用这条 cluster 作用域的 release fence。之后第 125 行做 `cluster_sync_with_relaxed_arrive()`（multicast）或 `__syncthreads()`（单 CTA），确保 peer CTA 也能看到 leader 初始化的 barrier 状态。

### 5.3 A/B stage 内部排布

**K-major（最常见）**：TMA box 是 `(inner = BLOCK_K = 64 elem = 128 B, outer = BLOCK_M)`，swizzle atom = 8 行 × 128 B。一个 stage 就是 `BLOCK_M/8` 个 atom 沿 M 方向线性堆叠：

```
smem_a[s]  (BLOCK_M=128, BLOCK_K=64, bf16 → 16 KB)
┌──────────────────────── atom 0 : rows   0.. 7 ────────────────────────┐  ← SBO = 1024 B
│ row r: 128 B = 8 个 16-B bank group，物理位置 g' = g ^ (r % 8)         │
├──────────────────────── atom 1 : rows   8..15 ────────────────────────┤
│ ...                                                                   │
├───────────────────────  atom 15 : rows 120..127 ──────────────────────┤
└───────────────────────────────────────────────────────────────────────┘
```

`^ (r % 8)` 的 bank-group 置换就是 `CU_TENSOR_MAP_SWIZZLE_128B` / `cute::SM90::GMMA::LayoutType::B128` 定义的图案，由 TMA 硬件在写入时施加、由 WGMMA 硬件在读取时反解，软件两侧都不参与。作用是把「同一列的 8 个元素」打散到 8 个不同 bank group，消除 tensor core 按列取数时的 SMEM bank conflict。

**MN-major**：TMA box 变成 `(inner = BLOCK_MN, outer = BLOCK_K)`，`BLOCK_INNER_ATOM = swizzle/elem = 64`，`tma::copy` 内部循环 `BLOCK_MN / 64` 次，每次目的地址 `smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM`。即 SMEM 布局是「**K 外、MN-atom 内**」。WGMMA 描述符的 `stride_k` 也随之从 K-major 的 `1` 变成 `get_inner_block_atom_size<...>()`（§9.4）。**Stage 合并**（§7.4）触发时，一个 stage 内是 `kNumStagesPerMerge` 个 64-宽 K-atom 沿「MN 外、K-atom 内」排列，描述符构造用 `BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge` 而非合并后的 `BLOCK_K`。

### 5.4 D 单缓冲内部排布

D 只有一块 `SMEM_D_SIZE = align(BLOCK_M × BLOCK_N × sizeof(cd), 1024)`，覆盖整个输出块（不像 A/B 分 stage）：

- **BF16 输出**（`kSwizzleDMode > 0`，恒 128）：一块是 `BLOCK_M` 行 × 128 B，行内 8 个 16-B bank group 的 `^ (r % 8)` 置换。每个 swizzle atom 覆盖 `TMA_D_BLOCK_N = kSwizzleDMode / sizeof(bf16) = 64` 个 N 元素，故 `BLOCK_N` 宽被切成 `BLOCK_N / 64` 个 atom，对应 `BLOCK_N / TMA_D_BLOCK_N` 条 TMA store。
- **FP32 输出**（`kSwizzleDMode == 0`）：不 swizzle，D 就是行主序的 `BLOCK_M × BLOCK_N` 个 float；`TMA_D_BLOCK_N` 退化为整个 `BLOCK_N`，一条 TMA store 覆盖整块（§11.3）。

因为单缓冲，**下一个输出块的 epilogue 必须等上一块的 TMA store 把 SMEM 读完**才能覆写——这就是 epilogue 开头 `tma_store_wait<0>()` 的作用（§11.4），也是 SM90 无法像 SM100 那样让 compute 与 epilogue 重叠的根因之一。

### 5.5 WGMMA 越界读的静态防护（含一处代码瑕疵）

第 77–79 行：

```cpp
// NOTES: Make sure we have enough shared memory for WGMMA padding
static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages,
                 "Memory Out of bound for WGMMA");
```

动机与 SM100 §5.5 同源：`WGMMA::M` 恒为 64，当 `BLOCK_M = 16/32` 时 `SMEM_A_SIZE_PER_STAGE < 64 行`，WGMMA 仍**按 64 行去读 A**，越过的部分落在后续 A stage、乃至 B ring 上。那些行算出的 D 在 epilogue 里不会被读（`do_wgmma_store` 只放行前 `BLOCK_M` 行，§3.4），读到垃圾无害——但**必须落在本 CTA 已分配的动态 SMEM 内**。

**瑕疵**：`sizeof(__nv_fp8_e4m3)` 是 **1 字节**，而本 kernel 的 A 是 bf16（2 字节）。WGMMA 实际读取的 A footprint 是 `64 × BLOCK_K × 2` 字节，断言左值却只算了 `64 × BLOCK_K × 1`——**恰好少算一半**。这几乎可以肯定是从 FP8 kernel（`sm90_fp8_gemm.cuh`，A 为 e4m3、1 字节）移植到 BF16 时漏改的 `sizeof`。

为什么目前不出事：断言右值 `SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE × kNumStages` 由 `kNumStages`（≥3，通常 5–6）个 B stage 主导，裕量极大。以 `BLOCK_M=16, BLOCK_N=256, BLOCK_K=64, kNumStages=5` 为例，真实需求 `64×64×2 = 8192` B，右值 `16×64×2 + 256×64×2×5 = 2048 + 163840 = 165888` B——无论用 1 字节还是 2 字节，断言都轻松通过。所以这是一个**潜伏的、被裕量掩盖的松检查**：它把安全边界放宽了 2×，一旦将来出现「极小 `BLOCK_M` + 极小 `kNumStages` + 极小 `BLOCK_N`」的组合，理论上可能漏判一次真实越界。修正方式是把 `sizeof(__nv_fp8_e4m3)` 改成 `sizeof(__nv_bfloat16)`（或 `sizeof(cd 的输入 dtype)`），代价为零。详见 §13.3。

### 5.6 Worked Example：8192 × 8192 × 8192，BF16→BF16，H100（132 SM）

取一组代表性配置：`BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, cluster=(1,2)`（multicast on A），`swizzle A/B/D = 128`。

推导链：

```
kNumTMAMulticast = 2      (cluster_m=1, cluster_n=2)
kIsTMAMulticastOnA = true (cluster_n > 1)
load_block_m = BLOCK_M = 128   (不除 cluster！multicast 各存整份 A)
load_block_n = BLOCK_N = 128
WGMMA::M = 64, WGMMA::N = 128, WGMMA::K = 16, kNumAccum = 64×128/128 = 64
kNumMathThreads = 256 (BLOCK_M=128 > 64) ⇒ 2 个 math warpgroup，总线程 384
WAVE_BLOCK_M = 128 (BLOCK_M>64 ⇒ WGMMA::M×2)，wave 数 = 128/128 = 1
每线程 accum = kNumAccum × waves = 64 × 1 = 64 个 FP32
kNumMathRegisters = 224 (2 wg)，kNumTMARegisters = 48
kDoMergeStages = false (num_stages=6 < 10)
kNum1DBlocksPerGroup: kIsMulticastOnA ⇒ 组在 N
   cand 8  → 8×128 + ceil(132/8)×128 = 1024 + 2176 = 3200
   cand 16 → 16×128 + ceil(132/16)×128 = 2048 + 1152 = 3200  ⇒ 平手取 8
```

num_stages 反解（`smem_capacity = 232448`）：

```
smem_cd        = align(128×128×2, 1024) = 32768   (单缓冲，无 ×2)
smem_barriers  = 16×8×2 = 256
smem_per_stage = 128×64×2 (A) + 128×64×2 (B) = 16384 + 16384 = 32768
num_stages     = min((232448 − 32768 − 256) / 32768, 16) = min(6.08, 16) = 6
smem_size      = 32768 + 256 + 6×32768 = 229632
```

SMEM 字节表（单 CTA）：

| 偏移 | 大小 | 内容 |
| --- | --- | --- |
| 0 | 32 768 | D 单缓冲：128 行 × 128 B（bf16 输出，swizzle-128B） |
| 32 768 | 98 304 | A ring：6 × (128 × 64 × 2 B = 16 KB) |
| 131 072 | 98 304 | B ring：6 × (128 × 64 × 2 B = 16 KB) |
| 229 376 | 96 | barriers：`full[6] + empty[6] = 12` 个 Barrier |
| **device 合计** | **229 472** | |
| **host 申请** | **229 632** | `33 024 (extra) + 6 × 32 768` |

**关键对照**：本例开了 multicast（cluster=2），但每个 CTA 仍存**整份** A（16 KB/stage），SMEM 占用与不开 multicast 时**完全相同**。multicast 省下的是 L2/GMEM 带宽——两个 CTA 的 A 来自同一次 GMEM 读（§7.2、§2.2 打分模型里 `block_m/cluster_n` 那一项）。反观 SM100 的 2-CTA UMMA，同样 cluster=2 时 `LOAD_BLOCK_M = BLOCK_M/2`，A 的 SMEM 直接减半，stage 数能翻倍——这是「复制 vs 切分」的本质差别。

每个 k_block：单 CTA 载入 32 KB（A 16 KB + B 16 KB），计算 `128 × 128 × 64 × 2 = 2.10 MFLOP`，由 2 个 math warpgroup 各发 `BLOCK_K/WGMMA::K = 4` 条 `m64n128k16` WGMMA 完成自己那 64 行。

---

## 6. 寄存器累加器布局

> 本章对应 SM100 文档的「§6 Tensor Memory 布局」。SM90 没有 TMEM，累加器住在 math warpgroup 的寄存器里，因此这一章讲的是**一条 `wgmma` 的 D 累加器如何按硬件固定图案分散到 128 个线程的寄存器**，以及 kernel 如何用 `accum[]` 数组 + wave/local_idx 索引去命中它。

### 6.1 累加器数组与分片规则

math warpgroup 分支第 223–225 行：

```cpp
constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;   // ≤64→BLOCK_M；>64→128
DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};                    // 每线程的累加器
```

- `WGMMA::kNumAccum = WGMMA::M × WGMMA::N / 128 = 64 × BLOCK_N / 128 = BLOCK_N / 2`（[mma/sm90.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/mma/sm90.cuh) 第 92 行）。物理含义：一条 `m64nNk16` WGMMA 产生 `64 × N` 个 FP32 累加器，硬件把它们**均匀分散到发射它的 128 个线程**，每线程恰好 `64×N/128 = N/2` 个。
- `BLOCK_M / WAVE_BLOCK_M` 是 **wave 数**（§3.4）：一个 math warpgroup 覆盖 `WGMMA::M = 64` 行，`WAVE_BLOCK_M` 行需要 `WAVE_BLOCK_M/64` 个 warpgroup 并排；`BLOCK_M` 行需要 `BLOCK_M/WAVE_BLOCK_M` 个 wave 串起来。累加器数组按 wave 分段，每段 `kNumAccum` 个。
- `= {0}`：整个数组在**每个输出块开头**清零。这是 SM90 与 SM100 的一个关键差异——SM100 靠首条 UMMA 的 `scale_c=0` 覆写来「隐式清零 TMEM」，SM90 靠寄存器数组初始化显式清零（对寄存器数组而言 `{0}` 是免费的，编译成若干 `mov` 或直接被后续 WGMMA 覆盖）。

### 6.2 每线程寄存器占用（对照表）

每线程累加器数 = `kNumAccum × waves = (BLOCK_N/2) × (BLOCK_M/WAVE_BLOCK_M)`：

| `BLOCK_M` | `BLOCK_N` | wave 数 | `kNumAccum` | 每线程 accum(FP32) | `kNumMathRegisters` | accum 占比 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 128 | 1 | 64 | 64 | 248 | 26% |
| 128 | 128 | 1 | 64 | 64 | 224 | 29% |
| 128 | 256 | 1 | 128 | 128 | 224 | 57% |
| 256 | 128 | 2 | 64 | 128 | 224 | 57% |
| 256 | 256 | 2 | 128 | 256 | — | **超出，被 host 过滤** |

最后一行正是 §2.2 那条 `block_m > 128 && block_n > 128 → 跳过` 的由来：`BLOCK_M=256` 且 `BLOCK_N=256` 时每线程要 256 个 FP32 存累加器，加上描述符、地址、循环变量，224 的寄存器配额根本放不下（会 spill 到 local memory，性能崩塌）。host 侧用这条过滤把「两个维度都大」的组合直接排除。

**与 SM100 的根本对照**：SM100 的累加器在 TMEM（256 KB 独立存储，128 行 × 512 列），**一个寄存器都不占**，所以 SM100 能同时开 `UMMA_M=256`（双缓冲占满 512 列 TMEM）而不受寄存器约束。SM90 把累加器放寄存器，代价是：① 需要 128 个线程「拿着」它（§3.3）；② `BLOCK_M × BLOCK_N` 的乘积被寄存器容量卡死；③ 需要 `setmaxnreg` 从 TMA warpgroup 抢寄存器（§3.2）。

### 6.3 WGMMA 的累加目标与 scale_d

内层 K 循环第 257–266 行：

```cpp
for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;   // 定位到本 wave 那段
    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
        /* 更新 a_desc.reg32_[0] / b_desc.reg32_[0]（§9.4） */
        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);          // scale_d 恒为 1
    }
}
```

`WGMMA::wgmma`（[mma/sm90.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/mma/sm90.cuh) 第 85–87 行）：

```cpp
static void wgmma(uint64_t desc_a, uint64_t desc_b, float* d, bool scale_d) {
    call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
}
// call_fma_impl: MMA::fma(desc_a, desc_b, d[Idx]..., scale_d ? ScaleOut::One : ScaleOut::Zero);
```

`cute::make_index_sequence<N_/2>` 把本线程的 `kNumAccum = N/2` 个寄存器 `d[0..N/2-1]` **逐个展开**成 `MMA::fma` 的操作数——这就是 §3.3「每个线程必须到场」的具体体现：`fma` 的 D 操作数就是本线程寄存器里的 `shifted_accum[i]`。

`scale_d` 在本 kernel **恒传 1**（累加）。清零不靠 scale_d，而靠 §6.1 的 `accum[] = {0}`。对照 SM100：那里 `scale_c` 是运行期谓词（首条 UMMA 传 0 覆写、其余传 1 累加），因为 TMEM 无法像寄存器数组那样「一行代码清零」，只能借第一条 MMA 的覆写语义。SM90 有寄存器数组，直接用语言层面的 `= {0}` 更简单。

### 6.4 wave 与 warpgroup 的行定位

两级偏移共同确定「warpgroup `g`、wave `w` 覆盖哪些全局行」：

```cpp
// 构造时（第 217 行）：warpgroup 定位到自己那 64 行
auto a_desc = make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(
                  smem_a[0], math_wg_idx * WGMMA::M, 0);   // mn_idx = math_wg_idx × 64
// K 循环内（第 262–263 行）：wave 内再推进 local_idx × WAVE_BLOCK_M
a_desc.reg32_[0] = advance_gmma_desc_lo<...>(a_desc_base_lo,
                     local_idx * WAVE_BLOCK_M,              // mn_idx：wave 偏移
                     (k * WGMMA::K) % BLOCK_ATOM_K,         // k_idx：atom 内 K 偏移
                     atom_k_idx * BLOCK_M * BLOCK_ATOM_K);  // offset：第几个 K-atom（合并时才 >0）
```

叠加后，warpgroup `g`、wave `w` 读的 A 行区间是 `[g×64 + w×WAVE_BLOCK_M, +64)`。累加器侧用 `shifted_accum = accum + kNumAccum × local_idx`（`local_idx` 即 wave 号 `w`）命中对应段。A 描述符的行偏移与累加器数组的 wave 分段**严格对齐**——这是「同一批线程既发 WGMMA 又持累加器又做 epilogue」能够正确工作的前提。

| `BLOCK_M` | math wg 数 | `WAVE_BLOCK_M` | wave 数 | wg0 覆盖行 | wg1 覆盖行 |
| --- | --- | --- | --- | --- | --- |
| 64 | 1 | 64 | 1 | 0–63（w0） | — |
| 128 | 2 | 128 | 1 | 0–63（w0） | 64–127（w0） |
| 256 | 2 | 128 | 2 | 0–63(w0)+128–191(w1) | 64–127(w0)+192–255(w1) |

---

## 7. 矩阵 Tiling 层次

### 7.1 六级 tiling

以 §5.6 的例子（`BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, cluster=2`）为例，从全局到指令共六级：

| 级别 | 尺度 | 承载者 | 说明 |
| --- | --- | --- | --- |
| L0 全局 | `M × N × K` | grid | persistent，`gridDim.x = kNumSMs` |
| L1 wave | `kNumSMs` 个输出块 | grid 的一轮 | `current_iter` 递增一次；L2 swizzle 在此层重排 |
| L2 cluster | **2 个独立输出块**（共享 A 或 B） | CTA-pair | **与 SM100 本质不同**：两个 CTA 各算各的块，仅共享一份操作数（§7.2） |
| L3 CTA tile | `BLOCK_M × BLOCK_N = 128 × 128` | 单 CTA | 落在本 CTA math warpgroup 的**寄存器累加器**（分散在 256 线程） |
| L4 k_block / stage | `BLOCK_M × BLOCK_K`（A）+ `BLOCK_N × BLOCK_K`（B） | SMEM ring 的一格 | 流水的调度单位，`kNumStages` 格在飞 |
| L5 WGMMA atom | `WGMMA::M × WGMMA::N × WGMMA::K = 64 × 128 × 16` | 一条指令 | 每 warpgroup 每 wave 发 `BLOCK_K / WGMMA::K = 4` 条消费一个 stage |
| L6 swizzle atom | `8 行 × 128 B` | SMEM 物理布局 | TMA 写入与 WGMMA 读出共用的最小图案单位 |

K 方向总迭代数：`num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K)`（第 166/243 行）。用的是 `scheduler.current_shape_k` 而非 `shape_k`——k-grouped 变体里每个 group 的 K 长度不同，且这个值是**运行期**的，所以 K 主循环不能整体展开，只能展开内层的 `BLOCK_K/WGMMA::K` 次 WGMMA。

L2 那一行是 SM90 与 SM100 最大的 tiling 差异：SM100 的 L2 是「一条 `cta_group::2` UMMA 覆盖的 `2×BLOCK_M × BLOCK_N`单一块」，SM90 的 L2 是「两个本应独立、只是搭伴共享一份 A/B 的 `BLOCK_M × BLOCK_N` 块」。下面展开。

### 7.2 SM90 multicast：复制操作数，不切分输出块

`kNumTMAMulticast == 2` 时，一对 CTA（cluster rank 0 / rank 1）**各算一个完整的输出块**，仅通过 TMA multicast 共享其中一份操作数。共享 A 还是 B 由 `kIsTMAMulticastOnA`（= `cluster_n > 1`）决定：

**情形：`kIsTMAMulticastOnA == true`（multicast A，`cluster_n = 2`，组在 N）**

```
              n_block j           n_block j+1
           ┌──────────────┐   ┌──────────────┐
  m_block i │  CTA0 算       │   │  CTA1 算       │   ← 两个独立输出块
           │  D[i, j]       │   │  D[i, j+1]     │
           └──────────────┘   └──────────────┘
              ▲                    ▲
              │ 各自存整份 B（不同 n_block）
         ┌────┴───────────────────┴────┐
         │  A[i] 由 rank0 一条 multicast  │  ← 一份 GMEM 读
         │  同时写进 CTA0/CTA1 的 smem_a   │     服务两个 CTA
         └─────────────────────────────┘
```

- 调度器把**相邻的 `n_block_idx`** 分给 cluster 内两个 CTA（`kIsMulticastOnA` → 组内 N 变化最快，§4.2），它们的 `m_block_idx` 相同 → A block 相同。
- **A**：`tma::copy` 内部只由 `block_rank_in_cluster() == 0` 发一条 `SM90_TMA_LOAD_MULTICAST_2D`（CTA mask `0b11`），一份 GMEM 读同时写进两个 CTA 的 `smem_a[stage]`，tx 信号也同时打到两个 CTA 的 `full_barriers[stage]`。
- **B**：每个 CTA 发自己的 `SM90_TMA_LOAD_2D`（`num_tma_multicast_b == 1`），各存自己 `n_block` 的整份 B。
- **计算**：每个 CTA 用自己的 A 副本 + 自己的 B，独立算出一个 `128 × 128` 块，写进自己的寄存器累加器、自己的 `smem_d`、自己的 GMEM D。

**省什么、不省什么**：multicast 省的是 **A 的 L2/GMEM 读带宽**（一次读服务两个 CTA），这正是 §2.2 打分模型里 `num_bytes_l2_ab` 用 `block_m/cluster_n` 而非 `block_m` 的原因。但**不省 SMEM**：`load_block_m = BLOCK_M`（§2.3），每个 CTA 仍存整份 A。也不省算力：两个 CTA 各发自己的 WGMMA，tensor core 负载与不开 multicast 时一样。

**与 SM100 2-CTA UMMA 的对照**（这是理解两代 cluster 协作的钥匙）：

| 维度 | SM90 multicast | SM100 2-CTA UMMA |
| --- | --- | --- |
| cluster 内两 CTA | 算**两个不同**输出块 | 合算**同一个**输出块（`UMMA_M=256`） |
| A 的 SMEM | 各存**整份**（`load_block_m = BLOCK_M`） | 各存**一半**（`load_block_m = BLOCK_M/cluster_n`） |
| tensor core | 各自发 `wgmma`（独立） | 两 SM 台成一条 `cta_group::2` UMMA |
| 省带宽 | ✓（一份读服务两 CTA） | ✓ |
| 省 SMEM | ✗ | ✓（stage 数可翻倍） |
| 能否运行期关闭 | ✓（`is_tma_multicast_valid`，§4.3） | ✗（硬件成对发射） |
| 累加器位置 | 各自的寄存器 | 各自的 TMEM（各存自己 128 行） |

情形 `kIsTMAMulticastOnA == false`（multicast B，`cluster_m = 2`，组在 M）角色互换：两 CTA 拿相邻 `m_block_idx`、共享同一份 B，A 各存自己的。

### 7.3 索引计算：`get_global_idx` 与「无 rank 偏移」

SM90 与 SM100 共用 `get_global_idx`（§4.4），四个索引的双模板开关语义一致：

```cpp
uint32_t m_idx   = scheduler.get_global_idx<(kGemmType == MGroupedMasked), MN>(shape_m, BLOCK_M, m_block_idx);
uint32_t n_idx   = scheduler.get_global_idx<(kMajorB == K),                MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
uint32_t k_a_idx = scheduler.get_global_idx<(kMajorA == MN),               K >(shape_k, BLOCK_K, k_block_idx, m_block_idx);
uint32_t k_b_idx = scheduler.get_global_idx<(kMajorB == MN),               K >(shape_k, BLOCK_K, k_block_idx, m_block_idx);
```

**关键差异：SM90 没有 SM100 那段「2-CTA 偏移」**。SM100 在算完 `m_idx/n_idx` 后会叠加 `block_rank_in_cluster() * load_block_m/n`（因为两 CTA 共算一块，各负责一半）；SM90 **完全不叠加**，因为两个 CTA 的 `m_block_idx/n_block_idx` 本就是调度器分好的**不同块**，各自的 `m_idx/n_idx` 已经是最终值。multicast 的「共享」只发生在 TMA 层（一份读写两处），不发生在索引层。

`n_idx` 的 `kWithGroupOffset = (kMajorB == K)`：B 的 group 维总是拼在外维（`make_tma_b_desc` 的 `gmem_outer_dim * num_groups`），K-major 时外维是 N 故加 `group * shape_n`；MN-major 时外维是 K，group 偏移由 `IndexType::K` 那条处理。`k_a_idx/k_b_idx` 的 `kWithGroupOffset = (major == MN)`：MN-major 时 K 是外维，k-grouped 的 `current_k_cumsum`/`current_k_start` 加在 K 上。

### 7.4 Stage 合并：用更大的 `BLOCK_K` 摊薄 `warpgroup_wait<0>`

第 46–57 行：

```cpp
// NOTES: this is for reducing the `warpgroup_wait<0>()` overhead
constexpr uint32_t kDoMergeStages =
    kNumStages_ >= 10 and kGemmType == GemmType::Normal and
    kMajorA == K and kMajorB == K and kNumMathThreads == 128;   // ← 比 SM100 多了这个条件
constexpr uint32_t kNumMinStages      = 5;
constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
constexpr uint32_t BLOCK_K            = BLOCK_K_ * kNumStagesPerMerge;   // 64 → 128/192
constexpr uint32_t kNumStages         = kNumStages_ / kNumStagesPerMerge;
```

动机（注释）：*reducing the `warpgroup_wait<0>()` overhead*。每个 k_block 末尾 math warpgroup 都要做一次 `warpgroup_commit_batch()` + `warpgroup_wait<0>()`（等本批全部 WGMMA 完成），这是 **math warpgroup 级的串行同步**。把 2–3 个 64-宽的 stage 合成 1 个 128/192-宽的 stage 后，同步次数减半/减三分之二、每次 WGMMA 连发数从 4 增到 8/12，而**总 SMEM 占用和流水深度（字节数）不变**。

两处与 SM100 的差异：

1. **多一个 `kNumMathThreads == 128` 条件**（即 `BLOCK_M ≤ 64`，单 math warpgroup）。SM100 无此限制。原因：单 math warpgroup 时并行度低、`warpgroup_wait<0>` 的串行开销更难被掩盖，合并收益最大；双 math warpgroup（`BLOCK_M > 64`）时寄存器压力已高，不再合并。
2. **`kNumMinStages = 5`（SM100 是 8）、触发阈值 `kNumStages_ ≥ 10`（SM100 是 ≥ 8）**。SM90 的 `kNumMaxStages = 16`（SM100 是 32），阈值相应下调。

合并后 SMEM 布局的关键点在三处保持一致（与 SM100 同构）：

1. **TMA**：`tma::copy<BLOCK_K=128, BLOCK_M, 128, bf16>` 内部 `BLOCK_INNER_ATOM = 128/2 = 64`，循环 2 次，第 i 次写到 `smem + i * BLOCK_M * 64`。
2. **WGMMA 描述符**：构造时用 `BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge = 64`（**不是** `BLOCK_K`，第 216 行），保证 `DG_STATIC_ASSERT(kSwizzleMode == BLOCK_ATOM_K * sizeof(dtype))` 即 `128 == 64×2` 仍成立；推进时 `atom_k_idx = k * WGMMA::K / BLOCK_ATOM_K`，偏移 `atom_k_idx * BLOCK_M * BLOCK_ATOM_K`。
3. **stage 步长**：`a_desc_lo` 用 `SMEM_A_SIZE_PER_STAGE`（按合并后的 `BLOCK_K` 算）作为 stage 间的步长（第 245 行）。

举例：`kNumStages_ = 10` → `kNumStagesPerMerge = 2`、`BLOCK_K = 128`、`kNumStages = 5`；每个 k_block 发 `128/16 = 8` 条 WGMMA，`atom_k_idx ∈ {0,0,0,0,1,1,1,1}`，`(k*16) % 64 ∈ {0,16,32,48,0,16,32,48}`。

### 7.5 `BLOCK_M < 64` 时的算力浪费（有意为之）

`WGMMA::M` 恒为 64，**与 `BLOCK_M` 无关**。当启发式因 `m ≤ 16`/`≤ 32` 选出 `BLOCK_M = 16`/`32` 时：

- WGMMA 仍按 M=64 发射，读 64 行 A（其中 48/32 行是 SMEM 越界垃圾，§5.5），往 128 个线程的寄存器写 64 行的 D；
- `kNumWGMMAStoreThreads = WAVE_BLOCK_M × (128/WGMMA::M) = BLOCK_M × 2`（因 `WAVE_BLOCK_M = BLOCK_M`），只有前 `kNumWGMMAStoreThreads/32` 个 warp 的结果会写回；
- 其余 warp 在 epilogue 入口 `if (not do_wgmma_store) continue;`（第 289–290 行）直接跳过，寄存器里的垃圾 D 被丢弃。

| `BLOCK_M` | `WGMMA_M_PER_WARP=16` 行/warp | `kNumWGMMAStoreThreads` | 参与 store 的 warp | 有效行 |
| --- | --- | --- | --- | --- |
| 16 | warp0→0–15 | 32 | 仅 w0 | 0–15（全部有效） |
| 32 | w0→0–15, w1→16–31 | 64 | w0–w1 | 0–31（全部有效） |
| 64 | w0–w3→0–63 | 128 | w0–w3 | 0–63 |

这是一个**明确的取舍**（与 SM100 §7.6 同理）：小 M 场景本来就是访存/延迟受限，选小 `BLOCK_M` 的目的正是注释里的 *avoid TMA L2 OOB bound*（不去 GMEM 白读 64 行），MMA 吞吐富余，用一条统一代码路径换掉「M=16/32 的另一套 WGMMA + 描述符/断言」的复杂度是划算的。

### 7.6 Tail-K：靠 TMA 零填充，无专用分支

**SM90 没有 SM100 那个 `kMayHaveTailKBlock` 编译期分支**。即便 K 是编译期常量，SM90 也不生成专门的 tail-K 代码，而是：

```cpp
const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);   // 向上取整
for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
    /* 每个 k_block 都发足 BLOCK_K/WGMMA::K 条 WGMMA，不区分是否尾块 */
}
```

正确性由两层硬件保证：

1. **TMA 零填充**：`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`（§2.6）使超出 `shape_k` 的元素在写入 SMEM 时自动填 0。尾块中不足 `BLOCK_K` 的部分被 0 填满。
2. **tx 计数不变**：尾块的 TMA 仍搬整个 box（零填充也算字节），所以 `arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)` 的期望值与实际到货字节始终匹配，barrier 协议无需特例。

于是尾块的 WGMMA 在零填充的 A/B 上照算，`0 × B = 0` 对累加器无贡献，D 结果正确。代价是尾块白发了一些「乘 0」的 WGMMA（最多浪费 `BLOCK_K-1` 个 K 元素的算力）。SM100 选择用 `for_each_static_prefix` 跳过这些无效 UMMA（省一点算力），SM90 选择不管（省一整块代码复杂度）——这是两代在「尾块处理」上的哲学差异。

> k-grouped 路径下 `current_k_start` 按 `kKAlignment=128` 对齐（§4.4），`BLOCK_K` 整除 128，所以尾块总是对齐的，`ceil_div` 不会真的向上取。

---

## 8. TMA 加载路径

### 8.1 调用形态

TMA warp（`kNumMathThreads/32 + 2` 的单个 lane）在每个 k_block 上按 `kMajorA/kMajorB` 四选一发射（第 186–197 行）：

```cpp
if constexpr (kMajorA == cute::UMMA::Major::K)
    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, bf16, kIsBatchedMM>(
        &tensor_map_a, &full_barrier, smem_a[stage_idx], k_a_idx, m_idx, num_tma_multicast_a, batch_idx);
if constexpr (kMajorA == cute::UMMA::Major::MN)
    tma::copy<BLOCK_M, BLOCK_K, kSwizzleAMode, bf16, kIsBatchedMM>(
        &tensor_map_a, &full_barrier, smem_a[stage_idx], m_idx, k_a_idx, num_tma_multicast_a, batch_idx);
// B 同理，num_tma_multicast_b
```

模板参数序是 `<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype, kIs3DTMA>`，函数参数序是 `(desc, barrier, smem_dst, inner_idx, outer_idx, num_multicast, batch_idx)`。**inner 恒为 SMEM 里连续的那一维**，所以 K-major 时 `(inner, outer) = (k, mn)`，MN-major 时 `(mn, k)`，两个 `if constexpr` 分支只是把实参顺序换了一下。与 SM100 唯一的形参差别：SM90 传的是 `num_tma_multicast_a/b`（每侧可能是 1 或 2，由 `is_tma_multicast_valid` 逐块决定，§4.3），SM100 传固定的 `kNumMulticast`。

### 8.2 swizzle-atom 循环

[common/tma_copy.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/tma_copy.cuh) 第 25–35 行：

```cpp
constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();
//   = kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t)
#pragma unroll
for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i)
    SM90_TMA_LOAD_2D::copy(desc_ptr, (uint64_t*)barrier_ptr, EVICT_NORMAL,
                           smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                           inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
```

TMA box 的内维被 host 压到 `swizzle/elem = 64` 个元素（128 B），所以一个逻辑上 `BLOCK_INNER` 宽的块要拆成 `BLOCK_INNER / 64` 条 TMA：

| 场景 | `BLOCK_INNER` | atom | TMA 条数 | SMEM 目的地址步进 |
| --- | --- | --- | --- | --- |
| K-major A，未合并（`BLOCK_K=64`） | 64 | 64 | 1 | — |
| K-major A，合并后（`BLOCK_K=128`） | 128 | 64 | 2 | `BLOCK_M * 64` |
| MN-major A，`BLOCK_M=128` | 128 | 64 | 2 | `BLOCK_K * 64` |

目的地址步进 `BLOCK_OUTER * BLOCK_INNER_ATOM` 正是 §5.3 描述的「atom 沿外维堆叠」布局，与 `make_gmma_desc` 的 SBO/LBO 推导严格对偶。

### 8.3 三种 TMA 变体（SM90 重点：rank0-only multicast）

```cpp
if (num_tma_multicast == 1) {
    cute::SM90_TMA_LOAD_2D::copy(...);                       // cp.async.bulk.tensor.2d…（单 CTA）
} else {
  #if __CUDA_ARCH__ >= 1000
    cute::SM100_TMA_2SM_LOAD_2D::copy(...);                  // 带 .cta_group::2（SM90 不走）
  #elif __CUDA_ARCH__ >= 900
    if (cute::block_rank_in_cluster() == 0)
        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(..., (1 << num_tma_multicast) - 1, ...);
  #endif
}
```

- **1-CTA**：`SM90_TMA_LOAD_2D`，Hopper 就有的 `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`。`num_tma_multicast == 1` 时（未开 cluster，或 `is_tma_multicast_valid` 逐块退回单播）走这里。
- **SM90 multicast**：`SM90_TMA_LOAD_MULTICAST_2D`，**只由 `block_rank_in_cluster() == 0` 发一条**带 CTA mask（`(1 << num_tma_multicast) - 1 = 0b11`）的 multicast load，一份 GMEM 读同时写进 cluster 内所有 CTA 的 SMEM，tx 信号也同时打到每个 CTA 的同名 mbarrier。rank 1 根本不发 A 的 multicast（被 `if (block_rank_in_cluster() == 0)` 拦下），但仍会在自己的 `full_barrier` 上收到 tx。
- **SM100 2-CTA**：`SM100_TMA_2SM_LOAD_2D`（带 `.cta_group::2`）在 `__CUDA_ARCH__ >= 1000` 才编译，SM90 cubin 里不存在。两者语义不同：SM100 是「两 CTA 各自发射、tx 只记到 leader」；SM90 是「只 rank0 发射、tx 记到所有接收方」。

3D 变体（Batched）同理：`SM90_TMA_LOAD_3D` / `SM90_TMA_LOAD_MULTICAST_3D`，多一个 `batch_idx` 坐标。Cache hint 统一 `EVICT_NORMAL`，开头有一条静态断言确保 SM90/SM100 两套枚举值一致。

### 8.4 Descriptor prefetch

```cpp
if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {   // TMA warpgroup 的第一个 warp，单 lane
    cute::prefetch_tma_descriptor(&tensor_map_a);
    cute::prefetch_tma_descriptor(&tensor_map_b);
    cute::prefetch_tma_descriptor(&tensor_map_cd);
}
__syncwarp();
```

在 kernel 最开头、**任何同步之前**执行。descriptor 在常量内存里，首次 TMA 访问会有冷启动延迟，提前 prefetch 可以把它藏进 barrier 初始化的时间里。与 SM100 的差别：SM100 用 `warp_idx == 0` 全体（`prefetch.tensormap` 不是单线程指令），SM90 用 TMA warpgroup 首 warp 的**单个 elected lane**（`elect_one_sync()`）——因为 SM90 把 math warpgroup 排在前面（warp 0..kNumMathThreads/32-1），TMA 角色全在后面的 warpgroup，prefetch 自然也用 TMA warpgroup 的 warp。

### 8.5 `expect_tx` 字节数

```cpp
full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);   // 第 198 行
```

- `SMEM_*_SIZE_PER_STAGE` 用的是**合并后**的 `BLOCK_K`，与 TMA 实际搬运的字节数一致。
- **不乘 `kNumTMAMulticast`**（对比 SM100 乘）：因为 SM90 每个 CTA 的 `full_barrier` 只统计**本 CTA 收到的字节**——multicast 的 A 字节已由硬件直接送到本 CTA 的 barrier，自己的 B 字节由自己的 load 送。两者相加恰好是一个 stage 的 A+B。
- `full_barriers[i]->init(1)`（§5.2）：只有**一次** arrive（就是这条 `arrive_and_expect_tx`），由本 CTA 的单个 TMA lane 发出。
- **顺序**：TMA 先发射（第 186–197 行），`arrive_and_expect_tx` 后执行（第 198 行）。合法——mbarrier 的 tx-count 是「期望值累加」，只要在该相位完成前把期望值补上即可。

### 8.6 完整的 TMA warp 循环

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);   // 逐块判定
    const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
    const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
    const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
        empty_barriers[stage_idx]->wait(phase ^ 1);      // ① 等消费者释放
        /* ② 算 m_idx / n_idx / k_a_idx / k_b_idx / batch_idx（无 rank 偏移） */
        /* ③ 发 A、B 的 TMA（各 1~2 条指令；multicast 侧只 rank0 发） */
        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);   // ④
    }
}
// 退出前（仅 multicast）：再等一轮 empty，确保 peer 不会再远程 arrive
if constexpr (kNumTMAMulticast > 1)
    for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
        empty_barriers[stage_idx]->wait(phase ^ 1);
```

整个 TMA warp 就是这四步的无限重复，没有任何计算。它的推进速度只受 `empty_barriers` 的释放节奏限制，因此可以超前 WGMMA 多达 `kNumStages` 个 k_block。末尾那段「额外一轮 empty wait」（第 202–206 行）是 multicast 的**退出协议**：peer CTA 的 math warp 会通过 `arrive(target_cta)` 远程写本 CTA 的 `empty_barriers`，本 CTA 必须等这些远程 arrive 全部落地后才能退出（否则 SMEM 释放后 peer 的远程 arrive 是非法访问），详见 §10.6。

---

## 9. WGMMA 发射路径

> 本章对应 SM100 文档的「§9 UMMA 发射路径」。两者都用 SMEM 描述符（`_SS`），但 SM90 是 **warpgroup 级协同的 `wgmma.mma_async`**，累加器在寄存器；SM100 是 **单线程的 `tcgen05.mma`**，累加器在 TMEM。

### 9.1 WGMMA 类型选择

```cpp
using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, kMajorA, kMajorB>::type;
```

`BF16MMASelector`（[mma/sm90.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/mma/sm90.cuh) 第 101–148 行）把 `BLOCK_N` 映射到一个 `BF16MMA<N, MMA_64xNx16_F32BF16BF16_SS<majorA, majorB>>`，`N` 从 8 到 256、步进 8，共 32 个**硬编码的 C++ 类型特化**。每个特化的静态常量：

```cpp
static constexpr int M = 64;                 // Hopper wgmma 的 M 原子固定 64
static constexpr int N = N_;                 // = BLOCK_N
static constexpr int K = 16;                 // BF16 的 wgmma K 原子固定 16
static constexpr int kNumAccum = M * N / 128; // = BLOCK_N / 2
```

**与 SM100 的根本差异**：SM100 把 MMA 形状编成一个运行期可改的 `runtime_instr_desc`（32-bit），`UMMA_N` 甚至能逐块改写（swap-AB）；SM90 把形状直接固化成**编译期的 C++ 类型**，每个 `BLOCK_N` 对应一个不同的 `MMA_64xNx16_..._SS` 类，展展成一条固定的 `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16` PTX。所以 SM90 的 `BLOCK_N` 必须是编译期常量（它是模板参），不能运行期变——这也是为什么 SM90 没有 swap-AB（swap-AB 需要运行期改 `UMMA_N`，而 wgmma 的 N 固定）。

`_SS` 后缀 = A、B 都来自 Shared memory 描述符（对比 `_RS` 变体 A 来自寄存器）。BF16 GEMM 永远走 `_SS`。

### 9.2 SMEM 描述符（`GmmaDescriptor`）

`make_gmma_desc()`（[mma/sm90.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/mma/sm90.cuh) 第 243–279 行）→ `make_smem_desc()` 填充 `cute::GmmaDescriptor` 的位域：

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `start_address_` | `__cvta_generic_to_shared(p) >> 4` | SMEM 地址，**16 B 为单位**（bit 0–13） |
| `layout_type_` | `to_gmma_layout_type<...>()` | `INTERLEAVE`（swizzle 0/16）/ `B32` / `B64` / `B128` |
| `leading_byte_offset_` (LBO) | `lbo >> 4` | atom 间在某一维的字节步长（bit 16–29） |
| `stride_byte_offset_` (SBO) | `sbo >> 4` | atom 间在另一维的字节步长（高 32 位） |
| `base_offset_` | `0` | — |

**K-major** 的 SBO/LBO 推导：

```cpp
DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");   // 128 == 64 * 2
const uint32_t stride_byte_offset  = num_non_contiguous * BLOCK_K * sizeof(dtype_t);   // 8*64*2 = 1024
const uint32_t leading_byte_offset = 0;
```

注释解释为什么 LBO 是 0：*on K, there is only 1 atom as asserted previously*——那条静态断言保证每个 block 在 K 方向恰好只有一个 swizzle atom，于是「K 方向 atom 间步长」无意义。SBO = 一个 atom 的字节大小 = `8 行 × 128 B = 1024 B`，即 MN 方向相邻 atom 的步长。`num_non_contiguous = 128 / 16 = 8`（常量，**无 SM100 那个 base32 特例**，BF16 永走 8）。

**MN-major**：

```cpp
constexpr uint32_t BLOCK_MN_ATOM = get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();  // 64
DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);        // 不允许 atom 内的 MN 偏移
uint32_t stride_byte_offset  = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);   // 8*64*2 = 1024
uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
if constexpr (kSwizzleMode == 16) math::swap(stride_byte_offset, leading_byte_offset);
```

语义约定：swizzle 时 `{SBO, LBO}` 是 atom 在 `{K, MN}` 上的步长；非 swizzle（`kSwizzleMode == 16`，*means non-swizzling but interleaving*）时是 `{MN, K}`，所以要 swap。

### 9.3 单个 `a_desc_lo`/`b_desc_lo` + 算术推进（对照 SM100 的 32-lane）

这是 SM90 与 SM100 描述符管理最直观的差别。math warpgroup 分支第 217–220 行：

```cpp
constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
auto a_desc = make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], math_wg_idx * WGMMA::M, 0);
auto b_desc = make_gmma_desc<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);   // 从 lane 0 广播
const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
```

然后在 K 循环内，每个 stage 的基地址用**算术加**得到（第 245–246 行）：

```cpp
const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);
```

| | SM90（本文） | SM100 |
| --- | --- | --- |
| per-stage 描述符存储 | **单个** `a_desc_lo`（lane0 广播），每 stage 加常量步长 | **32 个 lane** 各存一个 stage 的 `a_desc.lo`，`__shfl_sync(..., stage_idx)` 取用 |
| stage 上限 | 无硬约束（算术加，`kNumMaxStages=16` 是 SMEM 预算定的） | `kNumStages <= 32`（lane 数硬约束） |
| 为何可行 | 所有 stage 连续且等大，`start_address` 随 stage 线性递增 | 同左，但选择用 lane 寄存器存而非算术加 |

`__shfl_sync(0xffffffff, a_desc.reg32_[0], 0)` 把 lane 0 的值广播到全 warp。注释点明目的：*use `__shfl_sync` to encourage NVCC to use unified registers*——因为 `make_gmma_desc` 的输入（`smem_a[0]`、`math_wg_idx * 64`）本就是 warp-uniform 的，所有 lane 算出的 `a_desc` 相同，这次 shfl 在值上是恒等的，但向编译器**声明了“这个值是 warp 统一的”**，从而把它放进统一寄存器（uniform register），降低每-lane 寄存器压力。

注意只广播/存了 `.reg32_[0]`（低 32 位，含 `start_address_` 与 LBO），而 `.reg32_[1]`（高 32 位，含 SBO / base_offset / layout_type）是 **stage/k 无关的常量**，保留在 `a_desc` 结构体里不变。循环里只改 `.reg32_[0]`。

### 9.4 K 内层展开与描述符推进

```cpp
for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {   // wave
    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {                            // 4 条（未合并）
        const uint32_t atom_k_idx = k * WGMMA::K / BLOCK_ATOM_K;
        a_desc.reg32_[0] = advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, bf16>(
            a_desc_base_lo, local_idx * WAVE_BLOCK_M, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
        b_desc.reg32_[0] = advance_gmma_desc_lo<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, bf16>(
            b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
    }
}
```

`advance_gmma_desc_lo` 的算式（第 237–241 行）：

```cpp
return base + (((offset + mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t)) >> 4u);
// stride_k = (major == K) ? 1 : get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>()
// 注：模板形参名 `BLOCK_K` 在调用处实例化为 `BLOCK_ATOM_K`
```

- **K-major**：`mn_idx * BLOCK_ATOM_K` 是行偏移（每行 `BLOCK_ATOM_K` 个 K 元素连续），`k_idx * 1` 是行内 K 偏移，`offset = atom_k_idx * BLOCK_M * BLOCK_ATOM_K` 是跳到第 `atom_k_idx` 个 K-atom（仅合并时 > 0）。合计 `× 2 B >> 4` 换成 16-B 单位。
- **MN-major**：`stride_k = BLOCK_MN_ATOM = 64`，因为 MN-major 下 K 是**外维**，K 前进 1 要跨过一整个 64 宽的 MN atom。

举例（未合并、`BLOCK_M=128`、`BLOCK_ATOM_K=64`、K-major、`local_idx=0`）：`k ∈ {0,1,2,3}`，`k_idx = (k*16)%64 ∈ {0,16,32,48}`，`atom_k_idx = 0`，于是 `a_desc.reg32_[0]` 依次 `+0, +2, +4, +6`（每条 WGMMA 消耗 `16 × 2 B = 32 B = 2` 个 16-B 单位）。与 SM100 的 `{0,2,4,6}` 完全一致。

因为 `local_idx`、`k`、`atom_k_idx`、`k_idx` 全是编译期常量（`#pragma unroll` + 常量 `BLOCK_*`），`advance_gmma_desc_lo` 的返回值被常量折叠，每条 WGMMA 的描述符增量是**立即数**（一条 IADD3）。

### 9.5 fence / arrive / commit / wait 序列

一个 k_block 内的 WGMMA 发射被四条指令包住（第 251–276 行），[ptx/wgmma.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/wgmma.cuh)：

```cpp
full_barriers[stage_idx]->wait(phase);                       // ① 等 TMA 到货（A/B 已入 SMEM）
#pragma unroll
for (i = 0; i < kNumAccum*waves; ++i) ptx::warpgroup_fence_operand(accum[i]);   // ②
ptx::warpgroup_arrive();                                     // ③ wgmma.fence.sync.aligned
for (local_idx) for (k) { /* 改 desc */ WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1); }   // ④
ptx::warpgroup_commit_batch();                               // ⑤ wgmma.commit_group.sync.aligned
#pragma unroll
for (i = 0; i < kNumAccum*waves; ++i) ptx::warpgroup_fence_operand(accum[i]);   // ⑥
ptx::warpgroup_wait<0>();                                    // ⑦ wgmma.wait_group.sync.aligned 0
empty_barrier_arrive(stage_idx);                             // ⑧ 通知 TMA：本 stage 读完
```

| 步 | PTX | 作用 |
| --- | --- | --- |
| ②⑥ | `asm volatile("" : "+f"(reg))` | **编译器屏障**（无指令）：把累加器寄存器标为读+写，防止编译器把 `accum` 的初始化/读取跨越 wgmma fence/commit/wait 重排（wgmma 异步读写累加器，编译器看不到） |
| ③ | `wgmma.fence.sync.aligned` | 把 fence 之前的 SMEM 读（full 等到后 A/B 可见）与累加器状态排序进 async wgmma proxy；标志「进入 wgmma 区」 |
| ⑤ | `wgmma.commit_group.sync.aligned` | 把此前发的所有 `wgmma.mma_async` 打包成一个 commit group，供 wait 等待 |
| ⑦ | `wgmma.wait_group.sync.aligned 0` | **阻塞**直到本 group 全部 wgmma 完成（结果落到累加器寄存器）。`<0>` = 等全部 |

**为何每 k_block 都 `wait<0>`（而不是更深流水）**：`wgmma.mma_async` 从 SMEM 异步读 A/B，必须等它读完才能通过 `empty_barrier_arrive` 释放这个 SMEM stage（否则 TMA 可能在 wgmma 还在读时覆写 A/B）。所以顺序是 `wait<0>` → `empty_arrive`，每 k_block 一次。这就是 §1.1 说的「MMA **半同步**」：math warpgroup 在每个 k_block 的 `wait<0>` 处阻塞，暴露该 k_block 的 MMA 完成延迟（同一 group 内的 4 条 wgmma 彼此在 tensor pipe 里重叠，但跨 k_block 不重叠）。真正被重叠掉的是 **SMEM 预取**（TMA 超前 `kNumStages` 个 k_block，§10.5）。Stage 合并（§7.4）把每次 `wait<0>` 覆盖的 wgmma 从 4 增到 8/12、同步次数减半，正是为了摊薄这个半同步开销。

对照 SM100：`tcgen05.commit` 让 empty_barrier **异步跟踪** MMA 完成，MMA warp 不阻塞（发射完就继续）；且 SM100 的累加器在 TMEM，SMEM stage 的释放与 MMA 完成解耦。SM90 因累加器在寄存器 + `wait<0>` 的阻塞语义，必须同步等待。

### 9.6 math warpgroup 的完整循环

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    float accum[kNumAccum * waves] = {0};                        // ① 本块累加器清零
    auto empty_barrier_arrive = [&](uint32_t s) { ... };         // ② 定义 arrive lambda
    const auto num_total_k_blocks = ceil_div(current_shape_k, BLOCK_K);
    for (k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
        a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE/16);   // ③ 算术推 stage
        full_barriers[stage_idx]->wait(phase);                    // ④ 等 TMA
        /* ⑤ fence_operand → warpgroup_arrive → waves×4 条 wgmma → commit → fence_operand → wait<0> */
        empty_barrier_arrive(stage_idx);                          // ⑥ 释放 stage
    }
    if (not do_wgmma_store) continue;                             // ⑦ BLOCK_M<64 时无效 warp 跳过
    /* ⑧ epilogue：tma_store_wait<0> → NamedBarrier → STSM/st.shared → tma_store_fence → TMA store（§11） */
}
```

注意 epilogue（⑧）**在同一个 math warpgroup 的同一个块循环内**，紧接 K 循环之后——这就是 §1.1「compute 与 epilogue 串行」的代码体现。一个输出块的 K 循环跑完（累加器已全部在寄存器），同一批线程立刻做 epilogue 把它们搬出；期间不发新 wgmma。下一块的 K 循环要等 epilogue 结束才开始（累加器数组要重置 `{0}`）。SM100 则是 MMA warp 与 epilogue warp 分离、TMEM 双缓冲重叠，两者调度形态截然不同。

---

## 10. 生产者-消费者同步

### 10.1 barrier 全景（只有两类）

全部同步对象都是 `cutlass::arch::ClusterTransactionBarrier`（8 B SMEM 驻留的 mbarrier），**共 2 类**（SM100 是 5 类）：

| barrier | 个数 | `init` 计数 | 等待者 | 到达者 | 语义 |
| --- | --- | --- | --- | --- | --- |
| `full_barriers[s]` | `kNumStages` | `1` | math warpgroup（每 CTA 全体 math 线程） | TMA warp 单 lane 的 `arrive_and_expect_tx(A+B)` + TMA 硬件 tx 完成 | SMEM stage `s` 已装好，A/B 可读 |
| `empty_barriers[s]` | `kNumStages` | `kNumTMAMulticast × kNumMathThreads/32` | TMA warp（每 CTA 单 lane） | 每个 math warp 的 `empty_barrier_arrive`（multicast 时跨 CTA） | stage `s` 已被 WGMMA 读完，可覆写 |

只有一条流水（SM100 有三条）：

```
【A/B SMEM 环】  生产者 = TMA warp        消费者 = math warpgroup
     full  : TMA → math        empty : math → TMA

【累加器】  无环——每块就地 {0} 重置的寄存器数组，compute 与 epilogue 串行（§9.6）
【D SMEM】  单缓冲，不用 mbarrier，靠 tma_store_wait<0> + NamedBarrier 串行复用（§11.4）
```

**SM90 没有 SM100 的 TMEM 环（`tmem_full`/`tmem_empty`）**——因为累加器不是跨块复用的共享资源，而是每块私有的寄存器数组，compute→epilogue 在同一批线程内串行，不需要生产-消费握手。这是 SM90 同步结构比 SM100 简单得多的根本原因。

math warpgroup 既是 A/B 环的消费者，又（在 epilogue 里）是 D 单缓冲的生产者，是整条链的串行点。但它用整个 warpgroup（128/256 线程）而非 SM100 的单 warp：wgmma 是 warpgroup 协同指令（§3.3），且 epilogue 要把寄存器里的累加器搬进 SMEM，也需要这么多线程。

### 10.2 parity 相位约定与「首轮免等」

`Barrier::wait(p)` 对应 `mbarrier.try_wait.parity.shared::cta.b64 P, [bar], p`，含义是「等待奇偶性为 `p` 的那个相位**完成**」。A/B 环用显式相位变量（§4.5）：

```cpp
uint32_t stage_idx = 0, phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++ k_block_idx;
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;          // 只在回绕到 stage 0 时翻转
};
// 生产者（TMA） ：empty_barriers[stage_idx]->wait(phase ^ 1);
// 消费者（math）：full_barriers[stage_idx]->wait(phase);
```

第一轮（`phase = 0`）：

- TMA 等 `empty` 的 parity **1**。新建的 mbarrier 处于 phase 0 且未完成，对 parity 1 的 `try_wait` 立即成功——这就是「缓冲初始为空，生产者前 `kNumStages` 轮直接穿过」的标准手法。
- math 等 `full` 的 parity **0**，必须等第一次 tx 完成。

回绕一次后 `phase = 1`，两边等待的 parity 同时翻转，协议自洽。**SM90 没有 SM100 的「TMEM 环从 `current_iter` 推导 phase」那一套**（§10.1），因为无 TMEM 环；`current_iter` 在 SM90 只用于算块序号，不参与任何 barrier 相位。

### 10.3 arrive 计数逐条推导

**`full_barriers[s]->init(1)`**

- 每个 CTA 的 TMA warp 单 lane 发一次 `arrive_and_expect_tx(A+B)` → 本地 arrive 1 次 + 设置期望 tx。这就是 `init(1)` 的那 1 次。
- tx 字节来自：multicast 的 A（rank0 一条 load 写两个 CTA，tx 同时打到两 CTA 的 barrier）+ 自己的 B。
- **对比 SM100**：SM100 `init(kNumMulticast)`（两 CTA 各 arrive 一次到 leader）；SM90 `init(1)`（每 CTA 自己的 barrier 只被自己 arrive，multicast 的 A 由硬件送 tx 而非送 arrive）。

**`empty_barriers[s]->init(kNumTMAMulticast × kNumMathThreads/32)`**

`kNumMathThreads/32` = math warp 数（4 或 8）。`empty_barrier_arrive`（第 233–240 行）：

```cpp
auto empty_barrier_arrive = [&](uint32_t s) {
    if constexpr (kNumTMAMulticast == 1) {
        lane_idx == 0 ? empty_barriers[s]->arrive() : void();      // 每 warp 1 次，共 math_warps 次
    } else {
        auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
        lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();   // 每 warp 2 次，分投两 CTA
    }
};
```

- **`kNumTMAMulticast == 1`**：每个 math warp 的 lane 0 arrive 一次，单 CTA 共 `kNumMathThreads/32` 次 → `init(1 × kNumMathThreads/32)`。✓
- **`kNumTMAMulticast == 2`**：每个 math warp 的 lane 0/1 各 arrive 一次，`target_cta = lane_idx` 即 lane0→CTA0、lane1→CTA1。于是每个 CTA 的 `empty_barriers[s]` 收到：本 CTA 所有 math warp 投向自己的 1 次 + peer CTA 所有 math warp 投向自己的 1 次 = `2 × kNumMathThreads/32` → `init(2 × kNumMathThreads/32)`。✓
- **为何 multicast 时 peer 的 math warp 要 arrive 本 CTA 的 empty**：multicast-on-A 时，本 CTA 的 `smem_a[s]` 是 rank0 一条 multicast load 写的（两个 CTA 各一份副本）。rank0 的 TMA warp 要重发下一个 multicast、覆写两个 CTA 的 `smem_a[s]`，必须知道**两个 CTA 的 math warp 都读完了各自的副本**。所以两 CTA 的 math warp 都向两 CTA 的 empty arrive。
- **`is_peer_cta_alive == false` 的回退**（§4.3）：peer CTA 因落在矩阵边界外而无有效块时，两个 lane 都投向**本 CTA**（`block_rank_in_cluster()`）。于是本 CTA 的 empty 从自己的 math warp 收到 2 次/warp = `2 × kNumMathThreads/32`，恰好凑足 `init` 计数（peer 贡献 0，但 peer 已退出、不等自己的 empty）。

`arrive(target_cta)` 底层是 `mapa.shared::cluster` 把地址映射到目标 CTA 后 `mbarrier.arrive.shared::cluster`，即跨 CTA 远程 arrive（[ptx/ld_st.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/ld_st.cuh) 第 156–160 行的 `mapa_shared`）。

### 10.4 fence 序列

SM90 需要两类 fence：mbarrier 初始化可见性（async proxy）与 wgmma 的累加器/排序。共四处：

| 位置 | 指令 | 作用 |
| --- | --- | --- |
| barrier 初始化后（第 121 行） | `fence_barrier_init()` = `fence.mbarrier_init.release.cluster` | 让 init 后的 mbarrier 对 **async proxy**（TMA）可见，cluster 作用域保证 peer CTA 也看得到 |
| wgmma 前（第 255 行） | `warpgroup_arrive()` = `wgmma.fence.sync.aligned` | 把此前 SMEM 读（full 等到后 A/B 可见）与累加器状态排序进 async wgmma proxy |
| 累加器读写两侧（第 254/272 行） | `warpgroup_fence_operand(reg)` = `asm("" : "+f"(reg))` | 编译器屏障，防止累加器访问跨越 wgmma fence/commit/wait 重排（§9.5） |
| STSM 写完 SMEM、TMA store 前（第 360 行） | `tma_store_fence()` = `fence.proxy.async.shared::cta` | generic proxy（stmatrix/st.shared）→ async proxy（TMA）的 SMEM 可见性 |

**SM90 不需要 SM100 的 `tcgen05.fence::before/after_thread_sync` 与 `tcgen05.wait::ld`**——那些都是为 TMEM 异步读写服务的，SM90 无 TMEM。SM90 独有 `wgmma.fence/commit_group/wait_group` 三件套（为寄存器累加器的异步 wgmma 服务）与 `warpgroup_fence_operand`（编译器屏障）。两代的 fence 集合恰好互补，各自对应自己的异步代理（SM90：TMA + wgmma；SM100：TMA + tcgen05）。

### 10.5 稳态时序（一个输出块的完整生命周期）

以 `kNumStages = 6`、`num_total_k_blocks = 128`（K=8192, BLOCK_K=64）、`BLOCK_M=128`（2 个 math warpgroup）为例，纵轴是时间：

```
TMA warp  │ empty.wait │ TMA(s=0) │ empty.wait │ TMA(s=1) │ … │ TMA(s=5) │ empty.wait(阻塞) │ TMA(s=0) │ …
              │ expect_tx │              │ expect_tx │                                        ▲
              ▼            ▼              ▼            ▼                                        │
full[0] ────●───────────────────────────────────────────────────┐      │
full[1] ──────────●──────────────────────────────────────────┼─┐    │
                 │ wait(phase=0)                                                    │ │    │
math wg   │ full[0].wait │ fence→4×wgmma→commit→wait<0>→empty[0] │ full[1].wait │ 4×wgmma │ … │ (K循环完) │ EPILOGUE │
                                                                                                    ▼
                                                              tma_store_wait<0> → NamedBarrier
                                                              → STSM/st.shared 写 smem_d
                                                              → tma_store_fence → NamedBarrier → TMA store
```

两个重叠关系（比 SM100 少一个）：

1. **TMA 超前 math 最多 `kNumStages` 个 k_block**（受 A/B 环深度限）。epilogue 期间 TMA 仍在为下一块预取。
2. **compute 与 epilogue 不重叠**（同一批 math 线程串行，§9.6）。这是 SM90 相对 SM100 缺的一个重叠：SM100 的块 i epilogue 与块 i+1 的全部 MMA 重叠（TMEM 双缓冲），SM90 的块 i epilogue 期间 math warpgroup 不算，只能靠 TMA 预取把下一块的 A/B 先搬好。
3. **D 单缓冲不重叠**：下一块的 epilogue 必须等上一块 TMA store 把 `smem_d` 读完（`tma_store_wait<0>`，§11.4）。

### 10.6 退出协议（TMA drain，无最终 cluster sync）

SM90 的退出比 SM100 简单：**没有最终的 `cluster_sync`，也没有 TMEM free**。唯一的安全网是 TMA warp 在 multicast 时的「额外一轮 empty wait」（第 202–206 行）：

```cpp
// To safely deconstruct distributed shared barriers, we need another round of empty waits
if constexpr (kNumTMAMulticast > 1) {
    for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
        empty_barriers[stage_idx]->wait(phase ^ 1);
}
```

危险场景：multicast 时 peer CTA 的 math warp 通过 `arrive(target_cta)` **远程**写本 CTA SMEM 里的 `empty_barriers`。如果本 CTA 的 TMA warp 已跑完并退出（SMEM 释放），peer 那次远程 arrive 就是一次非法访问。因为 TMA 超前 math 最多 `kNumStages` 个 k_block，TMA 跑完所有块时 math 可能还有至多 `kNumStages` 个 k_block 的 empty arrive 未发出；多等一轮 `kNumStages` 次 empty，就把这些尾巴的远程 arrive 全部排空（drain），确保退出后不会再有人写本 CTA 的 barrier。

这与 SM100 退出前的「额外一轮 `tmem_empty_barriers` wait」同构，只是 SM90 drain 的是 A/B 环的 empty（唯一会被远程 arrive 的 barrier），SM100 drain 的是 TMEM 环的 tmem_empty。SM90 不需要 SM100 那次最终 `cluster_sync_with_relaxed_arrive()`，因为没有 TMEM 需要两 CTA 对齐后释放；prologue 的 cluster sync（第 125 行）已保证 barrier 初始化互见。

---

## 11. Epilogue

### 11.1 入口：与 math warpgroup 同一批线程

SM90 的 epilogue **不是独立的 warp 分支**，而是接在 math warpgroup 的 K 循环之后（第 279–383 行），由同一批线程串行完成（§9.6）。入口先做三个编译期检查与两个门控：

```cpp
constexpr uint32_t kNumElemBytes   = sizeof(nv_bfloat16);
constexpr uint32_t TMA_D_BLOCK_N   = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);   // bf16→64, fp32→BLOCK_N
constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;   // = 16
DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32, "...");
DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

if (not do_wgmma_store) continue;                    // BLOCK_M<64 时，无效 warp 直接跳过（§7.5）

if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)           // 只由将发 TMA store 的线程等
    cute::tma_store_wait<0>();                       // 等上一块的 TMA store 读完 smem_d（§11.4）
cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);   // 广播“smem_d 已释放”（§12.3）
```

`WGMMA_M_PER_WARP = 16`：一个 `m64` WGMMA 的 64 行按 4 个 warp 平分，每 warp 16 行。epilogue 里 `warp_idx * WGMMA_M_PER_WARP` 把每个 warp 定位到自己那 16 行。

### 11.2 BF16：STSM 写回 + swizzle 逐行推导

BF16 输出走 `stmatrix`（第 297–344 行）：

```cpp
DG_STATIC_ASSERT(kSwizzleDMode > 0, "Invalid swizzling type");
DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {   // wave
    auto m_offset = local_idx * WAVE_BLOCK_M;
    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
    for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {                            // = BLOCK_N/8 次
        uint8_t* smem_ptr = /* swizzle 地址计算，见下 */;
        // NOTES: only 16 lanes' addresses are used
        ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
            __float22bfloat162_rn({shifted_accum[i*4+0], shifted_accum[i*4+1]}),   // 2 fp32 → 1 bf162
            __float22bfloat162_rn({shifted_accum[i*4+2], shifted_accum[i*4+3]}),
            smem_ptr);
    }
}
```

swizzle 地址计算（`kSwizzleDMode > 0`）：

```cpp
constexpr uint32_t kNumBankGroupBytes = 16;
auto atom_offset    = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;   // 128B swizzle → true
auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
auto col = kHasShortcut ? (in_atom_offset)                : (bank_group_index % 8);
col ^= row % (kSwizzleDMode / 16);                                          // col ^= row % 8
smem_ptr = (uint8_t*)smem_d +
    warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +   // Warp 偏移（每 warp 16 行）
    m_offset * kSwizzleDMode +                        // Wave 偏移
    atom_offset * BLOCK_M * kSwizzleDMode +           // Swizzle atom 偏移（n 方向第几个 atom）
    row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;   // atom 内偏移
```

这与 SM100 §11.2 的 bank-group 置换逻辑**同构**（`col ^= row % 8` 把同列的8 行打散到 8 个物理 bank group，消除 bank conflict），差异在于：

1. **写回指令是 STSM 而非 `st.shared`**：`SM90_U32x2_STSM_N`（[ptx/ld_st.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/ld_st.cuh) 第 43–52 行）= `stmatrix.sync.aligned.x2.m8n8.shared.b16`，一条写 2 个 8×8 bf16 矩阵（= 4 个 bf16/lane）。注释 *only 16 lanes' addresses are used*：`stmatrix.x2` 只用 lane 0–15 提供的 16 个行地址（两个 8×8 = 16 行），lane 16–31 的地址被忽略。
2. **无需 TMEM load**：SM100 要先 `tcgen05.ld` 把累加器从 TMEM 读到寄存器、再 `st.shared`；SM90 的累加器**本就在寄存器**，直接 `__float22bfloat162_rn` 打包后 STSM。少一层搬运、少一个异步等待（无 `fence_view_async_tmem_load`）。
3. **循环次数**：`i` 跑 `kNumAccum/4 = BLOCK_N/8` 次（每次 STSM 消耗 4 个累加器），恰好覆盖本线程的 `kNumAccum` 个寄存器。

### 11.3 FP32：`st.shared` 直存

FP32 输出时 `kSwizzleDMode == 0`（§2.3），不走 STSM（`stmatrix.b16` 只适用于 16-bit），而是逐行 `st.shared.v2.f32`（第 345–359 行）：

```cpp
for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
    auto m_offset = local_idx * WAVE_BLOCK_M;
    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
    auto smem_d_0 = (float2*)(smem_d + (m_offset + warp_idx*WGMMA_M_PER_WARP + lane_idx/4 + 0) * BLOCK_N + (lane_idx%4)*2);
    auto smem_d_1 = (float2*)(smem_d + (m_offset + warp_idx*WGMMA_M_PER_WARP + lane_idx/4 + 8) * BLOCK_N + (lane_idx%4)*2);
    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
        ptx::st_shared(smem_d_0 + i*4, make_float2(shifted_accum[i*4+0], shifted_accum[i*4+1]));   // 行 lane/4
        ptx::st_shared(smem_d_1 + i*4, make_float2(shifted_accum[i*4+2], shifted_accum[i*4+3]));   // 行 lane/4+8
    }
}
```

这正是 `wgmma` 累加器 fragment 的标准布局：每 lane 持 `kNumAccum = N/2` 个值，对应两行（`lane/4` 与 `lane/4+8`，在本 warp 的 16 行内），列 `(lane%4)*2 + i*8 + {0,1}`。`smem_d_0 + i*4`（float2*）= `+i*8` 个 float，与列偏移一致。不 swizzle，D 就是行主序的 `BLOCK_M × BLOCK_N` 个 float，`TMA_D_BLOCK_N = BLOCK_N`（一条 TMA store 覆盖整块）。

### 11.4 单缓冲 D 的串行化：`tma_store_wait<0>`

SM90 的 D 是**单缓冲**（§5.4），下一块的 epilogue 必须等上一块的 TMA store 把 `smem_d` 读完才能覆写：

```cpp
if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) cute::tma_store_wait<0>();   // = cp.async.bulk.wait_group.read 0
cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
```

- **`wait<0>`（而非 SM100 的 `wait<1>`）**：`.read 0` = 等到「未完成的 TMA store group 数 ≤ 0」，即上一块的 store 全部把 SMEM 读完。SM100 用双缓冲 + `wait<kNumTMAStoreStages-1> = wait<1>`，允许一个 store 在飞；SM90 单缓冲只能 `wait<0>` 完全排空。
- **`.read` 后缀的语义**：cute 的 `tma_store_wait<N>()` 展开为 `cp.async.bulk.wait_group.read N`，只等到「TMA 引擎不再需要读这块 SMEM」，**不**等数据真正落到 GMEM。这正是覆写 `smem_d` 所需的最弱条件。
- **只部分线程等**：`cp.async.bulk.wait_group` 是**每线程**计数，而 `commit_group`（`tma_store_arrive`）只由 `threadIdx.x < BLOCK_N/TMA_D_BLOCK_N` 的线程发出（§11.5），所以只有它们持有非零的 group 计数、也只有它们能等。随后 `NamedBarrier` 把「smem_d 已释放」广播给全体 store 线程。

这是 SM90 epilogue 相对 SM100 的一个性能损失：单缓冲使块的 store 与下一块的 compute 无法在 D 侧重叠（§10.5）。

### 11.5 TMA store / reduce.add

写完 SMEM、`tma_store_fence()` + `NamedBarrier` 后，前 `BLOCK_N / TMA_D_BLOCK_N` 个线程各发一条 TMA store（第 363–382 行）：

```cpp
const auto m_idx = scheduler.get_global_idx<(not is_m_grouped_contiguous(kGemmType)), MN>(shape_m, BLOCK_M, m_block_idx);
DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
    auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
    auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;   // 第 threadIdx.x 个 n-atom
    using cute_tma_t = cute::conditional_t<kWithAccumulation,
        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;   // Batched 走 3D 变体
    cute_tma_t::copy(&tensor_map_cd, smem_ptr, n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
    cute::tma_store_arrive();   // cp.async.bulk.commit_group
}
__syncwarp();
```

- **D 在 SMEM 里按 n-atom 分块**：`smem_d` 布局是 `[BLOCK_N/TMA_D_BLOCK_N 个 n-atom][每 atom BLOCK_M 行 × TMA_D_BLOCK_N 列]`，第 `threadIdx.x` 个 atom 在 `smem_d + threadIdx.x * TMA_D_BLOCK_N * BLOCK_M`。bf16 时 `TMA_D_BLOCK_N = 64`，`BLOCK_N=128` 拆成 2 条 store（由 thread 0/1 发）；fp32 时 `TMA_D_BLOCK_N = BLOCK_N`，一条 store。
- **`m_idx` 的 `kWithGroupOffset = not is_m_grouped_contiguous(...)`**：m-grouped-contiguous 的 group 偏移已由 `make_tma_a_desc` 把 `m * num_groups` 拼进了 gmem 外维，故**不能**再加；masked/psum 变体需要加 `current_group_idx * shape_m`。
- **`kWithAccumulation`**：当调用方传了 `c`，TMA 从 `SM90_TMA_STORE_*` 换成 `SM90_TMA_REDUCE_ADD_*`（`cp.reduce.async.bulk.tensor.…add`），在**写回路径上做 GMEM 原子累加**，实现 `D += A@B` 而不需先读 C。Batched 时同理换成 3D 变体（多传 `scheduler.current_group_idx` 作 batch 坐标）。
- **`epilogue::transform`**：本 kernel 未显式调用 `apply_index_n`（n 索引直接用 `n_block_idx * BLOCK_N + in_block_n_offset`）。[epilogue/transform.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/epilogue/transform.cuh) 提供的 `EpilogueHeadSplits<kLeft,kMid,kRight>` 用于 attention 场景把 Q/K/V 三段拼接的 N 轴索引跳过中间 head 段（要求三段都能被 `STORE_BLOCK_N` 整除），BF16 GEMM 走恒等的 `EpilogueIdentity`。

### 11.6 dtype 转换

```cpp
// bf16：一次指令完成 2 个 fp32 → bf16 的舍入与拼接
__float22bfloat162_rn({shifted_accum[i*4+0], shifted_accum[i*4+1]})   // → nv_bfloat162（32-bit）
// fp32：直存
make_float2(shifted_accum[i*4+0], shifted_accum[i*4+1])               // → st.shared.v2.f32
```

`__float22bfloat162_rn` 与 SM100 的 `cast_into_bf16_and_pack`（[common/math.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/math.cuh) 第 72–76 行）是同一个东西——**一条指令完成两个 FP32 → BF16 的 round-to-nearest-even 舍入与拼接**。累加器永远是 FP32（`MMA_64xNx16_F32BF16BF16_SS` 的 F32 累加），BF16 只出现在输入与最终输出，不存在中间累加降精度。

---

## 12. 其他重要机制

### 12.1 PDL：把 prologue 藏进前驱 kernel 的尾巴

Programmatic Dependent Launch 允许后一个 kernel 在前一个 kernel **尚未执行完**时就被派发到 SM 上跑，只要它不碰前驱的输出。SM90 把 `cudaGridDependencySynchronize()`（PTX: `griddepcontrol.wait`）**刻意推迟**到 prologue 的最后（第 132 行）：

```
line  86  if (warp_idx == kNumMathThreads/32 && elect_one_sync()) prefetch_tma_descriptor(a/b/cd);  // tensormap 进 L2
line 113  if (warp_idx == kNumMathThreads/32+1 && elect_one_sync()) { …2×kNumStages 个 mbarrier init…; fence_barrier_init(); }
line 125  kNumTMAMulticast > 1 ? cluster_sync_with_relaxed_arrive() : __syncthreads();
line 132  cudaGridDependencySynchronize();     ◄── 真正的依赖点
line 136  …构造 scheduler…
line 148  TMA 分支：warpgroup_reg_dealloc<48>() → 第一条 TMA load
line 208  math 分支：warpgroup_reg_alloc<224/248>() → 第一条 wgmma
```

被提到依赖点之前的三件事（prefetch、mbarrier init、cluster sync）都**只读写本 kernel 自己的资源**（常量内存里的 descriptor、随 CTA 分配的私有 SMEM），与前驱的数据无因果关系；而第一条 `cp.async.bulk.tensor` 要读的 A/B 很可能是前驱 kernel 刚写出的，故留在依赖点之后。与 SM100 同构，只是 SM90 把**寄存器再配置（`setmaxnreg`）放在依赖点之后**（SM100 无此步）。

三个使用注意点与 SM100 完全一致（共用 `DeviceRuntime`）：① 默认关闭，需 `deep_gemm.set_pdl(True)`；② 不开 PDL 也正确（`cudaGridDependencySynchronize()` 立即返回）；③ 全仓库无 `cudaTriggerProgrammaticLaunchCompletion()`，只做「等待方」。

### 12.2 `cluster_sync_with_relaxed_arrive()` 与 `__syncthreads()` 的选择

```cpp
(kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();   // 第 125 行，仅 prologue
```

SM90 **只在 prologue 用一次** cluster sync（multicast 时），epilogue 无最终 cluster sync（§10.6）——对比 SM100 的三处。`cluster_sync_with_relaxed_arrive()`（[comm/barrier.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh) 第 14–19 行）= `cluster_arrive_relaxed()` + `cluster_wait()`，只做控制流汇合、不承诺内存序（比 `cute::cluster_sync` 快）。这里 relaxed 够用的原因：承担跨 CTA 可见性的是它上面第 121 行的 `fence_barrier_init()`（`fence.mbarrier_init.release.cluster`），cluster arrive 退化为纯汇合。单 CTA（`kNumTMAMulticast == 1`）时退化为 `__syncthreads()`。

### 12.3 NamedBarrier：epilogue 内的两级同步

epilogue 用 `cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0)`（第 295/361 行）而非 `__syncthreads()`：

```cpp
// 第一道（写完 smem_d 前）：把“上一块 TMA store 已读完 smem_d”从等待线程广播给全体 store 线程
if (threadIdx.x < BLOCK_N/TMA_D_BLOCK_N) tma_store_wait<0>();
NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
… STSM / st.shared 写 smem_d …
// 第二道（发 TMA store 前）：确保全体 store 线程的写入对 TMA 可见
cute::tma_store_fence();
NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
… TMA store …
```

`NamedBarrier`（`barrier.sync.aligned id, num_threads`，id = 0）**只绑 `kNumWGMMAStoreThreads` 个线程**（`BLOCK_M=128` 时 = 256，即两个 math warpgroup 全部）。不用 `__syncthreads()` 是因为后者会同步**全部**线程——包括正在异步跑自己那摊活的 TMA warpgroup，那会把 TMA 与 math 强行拉回同步，破坏流水。NamedBarrier 只在 math 线程内部建立两道屏障，TMA warpgroup 不受影响。

### 12.4 三层断言防御体系

与 SM100 共用 [common/exception.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/exception.cuh) 的三个宏：`DG_STATIC_ASSERT`（编译期 `static_assert`，零成本）、`DG_DEVICE_ASSERT`（`printf` + `trap`）、`DG_TRAP_ONLY_DEVICE_ASSERT`（只 `trap`）。

本 kernel 的关键 static 断言：

| 断言 | 行 | 检查内容 |
| --- | --- | --- |
| `BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M` | 62 | `BLOCK_M` 要么≥ 64 且整除 64，要么小于 64（§7.5） |
| `cd_dtype_t` 是 `float`/`bfloat16_t` | 65 | 输出 dtype 白名单 |
| `WGMMA_A_SIZE_PER_STAGE <= SMEM_A + SMEM_B*kNumStages` | 79 | WGMMA 越界读不冲出 A/B 区（§5.5，但用 fp8 size，偏松） |
| 三个 SMEM 区都是 1024 B 的倍数 | 95 | `__align__(1024)` + swizzle-128B 的前提 |
| `kNumTMAThreads >= 128` | 155 | TMA warpgroup 至少 4 warp（才能抽出第三个 warp 发 TMA） |
| `kNumTMAMulticast <= 2` | 164 | cluster 最多 2 CTA |
| `kGemmType` 与 `kMajorA` 组合合法 | 177 | m-grouped 必须 A K-major |
| `BLOCK_M >= 64 or kNumMathThreads == 128` | 228 | `BLOCK_M < 64` 只能单 math warpgroup |
| `BLOCK_M % WAVE_BLOCK_M == 0` | 224 | wave 划分整除 |
| `BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N/TMA_D_BLOCK_N <= 32` | 284 | TMA store 对齐且条数 ≤ 32 |
| `WGMMA::kNumAccum % 4 == 0` | 300 | STSM x2 向量化（bf16） |
| `kNumWGMMAStoreThreads >= BLOCK_N/TMA_D_BLOCK_N` | 365 | store 线程够发所有 TMA |

与 SM100 一样，**真正编出的 SM90 cubin 里 `DG_DEVICE_ASSERT` 一条都没有**（唯一一处在 `#else` 非-sm90 分支，第 388 行 `false and "This kernel only support sm_90a"`）。所有配置合法性都在 JIT 编译阶段被 `static_assert` 拦住；运行期检查只剩 `make_gmma_desc` 里 MN-major 分支的 `DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0)`（而 `mn_idx` 在编译期已知，实际不会触发）。

### 12.5 编译期常量折叠的收益链

JIT 全量模板特化触发的死代码消除链（以 `compiled_dims = "nk"` 为例）：

```
SHAPE_K != 0
  └─► shape_k 成为编译期常量（Normal 时 current_shape_k = shape_k）
        └─► num_total_k_blocks 可被常量传播（但外层 K 循环仍为运行期 for，不展开）
BLOCK_M / BLOCK_N / BLOCK_K 均为常量
  └─► 内层 local_idx（wave）、k（BLOCK_K/WGMMA::K）循环 `#pragma unroll` 完全展开
        └─► advance_gmma_desc_lo 的 offset/mn_idx/k_idx 全为立即数 → 折叠成一条 IADD3
        └─► WGMMA::wgmma → make_index_sequence<N/2> 展成 N/2 个寄存器操作数
  └─► epilogue 的 i 循环（kNumAccum/4）、swizzle 地址计算全展开
  └─► TMA_D_BLOCK_N、BLOCK_N/TMA_D_BLOCK_N 为常量 → store 线程数确定
```

被 `if constexpr` 彻底消除的分支维度：`kMajorA`（2）× `kMajorB`（2）× `kNumTMAMulticast`（2）× `kWithAccumulation`（2）× `kIsBatchedMM`（2）× `kGemmType`（7）× `cd_dtype_t`（2）× `kDoMergeStages`（2）。**一份 cubin 里只存在一条完全直线化的路径**。

与 SM100 的一个区别：SM90 **没有 tail-K 分支可消除**（§7.6 本就不生成），所以常量折叠链比 SM100 短一节；但描述符算术折叠、内层循环展开、STSM 展开的收益与 SM100 一致。同样地，`SHAPE_M` 默认不编译进来（推理场景 M 常变，避免 JIT 缓存爆炸），代价是 M 方向的 tail（`BLOCK_M < 64` 的无效行、`do_wgmma_store` 筛选）保留为运行期逻辑。

---

## 13. 不变式、限制与已知坑

### 13.1 必须成立的不变式

下表是「改动这个 kernel 或其 host 启发式时不能破坏」的硬约束。左列任一被打破，结果要么是编译失败，要么是静默的数值错误 / 死锁。与 SM100 相比，SM90 的不变式**少了 TMEM 那一整类**（`2×UMMA_N ≤ 512`、`32 ≤ kNumTmemCols ≤ 512`、base address 必须是第 0 列），**多了寄存器预算与描述符 16 B 整除**这两类——根源都是「累加器在寄存器」。

| 不变式 | 由谁保证 | 破坏后的表现 |
| --- | --- | --- |
| `BLOCK_K_ == 64` | host `block_k = 128 / element_size(BF16=2)` | 一个 swizzle atom 的 K 字节数必须 128 B；否则 `make_gmma_desc` 的 SBO/LBO 全错 |
| `BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M`（即 `%64==0` 或 `<64`） | host `block_m` 候选 `{16,32,64,128,256}` | `static_assert`（第 62 行） |
| `WGMMA::M == 64`、`WGMMA::K == 16` | `BF16MMASelector` / `BF16MMA` 硬编码 | wgmma 原子形状非法；`WAVE_BLOCK_M`、`kNumWGMMAStoreThreads` 推导失效 |
| 三个 SMEM 区各自 1024 B 对齐且为 1024 的倍数 | `SMEM_*_SIZE` 构造 + `constexpr_align` | `static_assert`（第 95 行）；swizzle-128B atom 跨边界 → 数据错乱 |
| `SMEM_A_SIZE_PER_STAGE % 16 == 0`（描述符 start_address 以 16 B 为单位） | 1024 对齐蕴含 | `a_desc_lo + stage_idx*(SMEM_A_SIZE_PER_STAGE/16)` 的整除截断出错（§9.3） |
| `WGMMA_A_SIZE_PER_STAGE <= SMEM_A + SMEM_B*kNumStages` | 第 79 行断言（**用 fp8 size，偏松 2×**） | WGMMA 按 M=64 读 A 越界冲出 A/B 区 → 非法 SMEM 访问（见 §13.3 第 1 条） |
| `kNumTMAMulticast ∈ {1,2}` | host cluster 过滤 + 第 164 行 | `static_assert`；>2 CTA 的 cluster multicast 不支持 |
| `kNumSMs % cluster_size == 0` | host 过滤（§2.2） | cluster 跨 grid 边界，launch 失败 |
| `ceil_div(n, block_n) % cluster_size == 0`（masked/psum 布局） | host 过滤 | cluster 内两 CTA 落到不同边界外，multicast 的 `arrive` 计数凑不齐 → 死锁 |
| 每个 barrier 的 `init` 计数与 arrive 次数严格相等（`full=1`，`empty=kNumTMAMulticast × kNumMathThreads/32`） | §10.3 的逐条推导 | 计数多 → 永久等待（死锁）；计数少 → parity 提前翻转（数据竞争） |
| `kNumTMAThreads >= 128` | host 恒 128 | `static_assert`（第 155 行）；抽不出第三个 warp 专发 TMA |
| 寄存器预算：`kNumMathThreads×kNumMathRegisters + 128×48 <= 65536` | `setmaxnreg` 取值 224/248（§3.2） | 超过 64K 寄存器/SM → `warpgroup_reg_alloc` 后 launch/执行失败 |
| m-grouped ⇒ A 必须 K-major | host `DG_HOST_ASSERT(major_a==K)` + 第 177 行 | `get_global_idx` 的 group 偏移公式失效 |
| k-grouped contiguous ⇒ A/B 都必须 MN-major | host `DG_HOST_ASSERT`（hpp 第 270 行） | K 方向的 group cumsum 索引算错 |
| `BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N/TMA_D_BLOCK_N <= 32` | 第 284 行 | TMA store 不对齐，或条数超过 32 个 store 线程所能发射的上限 |
| `kNumWGMMAStoreThreads >= BLOCK_N/TMA_D_BLOCK_N` | 第 365 行 | store 线程不够发所有 TMA 条 |
| `WGMMA::kNumAccum % 4 == 0`（bf16 路径） | 第 300 行 | STSM x2 向量化（一次 4 个累加器）失败 |
| `swap_ab == 0` | host `DG_HOST_ASSERT(layout.swap_ab==0)`（§2.3） | SM90 路径根本不支持 swap-AB |
| D 必须 N-major | `make_tma_cd_desc`（§2.6） | 输出布局错乱 |

### 13.2 功能与平台限制

- **仅 SM90 / `sm_90a`**。`#if __CUDA_ARCH__ >= 900` 之外只有 `DG_DEVICE_ASSERT(false and "This kernel only support sm_90a")`（第 386–389 行）。编译 flag 必须是 `--gpu-architecture=sm_90a`（`wgmma` 与 TMA multicast 都需要 `a` 后缀特性）。SM100 走同仓库另一套 kernel（`sm100_*`），流水组织完全不同。
- **输入 dtype 硬编码 BF16**。`smem_a`/`smem_b` 一律 cast 成 `cutlass::bfloat16_t*`，`BF16MMASelector<BLOCK_N>` 选出 `MMA_64xNx16_F32BF16BF16_SS`，累加器恒 FP32。输出 `cd_dtype_t` 只有 `float` 与 `bfloat16_t` 两种，没有 FP16。FP8/FP4 走别的 kernel。
- **不支持 swap-AB**。host 侧 `DG_HOST_ASSERT(layout.swap_ab == 0)` 直接拦死。SM100 支持 swap-AB（MoE 小 M 场景），SM90 没有这条路径——小 M 只能靠 `block_m ∈ {16,32}` + 承受 M=64 的算力浪费（§7.5）。
- **cluster 最多 2 CTA，且协作只有 TMA multicast**。SM90 的 cluster 协作是把一份 A 或 B **复制**给 pair 内两个 CTA（省 L2/GMEM 读带宽，**不省 SMEM**，两 CTA 各算各的块），没有 SM100 的 2-CTA UMMA（**切分**操作数，两 SM tensor core 合算一块，既省带宽又省 SMEM）。所以 SM90 开 cluster 不会增加 stage 数。
- **无 split-K**。K 方向由单个 CTA 串行跑完整个 `num_total_k_blocks`，K 很大而 M/N 很小的瘦长 shape 无法靠增加并行度填满 SM。`kWithAccumulation`（`cp.reduce.async.bulk.tensor…add`）提供「多次调用累加到同一块 D」的能力，算是 host 层面的手工 split-K，但调用之间没有 kernel 内同步。
- **persistent 调度是静态的**。`next_block_idx = (++current_iter) * kNumSMs + blockIdx.x`，纯算术映射，没有原子操作也没有工作窃取。好处是零同步开销、每个 CTA 独立推算自己的块序列；代价是块间代价不均时无法再平衡——`MGroupedMasked` 下各 group 的 `masked_m` 差异很大时，末 wave 长尾直接暴露在关键路径上。
- **compute 与 epilogue 串行、D 单缓冲、无重叠**（§9.6 / §11.4）。同一批 math 线程先跑完整个 k-loop（累加器占满寄存器），再做 epilogue；epilogue 期间 tensor core 完全空闲，D 靠 `tma_store_wait<0>()` 串行复用单缓冲。这是 SM90 相对 SM100（MMA warp / epilogue warp 分离 + TMEM 双缓冲重叠）最主要的结构性性能损失，**无法靠调参消除**。
- **无 tail-K 专用分支**（§7.6）。靠 TMA 的 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` 零填充处理 K 非整除 `BLOCK_K` 的尾巴：wgmma 内层恒发满 `BLOCK_K/WGMMA::K = 4` 条，被零填充的 K 部分算入 0，结果正确但末尾块有无效算力。好处是代码路径单一（少一整块 SM100 那样的 `issue_tail_k_block` 复杂度）。
- **无 Tensor Core 利用率控制**。SM100 有 `kTensorCoreUtilControl` 旋钮（给功耗受限集群做可复现对比），SM90 路径没有对应模板参数，也没有 `clock64()` 自旋降速逻辑。
- **`gridDim.x == num_sms`**，且 `deep_gemm.set_num_sms()` 同时影响 grid 大小与调度器的 `kNumSMs`；设成小于物理 SM 数会有 SM 闲置，两者必须一致否则块分配会漏。

### 13.3 已知坑与代码瑕疵

按「踩到的概率 × 排查成本」排序：

1. **`WGMMA_A_SIZE_PER_STAGE` 用 fp8 的 `sizeof` 而非 bf16**（第 78 行）。`WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3)`——`sizeof(__nv_fp8_e4m3) == 1`，但本 kernel 是 BF16（应 ×2）。这是从 FP8 kernel 移植遗留的瑕疵，导致第 79 行的越界防护**松了 2×**：真正要保证的是「WGMMA 按 `M=64` 读 A 时不冲出 A/B 区」，但断言只按一半大小检查。之所以不出事：`SMEM_A_SIZE_PER_STAGE` 本身已是 `BLOCK_M*BLOCK_K*2`，且 B ring 通常有极大裕量（`kNumStages × SMEM_B_SIZE_PER_STAGE`），WGMMA 越界读到的垃圾行在 epilogue 里根本不会被读（只读 `BLOCK_M` 行）。但若把 `BLOCK_M` 调到 `< 64` 同时 B ring 很窄，理论上可能踩到未分配 SMEM。这是本 kernel 最值得警惕的潜伏 bug，改 `BLOCK_M` 候选或 SMEM 布局时要手工复核真实的 2× 边界。

2. **compute 与 epilogue 串行、无重叠**（§9.6 / §11.4）。累加器在寄存器 ⇒ 一个输出块的 k-loop 与 epilogue 由同一批 math 线程串行完成，epilogue 期间 tensor core 空闲，D 单缓冲靠 `tma_store_wait<0>` 串行复用。看 nsys 时间线会看到「wgmma 段」与「STSM+TMA store 段」交替而非重叠，这不是调度没调好，是架构约束（对照 SM100 的 TMEM 双缓冲）。想量化 tensor core 利用率时要意识到这段空窗是固有的。

3. **`warpgroup_wait<0>()` 每个 k_block 都阻塞**（§9.5，「MMA 半同步」）。因为累加器在寄存器 + wgmma 从 SMEM 异步读，必须等这一 stage 的 MMA 全部完成，才能 `empty_barrier_arrive` 释放 SMEM stage 给 TMA 覆写。`kDoMergeStages`（§7.4，stage≥10 且 NT-Normal-单 math warpgroup 时把 `BLOCK_K` 放大 `kNumStagesPerMerge` 倍）正是为了摊薄这个 `wait<0>` 的频率——一次 wait 覆盖更多 K。若关掉 stage 合并，小 `BLOCK_K` 会让 `wait<0>` 成为流水的主要气泡。

4. **`BLOCK_M < 64` 时算力浪费 + store warp 空转**（§7.5 / §3.4）。`WGMMA::M` 恒为 64，即使 `BLOCK_M=16/32`，wgmma 仍按 M=64 发射，累加器 64 行里只有 `BLOCK_M` 行有用；且 `do_wgmma_store = BLOCK_M>=64 or warp_idx < kNumWGMMAStoreThreads/32` 只让部分 math warp 参与 store，其余空转。换来的是避免小 M 块的 TMA L2 OOB + 单一代码路径。profiler 里「wgmma 指令数 / 有效 FLOP」比值异常时先确认 `BLOCK_M`。

5. **multicast 省带宽不省 SMEM**（§7.2）。SM90 的 `load_block_m/n` **不除 cluster**（§2.3），每个 CTA 存整份操作数。`is_tma_multicast_valid` + `SM90_TMA_LOAD_MULTICAST` 让 rank0 的一条 load 把操作数复制进 pair 内两个 CTA 的 SMEM，省的是 L2/GMEM 读带宽。若误以为它像 SM100 2-CTA UMMA 那样也省 SMEM，会错误预估 stage 数。

6. **`is_peer_cta_alive` 为假时 `empty_barrier_arrive` 的回退**（§10.3，第 237 行）。multicast 下 `target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster()`——当 peer CTA 已退出（persistent 尾部块数不均），两个 lane 都把 arrive 投给**本 CTA** 以保持 `empty_barriers`（init 为 `kNumTMAMulticast × kNumMathThreads/32`）的计数平衡。误改这个三元表达式、或把 `lane_idx < kNumTMAMulticast` 的门槛动一动，都会在尾部死锁。

7. **`a_desc_lo` 单值算术推进依赖 16 B 整除**（§9.3，第 245 行）。`a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16)`——GmmaDescriptor 的 start_address 以 16 B 为单位，所以这里整除 16。SM90 只用**一个** `a_desc_lo`/`b_desc_lo` 标量按 stage 加常量步长推进（对比 SM100 用 32 个 lane 各存一个 stage 的描述符低位）。这依赖 `SMEM_A_SIZE_PER_STAGE % 16 == 0`（由 1024 对齐蕴含，§13.1）。

8. **epilogue STSM「only 16 lanes' addresses are used」**（第 337 行注释）。`SM90_U32x2_STSM_N` 的 `stmatrix.x2.m8n8.b16` 只用一个 warp 里 **16 个 lane** 的地址（另 16 个被忽略），swizzle 地址计算（`col ^= row % (kSwizzleDMode/16)`、`kHasShortcut` 分支）必须与之匹配。读这段地址算术时容易误以为 32 lane 都在写 SMEM。

9. **TMA warpgroup 里大量 warp 空转**（§3.1）。TMA warpgroup 128 线程 / 4 warp 里，稳态只有**第三个 warp的 1 个 elected lane** 发 TMA；w8 prefetch（一次性）、w9 init barrier（一次性）、w11 恒空转。但与 SM100 不同的是：math warpgroup 的 128/256 线程稳态**都**在算或搬，所以 SM90 的「线程填充率」仍是有效健康指标（§1.4 末行），不像 SM100 那样 `sm__warps_active` 会把人引向错误的「低效」结论。

10. **`__shfl_sync(0xffffffff, …)` 要求整 warp 收敛**（第 82 / 213 / 219 / 220 行）。`warp_idx`、`math_wg_idx`、`a_desc_lo`、`b_desc_lo` 都靠整 warp 的 `__shfl_sync` 取统一值（注释：*encourage NVCC to use unified registers*）。它们都位于 warp-uniform 的位置，合法；但若将来有人在 math warpgroup 里引入依赖 `lane_idx` 的分支，这几处会立刻变成未定义行为。

---

## 14. 小结

这份 kernel 的核心思想可以压缩成三句话：

1. **累加器住在寄存器，是塑造这个 kernel 一切形态的根本约束**。没有 TMEM ⇒ `wgmma.mma_async` 必须 128 线程协同发射（每线程「拿着」自己那份累加器）⇒ math warpgroup 要 224/248 的寄存器配额、逼出 `setmaxnreg` 再配置 ⇒ compute 与 epilogue 由同一批线程串行、D 只能单缓冲 ⇒ `BLOCK_M < 64` 也得按 M=64 发射。SM100 把这些约束统统甩给 Tensor Memory，于是能做到 MMA warp / epilogue warp 分离、TMEM 双缓冲重叠、单线程发射 UMMA。两代架构的分野，物理根源就在「累加器放哪」。
2. **把同步外化成两类 mbarrier 的 parity 相位**（`full` / `empty`）。A/B 的 GMEM→SMEM 全异步、可超前 MMA 多达 `kNumStages` 个 k_block；但 MMA→释放 SMEM 这一段因 `warpgroup_wait<0>()` 而**半同步**——这是累加器在寄存器 + wgmma 从 SMEM 异步读带来的必然妥协（§9.5），也是 stage 合并存在的理由。
3. **把能变成编译期常量的东西全部变成编译期常量**。JIT 全量特化 ⇒ 内层 wave/k 循环完全展开、描述符增量成即数、所有 `if constexpr` 分支塌缩成一条直线路径。SM90 比 SM100 少一节（没有 tail-K 分支可消除，因为它本就不生成），但收益同源——这才是 DeepGEMM 相对通用库的真正护城河。

理解「累加器在寄存器」这一条之后，其余所有细节——为什么要 128 线程发 wgmma、为什么要 `setmaxnreg`、为什么 compute/epilogue 串行、为什么 D 单缓冲、为什么 `BLOCK_M<64` 仍发 M=64、为什么 multicast 不省 SMEM、为什么 `WGMMA_A_SIZE_PER_STAGE` 的 fp8 瑕疵不出事——都是它的自然推论。把这份文档与 [SM100 姊妹篇](./deepgemm_sm100_bf16_gemm_design.md) 对照着读，两代 Hopper/Blackwell tensor core 编程模型的差异会一目了然：一个是把累加器塞进寄存器的 warpgroup MMA，一个是把累加器外置到 Tensor Memory 的单线程 UMMA。

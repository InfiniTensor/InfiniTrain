# DeepGEMM `sm100_bf16_gemm` Kernel Detailed Design

面向 Blackwell（SM100，B200/GB200）的 BF16 GEMM 内核详细设计。本文覆盖：warp specialization 与 pipeline 组织、多级 tiling、SMEM/TMEM 排布、Tensor Core（`tcgen05.mma`）调用方式、生产者-消费者 mbarrier 同步协议、TMA 指令的发射与完成语义，以及若干容易被忽略但对正确性/性能关键的设计点。

## 0. Code Index

| 层次 | 文件 | 职责 |
| --- | --- | --- |
| Device 主体 | [sm100_bf16_gemm.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh) | kernel 本体：SMEM/TMEM 布局、三个 warp 角色、流水推进 |
| UMMA 描述符 | [mma/sm100.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/mma/sm100.cuh) | `SmemDescriptor` 构造、SBO/LBO 推导、K 方向描述符推进 |
| tcgen05 PTX | [ptx/tcgen05.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh) | `tcgen05.mma.*` 内联汇编、`tcgen05.fence` |
| TMA load | [common/tma_copy.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/tma_copy.cuh) | swizzle-atom 循环、1SM/2SM/multicast 分支 |
| TMA PTX | [ptx/tma.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/tma.cuh) | `cp.async.bulk*`、`mbarrier.*`、tensormap 改写 |
| 调度器 | [scheduler/gemm.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh) | persistent 块分配、L2 swizzle、grouped/batched 索引 |
| Epilogue | [epilogue/sm100_store_cd.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/epilogue/sm100_store_cd.cuh)<br>[epilogue/sm100_store_cd_swap_ab.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh) | TMEM→RF→SMEM→GMEM，swizzle 计算与 STSM 转置 |
| 通用工具 | [common/utils.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/utils.cuh) | `PatternVisitor`、编译期循环展开、TMEM 列对齐 |
| Host JIT | [impls/sm100_bf16_gemm.hpp](../third_party/DeepGEMM/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp) | 模板实参拼装、TMA descriptor 构造、launch |
| Host 启发式 | [heuristics/sm100.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/sm100.hpp)<br>[heuristics/config.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/config.hpp)<br>[heuristics/utils.hpp](../third_party/DeepGEMM/csrc/jit_kernels/heuristics/utils.hpp) | BLOCK_M/N/K、cluster、swizzle、stage 数、线程数推导 |

> **注**：本仓库的 `third_party/DeepGEMM/third-party/cutlass` 子模块为空目录，CUTLASS 头文件未随仓库落地。文中涉及 `cute::UMMA::SmemDescriptor` 位域、`cutlass::arch::umma_arrive`、`cute::SM100_TMA_2SM_LOAD_2D` 等具体 PTX 文本时，均标注为「依据 CUTLASS 约定/由本仓库调用方式反推」，不做逐字断言。

---

## 1. 设计总览

### 1.1 一句话概括

一个 **persistent + warp-specialized + 全异步** 的三段流水线内核。数据通路为：

```
GMEM ──(cp.async.bulk.tensor / TMA)──► SMEM(A,B ring)
                                          │
                                          ▼
                            tcgen05.mma.cta_group::{1,2}.kind::f16
                                          │
                                          ▼
                                     TMEM(累加器双缓冲)
                                          │
                                          ▼  tcgen05.ld (32dp32b*)
                                         RF
                                          │
                                          ▼  st.shared.v4 / stmatrix.trans
                                    SMEM(C/D ring)
                                          │
                                          ▼  cp.async.bulk.tensor store / reduce.add
                                        GMEM(D)
```

从 GMEM 到 GMEM **没有任何一处让数据「停在通用寄存器里等一个 `__syncthreads()`」**：所有跨阶段依赖都由 `mbarrier` 的 parity 相位 + `tcgen05.commit` 表达，每个阶段都能独立超前推进。

### 1.2 框图（单个 CTA 内）

```
                     ┌──────────────────────────── 256 threads / 8 warps ────────────────────────────┐
                     │                                                                               │
   warp0 (1 lane)    │   warp1 (1 lane, leader CTA)          warp2                warp4..7 (128 thr) │
  ┌──────────────┐   │  ┌────────────────────────┐      ┌──────────────┐      ┌───────────────────┐   │
  │ TMA LOAD     │   │  │ MMA ISSUE              │      │ TMEM ALLOC   │      │ EPILOGUE          │   │
  │  persistent  │   │  │  persistent            │      │ (prologue    │      │  persistent       │   │
  │  k-loop      │   │  │  k-loop                │      │  only)       │      │  store-loop       │   │
  └──────┬───────┘   │  └───┬──────────────▲─────┘      └──────────────┘      └────▲──────┬───────┘   │
         │           │      │              │                                       │      │           │
    wait │ empty[s]  │ wait │ full[s]      │ tcgen05.commit                    wait │      │ arrive    │
         ▼           │      ▼              │  → empty[s]                       tmem_full  ▼ tmem_empty│
   ┌───────────┐     │  ┌───────────┐      │  → tmem_full[a] (last k)         [a]  │  ┌───────────┐   │
   │ SMEM ring │─────┼─►│ tcgen05   │──────┼───────────────────────────────────────┼─►│ TMEM ring │   │
   │ A[s],B[s] │     │  │   .mma    │      │                                       │  │  acc[a]   │   │
   │ kNumStages│     │  └───────────┘      │                                       │  │  2 stage  │   │
   └───────────┘     │                     │                                       │  └───────────┘   │
                     └─────────────────────┴───────────────────────────────────────┴──────────────────┘
```

warp3 以及 warp1 在 peer CTA 上的副本是**空转**的（见 §3.1）。注意这里的「空转」仅指 SIMT 线程层面：**grid 是 `kNumSMs` 个 persistent CTA，每个 SM 上都有一个自己的 lane 在发 UMMA，没有任何一个 SM 的 tensor core 被浪费**——完整推导见 §3.4。

### 1.3 关键设计选择

| 设计点 | 取值 | 理由 |
| --- | --- | --- |
| 编译方式 | 每个 (shape, config) 组合 JIT 生成一份全量模板特化的 `.cu` 编到 cubin | BLOCK_*、swizzle、stage 数、甚至 N/K 本身都成为编译期常量 → 内层 K 循环完全展开、描述符偏移常量折叠、tail-K 分支整段消除 |
| Grid | `gridDim.x = num_sms`，`__launch_bounds__(256, 1)` | persistent kernel，1 CTA/SM；配合近满额 SMEM 与独占 TMEM，物理上排除 2 CTA 共 SM |
| Cluster | 1 或 2（`cluster_m×cluster_n ≤ 2`） | 用 `cta_group::2` UMMA 把 M 做到 256，B 沿 N 对半切 → B 的 SMEM 占用与 L2 流量双双减半 |
| MMA 指令 | `tcgen05.mma.cta_group::{1,2}.kind::f16`，`UMMA_K = 16` | BF16 的 UMMA K 原子固定 16；A/B 均来自 SMEM（`_SS` 变体），累加器在 TMEM |
| `UMMA_M` | 恒为 `128 × kNumMulticast` | TMEM 的 datapath（行）固定 128，A/D 布局原子就是 128 行；`BLOCK_M ∈ {32,64}` 时也照发 M=128 的指令（见 §7.6） |
| 累加器缓冲 | TMEM 双缓冲（`kNumEpilogueStages = 2`） | 让第 i+1 块的 MMA 与第 i 块的 epilogue 重叠 |
| C/D 缓冲 | SMEM 双缓冲（`kNumTMAStoreStages = 2`） | 让 STSM 写与 TMA store 读重叠 |
| A/B 环 | 尽可能多的 stage（host 端按 SMEM 预算反解，上限 32） | 掩盖 HBM 延迟；stage ≥ 8 且 NT-Normal 时再做「stage 合并」把 `BLOCK_K` 放大（见 §7.5） |
| PDL | `cudaGridDependencySynchronize()` 放在 prologue **之后** | barrier 初始化、TMEM 分配、TMA descriptor prefetch 与前驱 kernel 的尾巴重叠 |

---

## 2. Host 侧：JIT 特化与配置推导

Kernel 的全部行为由 26 个模板实参决定，它们在 [sm100_bf16_gemm.hpp](../third_party/DeepGEMM/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp) 的 `generate_impl()` 里被格式化成一段只包含 `__instantiate_kernel()` 的 `.cu` 源码，再交给 nvcc 编成 cubin。

### 2.1 模板实参来源

```
kMajorA / kMajorB        ← 由 a/b 的 stride 推断（get_major_type_ab）
SHAPE_M / SHAPE_N / SHAPE_K ← get_compiled_dim(dim, 'm'/'n'/'k', compiled_dims)：
                            在 compiled_dims 里 → 填真实值；否则填 0（= 运行期参数）
BLOCK_M / BLOCK_N / BLOCK_K_ ← Layout
kNumGroups               ← GemmDesc
kSwizzle{A,B,CD}Mode     ← StorageConfig
kNumStages_              ← PipelineConfig.num_stages
kNumNonEpilogueThreads / kNumEpilogueThreads ← LaunchConfig（固定 128 / 128）
kNumMulticast            ← Layout.get_cluster_size()
kIsMulticastOnA          ← (Layout.cluster_n > 1)
kNumSMs                  ← LaunchConfig.num_sms（= gridDim.x）
kKAlignment              ← heuristics_runtime->get_mk_alignment_for_contiguous_layout()
kSwapAB / kEnsureZeroPadding / kGemmType / kWithAccumulation / cd_dtype_t ← GemmDesc
kTensorCoreUtilControl   ← device_runtime->get_tc_util()（默认 100）
```

Python 侧 `bf16_gemm_nt` 等 API 的 `compiled_dims` 默认是 `"nk"`（grouped contiguous/masked 也是 `"nk"`，两个 batched einsum 变体是 `"mn"`）。因此**典型场景下 `SHAPE_M == 0`、`SHAPE_N`/`SHAPE_K` 为编译期常量**，kernel 内第 117–119 行的覆写：

```cpp
shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
```

会把 N/K 变成常量，直接影响第 313 行：

```cpp
constexpr bool kMayHaveTailKBlock = is_k_grouped_contiguous(kGemmType)
        ? (kKAlignment % BLOCK_K != 0)
        : (SHAPE_K == 0 or SHAPE_K % BLOCK_K != 0);
```

当 K 被编译进来且能被 `BLOCK_K` 整除时，`kMayHaveTailKBlock == false`，整个 tail-K 处理分支（`for_each_static_prefix` + `switch`）**不会生成任何 SASS**。这是 JIT 相比通用库最大的收益之一。

### 2.2 Layout 候选枚举

`SM100ArchSpec::get_layout_candidates()`：

- **`block_k` 恒定**：`128 / element_size(BF16=2) = 64`。即 `BLOCK_K_ == 64`，对应 kernel 里的 `DG_STATIC_ASSERT(BLOCK_K_ == 64)`。物理含义：一个 swizzle atom 的 K 方向字节数固定为 128 B。
- **m-grouped 三种类型强制走 swap-AB**：`block_n = 128`，`block_m = get_mk_alignment_for_contiguous_layout()`（SM100 上按 `expected_m` 在 `[32, 224]` 内以 32 为步长收缩），`cluster_m = 1`，`cluster_n = 2`（当 `ceil(n/128)` 与 `num_sms` 均为偶数）。
- **其余类型全枚举** `swap_ab × block_m × block_n × cluster_m × cluster_n`，逐条过滤：
  - `swap_ab == 1 && cluster_m > 1` → 跳过；`swap_ab == 0 && cluster_n > 1` → 跳过（只支持 layout A/D 方向的 cluster）。
  - `cluster_m * cluster_n > 2`、`num_sms % cluster_size != 0` → 跳过。
  - 非 swap 时 `block_m ∈ {32, 64, 128}`，按 `desc.m ≤ 32 / ≤ 64 / else` 三选一（注释：*smaller block M can avoid TMA L2 OOB bound*）。
  - 非 swap 时 `block_n` 上界：`desc.k <= 256 ? 128 : 256`（注释：*For small K, fewer store blocks improve store/compute overlap*），步长 `lcm(32, block_n_multiple_of)`，另加一个 16 的候选。
  - MN-major 时要求 `(block_m/cluster_n) % 64 == 0`（swizzle 对齐）；K-major 时只要求 `% 8 == 0`。
  - `ceil_div(desc.m, block_m) % cluster_m != 0` 或 `ceil_div(desc.n, block_n) % cluster_n != 0` → 跳过（保证 cluster 不跨边界）。
  - `swap_ab && block_n != 128` → 跳过（`LAYOUT_AD_M` 必须是 128）。
  - **TMEM 容量**：`2 * umma_n + tmem_sf_cols > 512` → 跳过（BF16 无 SF，`tmem_sf_cols = 0`，故 `umma_n ≤ 256`）。
  - 当 A 或 B 至少有一个是 K-major 时，要求算出的 `swizzle_a_mode == swizzle_b_mode == 128`（注释：*32B swizzle yields poor performance*）。

打分 `compare()` 的优先级序列：**单 wave 最优 → cluster 大者优 → wave 数少者优 → 末 wave 利用率高者优 → `block_m + block_n` 小者优（= stage 更多） → `block_m × block_n` 小者优**。

### 2.3 StorageConfig

```cpp
load_block_m  = block_m / cluster_n;
load_block_n  = block_n / cluster_m;
store_block_m = swap_ab ? 16 /* umma_step_n */ : min(128 /* layout_ad_m */, block_m);
store_block_n = block_n;

swizzle_mode_a  = get_swizzle_mode(major_a == K ? block_k : load_block_m, sizeof(a_dtype));
swizzle_mode_b  = get_swizzle_mode(major_b == K ? block_k : load_block_n, sizeof(b_dtype));
swizzle_mode_cd = get_swizzle_mode(store_block_n, sizeof(cd_dtype));
```

`get_swizzle_mode()` 从 `{128, 64, 32, 16}` 里挑第一个能整除 `block_size * elem_size` 的值。BF16 + `block_k = 64` → 128 B，恒为 `swizzle = 128`。

注意 host 的 `store_block_n = block_n` 与 kernel 的 `STORE_BLOCK_N` 并不相等：kernel 里非 swap 分支是 `kSwizzleCDMode / sizeof(cd_dtype_t)`（即 TMA box 的内维被压到一个 swizzle atom 宽）。host 的 `store_block_n` 只用于算 swizzle mode，`make_tma_2d_desc()` 里又会被 `smem_inner_dim = swizzle_mode / elem_size` 覆盖掉。两者最终一致，但读代码时容易误判。

### 2.4 PipelineConfig（SMEM 预算 → stage 数）

```cpp
constexpr int smem_capacity = 232448;                       // 227 KB
int smem_cd       = swap_ab ? store_block_m * store_block_n * elemsize * 2
                            : store_block_m * swizzle_cd_mode * 2;   // × 2 = 双缓冲
int smem_barriers = 32 * 8 * 3 + 2 * 8 * 2 + 8;             // = 808 B，按 kNumMaxStages=32 预留
int smem_tmem_ptr = 4;
int smem_a_per_stage = load_block_m * block_k * elemsize_a;
int smem_b_per_stage = load_block_n * block_k * elemsize_b;

num_stages = min((smem_capacity - (smem_cd + smem_barriers + smem_tmem_ptr))
                 / (smem_a_per_stage + smem_b_per_stage), 32);
smem_size  = smem_extra + num_stages * smem_per_stage;
```

`smem_barriers` 的三项与 device 侧布局精确对应（见 §5.2）：`32*8*3` 是每 stage 三组 barrier（full / empty / **with-SF full**），`2*8*2` 是 `tmem_full[2] + tmem_empty[2]`，`+8` 是 `tensor_core_full_barrier`。BF16 没有 scale factor，第三组永不使用，但仍按 1D1D（FP8/FP4）kernel 的约定占位——所以 host 的预留量对 `kNumStages ≤ 32` 恰好是**紧上界**。

### 2.5 LaunchConfig

```cpp
return { desc.num_sms, layout.get_cluster_size(), 256, 32, 128, 128, 128 };
//        num_sms      num_sms_per_cluster  num_threads tma math non_epi epi
```

只有 `num_threads = 256`、`num_non_epilogue_threads = 128`、`num_epilogue_threads = 128` 会进模板实参；`num_tma_threads = 32` / `num_math_threads = 128` 是给 SM90 kernel 用的字段，SM100 路径忽略。

### 2.6 TMA descriptor

三个 descriptor 都以 `__grid_constant__ cute::TmaDescriptor` 按值传参（128 B 常量内存，避免走 GMEM）。构造见 [runtime_utils.hpp](../third_party/DeepGEMM/csrc/jit_kernels/impls/runtime_utils.hpp)：

| descriptor | gmem (inner, outer) | smem box (inner, outer) | 备注 |
| --- | --- | --- | --- |
| A | K-major: `(k, m*G)`；MN-major: `(m*G, k)` | K-major: `(block_k→64, block_m)`；MN-major: `(block_m→64, block_k)` | `num_groups > 1` 时强制 K-major；box 内维被 swizzle 覆写为 `swizzle/elem = 64` |
| B | K-major: `(k, n)`；MN-major: `(n, k)` | 同上，`block_n` 换 `load_block_n` | `num_groups` 只作用在外维：`gmem_outer_dim * num_groups` |
| C/D | `(n, m*G)` | `(store_block_n→swizzle/elem, store_block_m)` | D 必须 N-major |

公共属性：`CU_TENSOR_MAP_INTERLEAVE_NONE`、`CU_TENSOR_MAP_L2_PROMOTION_L2_256B`、`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`（越界元素零填充——这是 tail-K 与 M/N 非整除时结果仍正确的硬件保证）、swizzle 由 `mode_into_tensor_map_swizzle()` 映射到 `CU_TENSOR_MAP_SWIZZLE_{NONE,32B,64B,128B}`。

Batched（`sm100_bf16_bhr_hdr_bhd` / `bhd_hdr_bhr`）走 `make_tma_3d_desc()`，第三维是 head，device 侧 `kIsBatchedMM` 打开 `SM90_TMA_LOAD_3D` / `SM100_TMA_2SM_LOAD_3D` 分支。

### 2.7 Launch 属性

`construct_launch_config()`（[handle.hpp](../third_party/DeepGEMM/csrc/jit/handle.hpp)）：

1. `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size)` —— 动态 SMEM 远超 48 KB 静态上限，必须显式抬。
2. `cluster_dim > 1` → `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = {cluster_dim, 1, 1}`。
3. `enable_pdl` → `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 1`。默认 `DeviceRuntime::enable_pdl = false`，需 `deep_gemm.set_pdl(True)` 打开。
4. `cuLaunchKernelEx` 发射。

编译 flag：`--gpu-architecture=sm_100f`（nvcc ≥ 12.9 用 arch-family 后缀 `f`，否则 `100a`）、`-O3 --expt-relaxed-constexpr --expt-extended-lambda`，产物直接是 `-cubin`。kernel 内 `#if __CUDA_ARCH__ >= 1000` 之外的编译分支只留一个 `DG_DEVICE_ASSERT(false and "This kernel only support sm_100f")`。

---

## 3. 线程组织与 warp 角色

### 3.1 角色表（256 threads = 8 warps）

| warp | 条件 | 角色 | 实际活跃线程 |
| --- | --- | --- | --- |
| 0 | `warp_idx == 0 && elect_one_sync()` | **TMA load**：persistent 遍历所有块，每块跑完整 K 循环，发 A/B 的 TMA | **1 lane**（TMA 是单线程指令） |
| 1 | `warp_idx == 1 && elect_one_sync()` | barrier 初始化（prologue） | 1 lane |
| 1 | `warp_idx == 1 && is_leader_cta` | **MMA issue**：persistent 遍历，每块跑完整 K 循环，发 `tcgen05.mma` 与 `tcgen05.commit` | 发射 UMMA 时 1 lane，`commit` 时整 warp |
| 2 | `warp_idx == 2` | **TMEM 分配**（`tcgen05.alloc`，prologue 一次性） | 整 warp（`.sync.aligned` 要求） |
| 3 | — | **空转** | 0 |
| 4 … 4+`kNumUMMAStoreThreads/32`-1 | `warp_idx >= kNumNonEpilogueThreads/32 && < (kNumNonEpilogueThreads+kNumUMMAStoreThreads)/32` | **Epilogue**：TMEM→RF→SMEM→TMA store | `kNumUMMAStoreThreads` 个线程 |
| 其余 | — | 空转 | 0 |

`kNumUMMAStoreThreads` 的取值决定了 epilogue 用几个 warp：

| 场景 | `STORE_BLOCK_M` | `kNumUMMAStoreThreads` | 参与 warp |
| --- | --- | --- | --- |
| 非 swap，`BLOCK_M = 128` | 128 | 128 | w4–w7（满 warpgroup） |
| 非 swap，`BLOCK_M = 64` | 64 | 64 | w4–w5 |
| 非 swap，`BLOCK_M = 32` | 32 | 32 | w4 |
| swap-AB | 16 | `kNumEpilogueThreads = 128` | w4–w7 |

非 swap 时 `kNumUMMAStoreThreads = STORE_BLOCK_M` 的原因：**TMEM 的一个 datapath（行）由一个线程负责**，读 `STORE_BLOCK_M` 行就需要 `STORE_BLOCK_M` 个线程。swap-AB 时 `STORE_BLOCK_N == 128`（静态断言强制），要覆盖全部 128 个 TMEM 行，因此必须是完整 warpgroup。

Peer CTA（`block_rank_in_cluster() != 0`）上的 warp 1 什么都不做——`tcgen05.mma.cta_group::2` 只由 CTA-pair 的 leader 发射，硬件自己从两个 CTA 的 SMEM 取操作数、往两个 CTA 的 TMEM 写结果。但这**不**意味着 peer CTA 的 tensor core 闲置，见 §3.4.4。

### 3.2 为什么必须 1 CTA/SM

三条互相加强的约束：

1. `__launch_bounds__(256, 1)` 显式声明每 SM 最多 1 个 block。
2. SMEM 占用接近 227 KB 上限（§2.4 就是按「装满」反解 stage 数的）。
3. Epilogue 开头的 `DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0)`（第 405 行）——断言 `tcgen05.alloc` 返回的基址列号为 0，即本 CTA 拿到了 TMEM 的第 0 列起的全部 `kNumTmemCols`。源码注释写明：*we also forbid two CTAs to share the same SM and its tensor memory*。

第 3 条同时是一个**功能简化**：因为基址恒为 0，代码里所有 TMEM 地址都可以直接写成 `accum_stage_idx * UMMA_N + 列内偏移`，不需要把分配返回的 base 加进去；行号（datapath）也只由 warp/lane 隐式决定，注释指出*hardware will ignore the warp index bits, i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`*。

### 3.3 `elect_one_sync()` 的使用纪律

`cute::elect_one_sync()` 底层是 `elect.sync`，**要求整 warp 收敛执行**。本 kernel 所有调用都写成 `if (warp_idx == K and cute::elect_one_sync())` 或在已经 warp-uniform 的分支内部，`warp_idx` 对同一 warp 的 32 lane 恒等，因此短路求值不会造成 `elect.sync` 的部分参与。同理，第 324 行发射 UMMA 的 `if (cute::elect_one_sync())` 位于 `full_barriers[...]->wait()` 与 `__shfl_sync(0xffffffff, ...)` 之后——`__shfl_sync` 用满 mask，也隐含要求收敛。

### 3.4 为什么「1 个 lane 发射 UMMA」不等于算力浪费

§3.1 的角色表里，MMA issue 一栏写的是「发射 UMMA 时 1 lane」。这个描述的**作用域是单个 CTA 内部**，很容易被误读成「整个 GPU 只有一个线程在驱动 tensor core，其他 SM 的算力全浪费了」。本节把这件事彻底说清。

#### 3.4.1 层次：grid 上是 `kNumSMs` 个发射者，不是 1 个

```
gridDim.x = kNumSMs（B200 = 148）     ← LaunchConfig 第一项，persistent kernel
__launch_bounds__(256, 1)              ← 每 SM 恰好 1 个 CTA（§3.2）
    └─► 每个 CTA 都有自己的 warp 1
          └─► 每个 warp 1 都有 1 个 elected lane 在发 tcgen05.mma
                ⇒ 整个 GPU 同时有 148 个 lane 在并行发射 MMA
```

调度器的 `next_block_idx = (++current_iter) * kNumSMs + blockIdx.x` 保证这 148 个 CTA 各自负责互不重叠的块序列（§4.1），每个 SM 的 tensor core 都有活干。**不存在「其他 SM 被浪费」的情形**。

#### 3.4.2 单 CTA 内为什么 1 个 lane 就够：SM90 → SM100 的架构跃迁

这不是 DeepGEMM 的取巧，而是 Blackwell tensor core 编程模型的根本变化：

| | SM90 `wgmma.mma_async` | SM100 `tcgen05.mma` |
| --- | --- | --- |
| 发射宽度 | 整个 warpgroup（128 线程）必须协同 | **单线程** |
| A 操作数 | 可来自寄存器（每线程持有分片） | 只来自 SMEM descriptor |
| B 操作数 | SMEM descriptor | SMEM descriptor |
| D 累加器 | **分散在 128 个线程的寄存器里** | **TMEM**（独立于寄存器文件的 256 KB 存储） |
| 为何需要那么多线程 | 累加器每个元素都得有个线程「拿着」 | 没人需要「拿着」任何东西 |

SM90 的 128 线程是被**累加器的存储位置**逼出来的，与算力无关。SM100 把累加器搬进 TMEM 之后这个约束消失，UMMA 发射退化成「一个线程往硬件异步队列里投递一条描述符」——它是一条**控制指令**，不是一条需要 32/128 个线程各自贡献数据分片的 SIMT 指令。

本 kernel 用的 `_SS` 后缀（`SM100_MMA_F16BF16_SS`）正是这个意思：A 和 B **都**来自 SMEM。三个操作数（A、B、D）没有一个住在寄存器里，自然没有任何理由让多个线程参与发射。

#### 3.4.3 量化：发射端有 ≥ 40 倍余裕

kernel 自己在 tensor core 利用率控制里给出了执行周期公式（第 382 行）：

```cpp
constexpr static uint64_t kNumUMMACycles = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
```

取典型配置 `UMMA_M = 128`、`UMMA_N = 256`、`BLOCK_K = 64`：

| 量 | 值 | 来源 |
| --- | --- | --- |
| 单条 UMMA 覆盖的 FLOP | `2 × 128 × 256 × 16` = **1,048,576** | `UMMA_K = 16`（BF16 固定） |
| 一个 stage 发射的 UMMA 条数 | `BLOCK_K / UMMA_K` = **4** | `issue_full_k_block`（第 341 行） |
| 一个 stage 的总 FLOP | `2 × 128 × 256 × 64` = **4,194,304** | — |
| 执行所需 tensor-core cycle | 4,194,304 / 8192 = **512** | 上式 |
| 发射所需 lane-cycle | 4 条 UMMA + 8 条描述符推进 IADD ≈ **12** | `issue_umma` 内两次 `advance_umma_desc_lo` |
| **发射 : 执行** | ≈ **1 : 43**（纯 UMMA 算则 1 : 128） | — |

`8192` 不是随意取的魔数，它就是 **SM100 每 SM 每 cycle 的 BF16 FLOP 峰值**：B200 约 2.25 PFLOPS ÷ 148 SM ÷ ~1.83 GHz ≈ 8310，取整到 8192。因此 `kNumUMMACycles` 的物理含义是「这个 stage 的 UMMA 在满速 tensor core 上要跑多少 cycle」。

**40 倍以上的比值意味着发射端从来不是瓶颈。** 一个 elected lane 每 512 个 cycle 只需忙约 12 个 cycle，其余时间都卡在 `full_barriers[...]->wait()` 上睡觉。

最有力的反证是 `kTensorCoreUtilControl` 这个旋钮的存在（§9.7）：作者需要**主动插入 `clock64()` 自旋**才能把 tensor core 拖慢、降低功耗掉频的可能性（第 372–387 行）。如果发射能力不足，这个功能没有任何意义。

#### 3.4.4 2-CTA 模式：控制流集中，执行分布

`cta_group::2` 时，peer CTA 的 warp 1 完全不执行 MMA 分支，但**它的 tensor core 一点没闲着**：leader CTA 的 1 个 lane 发出的那一条指令，会同时驱动 CTA-pair 两个 SM 的 tensor core——从两个 CTA 的 SMEM 各取一半操作数，往两个 CTA 的 TMEM 各写一半结果，把 `UMMA_M` 拼到 256（§7.2）。

这是 SM100 UMMA 最反直觉的地方：**指令流宽度与算力宽度彻底解耦**。1 个 lane 的指令流对应 2 个 SM 的全部 tensor core。

#### 3.4.5 真正会让 tensor core 空闲的因素

既然发射不是瓶颈，tensor core 停转只可能来自三条依赖边。kernel 的全部复杂度都花在让它们永不成为关键路径上：

| 等待点 | 物理含义 | 掩盖手段 |
| --- | --- | --- |
| `full_barriers[stage_idx]->wait(phase)` | SMEM 里还没有 A/B（TMA 未返回） | `kNumStages` 深的 A/B 环（最多 32，§2.4） |
| `tmem_empty_barriers[a]->wait(...)` | TMEM 双缓冲都被占（epilogue 未搬完） | `kNumEpilogueStages = 2` + TMEM 早释放（§11.4） |
| 输出侧 TMA store 带宽 | D 写不回 GMEM | `kNumTMAStoreStages = 2` + `wait_group.read`（§11.5） |

换言之：优化方向是「加深环、提前释放、提高 L2 命中」，而不是「多找几个线程来发射 MMA」。后者在 SM100 上已经没有任何收益空间。

#### 3.4.6 观测陷阱：不要用 warp 活跃度评估本 kernel

256 个线程里稳态活跃的只有约 34 个（TMA 1 lane + MMA 1 lane + epilogue 32~128），warp 3 恒空转（详见 §13.3 第 8 条）。这说的是 **SIMT 通路的活跃度低**，不是 tensor core 闲置。用 nsys / ncu 观察时：

- ❌ `sm__warps_active`、issue-slot 利用率、`smsp__inst_executed`——会得出「这个 kernel 效率极低」的**完全错误**结论。
- ✅ `sm__pipe_tensor_cycles_active`（tensor core 管线活跃 cycle）、TMA 的 L2 吞吐、`sm__mio_inst_issued`（TMEM 读写）——才是本 kernel 的真实健康指标。

Warp specialization 的设计目标本来就是把通用 SIMT 通路腾空，让数据搬运全部交给 TMA / tensor core / TMEM 这些专用引擎。用 SIMT 时代的指标去衡量它，等于用「多少工人在挥铲子」去评价一台挖掘机。

---

## 4. Persistent 调度器

### 4.1 复制式状态机，不是共享工作队列

第 180 行在**角色分派之前**构造 `scheduler`：

```cpp
auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups,
                                  kNumMulticast, kIsMulticastOnA, kNumSMs,
                                  kEnsureZeroPadding, kKAlignment, kKAlignment>(
    shape_m, shape_n, shape_k, grouped_layout);
```

它是个**寄存器里的值对象**，每个线程各持一份私有副本。三个角色 warp 各自独立调用 `get_next_block()`，靠 `next_block_idx = (++current_iter) * kNumSMs + blockIdx.x` 这一条纯算术式子得到**完全相同**的块序列：

```
iter 0 → blockIdx.x
iter 1 → blockIdx.x + kNumSMs
iter 2 → blockIdx.x + 2*kNumSMs
...
```

因此：

- **零原子操作、零全局 ticket 计数器**，调度开销是几条整数指令；
- 三个角色天然锁步，无需为「现在在处理哪一块」建立任何额外通信；
- `scheduler.current_iter` 直接充当**全局逻辑时钟**，MMA warp 与 epilogue warp 各自用它算出累加器缓冲的 stage/phase（§6.3），这就是两条流水之间唯一的「隐式」耦合。

代价是负载不能动态窃取：尾波（tail wave）的空闲 CTA 只能干等。启发式打分里的 `last_wave_util` 就是在 host 侧提前把这件事量化。

### 4.2 L2 swizzle 分组

`get_swizzled_block_idx()` 把线性的 `block_idx` 重映射成 (m, n)，目的是让**同时在飞的 kNumSMs 个 CTA 尽量共享 L2 里的 A/B**。

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

组大小的选择是**最小化 L2 工作集**：

```cpp
usage = kIsMulticastOnA ? candidate * BLOCK_N + ceil_div(kNumSMs, candidate) * BLOCK_M   // 在 N 上分组
                        : candidate * BLOCK_M + ceil_div(kNumSMs, candidate) * BLOCK_N;  // 在 M 上分组
// candidate ∈ {8, 16}，取 usage 最小者
```

`DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0)` 保证一个 cluster 的两个 CTA 不会跨组边界——否则它们拿到的 B tile 就不同了，2-CTA UMMA 会算错。

`#if __CUDA_ARCH__ < 1000` 分支里那段「修正不对齐的 TMA multicast」只对 SM90 生效，注释说明 SM100 的 2-CTA 模式**不能动态关闭**，因此 SM100 靠 host 侧的整除性过滤（§2.2）来保证。

### 4.3 GemmType 变体

| GemmType | `num_blocks` | 额外状态 | 说明 |
| --- | --- | --- | --- |
| `Normal` | `num_m_blocks * num_n_blocks` | — | `get_global_idx` 退化为 `block_idx * block_size` |
| `Batched` | 同上，`× kNumGroups` | `current_group_idx` 作为 batch_idx | 不走 swizzle，按 `kIsMulticastOnA` 决定 m/n 谁变化快；TMA 走 3D |
| `MGroupedContiguous` | 同上 | `grouped_layout[m]` = 每行所属 group | B 的外维加 `group * shape_dim` 偏移 |
| `MGroupedMasked` | 逐 group 累加 | `current_m_cumsum` | 边扫边把 `next_block_idx` 落到对应 group，`num_m_blocks` 每 group 重算 |
| `MGroupedContiguousWithPsumLayout` | 逐 group 累加 | `last_psum_m` / `current_psum_m` / `current_m_block_cumsum` | group 边界按 psum 偏移切分，`m_block_idx += last_psum_m / BLOCK_M` |
| `KGroupedContiguous{,WithPsumLayout}` | 同上 | `current_shape_k` / `current_k_cumsum` / `current_k_start,end` | 每个 group 的 K 长度不同 → `num_total_k_blocks` 逐块变化；要求 A/B 都是 MN-major |

对本 kernel 最关键的一条约束在第 216 行：

```cpp
DG_STATIC_ASSERT(kGemmType == Normal or is_k_grouped_contiguous(kGemmType) or kGemmType == Batched or
                 kMajorA == cute::UMMA::Major::K, "Invalid major");
```

即所有 m-grouped 变体的 A 必须 K-major（注释：*for all m-grouped GEMMs, A must be K-majored*），因为 group 偏移是加在外维上的。

### 4.4 跨块连续的流水线状态

第 184–191 行：

```cpp
uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++ k_block_idx;
    stage_idx = (stage_idx + 1) % kNumStages;
    phase ^= stage_idx == 0;      // 只在回绕到 stage 0 时翻转相位
};
```

`stage_idx`/`phase` 声明在**块循环之外**，`tma_stage_idx` 也在 epilogue 里注释为 *Share store pipeline between blocks*。这意味着 A/B 环与 C/D 环**跨输出块连续运转**：TMA warp 可以在 MMA warp 还在算第 i 块最后一个 k_block 时，就开始往刚被释放的 stage 里灌第 i+1 块的 k=0 数据。块边界上没有任何「排空-重启」的开销，这是 persistent kernel 相较 grid-per-tile 实现的核心优势。

`advance_pipeline` 同时被 TMA warp（第 203 行）和 MMA warp（第 314 行）作为 `for` 的递增表达式使用，两边独立维护但推进规则一致，因此 `stage_idx`/`phase` 序列天然对齐。

---

## 5. 共享内存布局

### 5.1 线性布局

```cpp
extern __shared__ __align__(1024) uint8_t smem_buffer[];   // 1024 B 对齐，服务于 swizzle-128B
```

`utils::PatternVisitor` 是个零开销的「下标 → 指针」闭包包装器（`operator[](i)` 直接调 lambda），用它替代指针数组，避免在 SMEM/寄存器里存 stage 指针表：

| 区段 | 起址 | 大小 | 访问器 |
| --- | --- | --- | --- |
| C/D store ring | `smem_buffer + 0` | `SMEM_CD_SIZE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd) * kNumTMAStoreStages` | `smem_cd[i]` |
| A ring | `smem_buffer + SMEM_CD_SIZE` | `kNumStages * SMEM_A_SIZE_PER_STAGE`，`SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * 2` | `smem_a[i]` |
| B ring | `+ kNumStages * SMEM_A_SIZE_PER_STAGE` | `kNumStages * SMEM_B_SIZE_PER_STAGE`，`SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * 2` | `smem_b[i]` |
| Barriers | `+ kNumStages * SMEM_B_SIZE_PER_STAGE` | 见 §5.2 | `full_barriers[i]` 等 |
| TMEM 基址 | barriers 之后 | 4 B | `tmem_ptr_in_smem` |

三个 `DG_STATIC_ASSERT(... % 1024 == 0)` 保证每个区段起点都 1024 B 对齐——这是 swizzle-128B 的硬件要求（一个 swizzle atom 是 8 行 × 128 B = 1 KB，若基址不按 1 KB 对齐，TMA 写入的 swizzle 图案与 UMMA 描述符解读的图案会错位）。

### 5.2 Barrier 区（含一段历史包袱）

以 `Barrier`（= `cutlass::arch::ClusterTransactionBarrier`，8 B）为单位，`barrier_start_ptr` 起：

| 索引区间 | 名称 | 个数 | `init()` 计数 |
| --- | --- | --- | --- |
| `[0, S)` | `full_barriers` | `kNumStages` | `kNumMulticast` |
| `[S, 2S)` | `empty_barriers` | `kNumStages` | `1` |
| `[2S, 2S+2)` | `tmem_full_barriers` | 2 | `1` |
| `[2S+2, 2S+4)` | `tmem_empty_barriers` | 2 | `kNumMulticast * kNumUMMAStoreThreads` |
| `[2S+4, 3S+4)` | **（空洞，未使用）** | `S` | — |
| `3S+4` | `tensor_core_full_barrier` | 1 | `1`（仅当 `kTensorCoreUtilControl < 100`） |
| `3S+5`（字节偏移 `+4`） | `tmem_ptr_in_smem` | 4 B | — |

（`S = kNumStages`）

那个 `S` 大小的空洞来自 `tensor_core_full_barrier = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2`：索引算术按「每 stage **三**组 barrier」排布，第三组是 FP8/FP4 1D1D kernel 的 *with-SF full barriers*。BF16 没有 scale factor，于是这 `kNumStages * 8` 字节被跳过但**不回收**。host 侧 `smem_barriers = 32*8*3 + 2*8*2 + 8` 与之精确呼应，所以整体仍是紧的。

初始化由 warp 1 的单个 lane 完成，随后：

```cpp
cutlass::arch::fence_barrier_init();     // fence.mbarrier_init.release.cluster
```

注释写明目的：*Make initialized barrier visible in async proxy*。mbarrier 会被 TMA/UMMA 这些**异步代理**访问，普通的 `__syncthreads()` 不足以建立 generic proxy → async proxy 的可见性，必须用这条 cluster 作用域的 release fence。之后第 172 行做 `cluster_sync_with_relaxed_arrive()`（2-CTA）或 `__syncthreads()`（1-CTA），确保 peer CTA 也能看到 leader 的 barrier 状态、以及 warp 2 写入的 `tmem_ptr_in_smem`。

### 5.3 A/B stage 内部排布

**K-major（最常见）**：TMA box 是 `(inner = BLOCK_K = 64 elem = 128 B, outer = LOAD_BLOCK_M)`，swizzle atom = 8 行 × 128 B。一个 stage 就是 `LOAD_BLOCK_M/8` 个 atom 沿 M 方向线性堆叠：

```
smem_a[s]  (LOAD_BLOCK_M=128, BLOCK_K=64, bf16 → 16 KB)
┌──────────────────────── atom 0 : rows   0.. 7 ────────────────────────┐  ← SBO = 1024 B
│ row r: 128 B = 8 个 16-B bank group，物理位置 g' = g ^ (r % 8)         │
├──────────────────────── atom 1 : rows   8..15 ────────────────────────┤
│ ...                                                                   │
├───────────────────────  atom 15 : rows 120..127 ──────────────────────┤
└───────────────────────────────────────────────────────────────────────┘
```

`^ (r % 8)` 的 bank-group 置换就是 `CU_TENSOR_MAP_SWIZZLE_128B` / `cute::UMMA::LayoutType::SWIZZLE_128B` 定义的图案，由 TMA 硬件在写入时施加、由 UMMA 硬件在读取时反解，软件两侧都不需要参与。它的作用是把「同一列的 8 个元素」打散到 8 个不同的 bank group，消除 tensor core 按列取数时的 SMEM bank conflict。

**MN-major**：TMA box 变成 `(inner = LOAD_BLOCK_MN, outer = BLOCK_K)`，`BLOCK_INNER_ATOM = swizzle/elem = 64`，于是 `tma::copy` 内部会循环 `LOAD_BLOCK_MN / 64` 次，每次的目的地址是 `smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM`。即 SMEM 布局是「**K 外、MN-atom 内**」：先把第 0 个 64 宽的 MN atom 的全部 `BLOCK_K` 行放完，再放第 1 个 atom。UMMA 描述符的 `stride_k` 也随之从 K-major 的 `1` 变成 `get_inner_block_atom_size<...>()`（§9.4）。

### 5.4 C/D stage 内部排布

非 swap：一个 stage 是 `STORE_BLOCK_M` 行 × `kSwizzleCDMode`（128 B），即 `STORE_BLOCK_M` 个 128-B 行；行内同样是 8 个 16-B bank group 的 `^ (r % 8)` 置换。每个 stage 只覆盖 `STORE_BLOCK_N = 128/sizeof(cd)` 个 N 元素（bf16 → 64，fp32 → 32），正好是一个 swizzle atom 宽，所以一个 stage 对应**一条** TMA store。

swap-AB：一个 stage 是 `STORE_BLOCK_M(16) × STORE_BLOCK_N(128)`，被切成 `STORE_BLOCK_N / STORE_BLOCK_N_ATOM = 2` 个 atom，每 atom `16 × 128 B`；4 个 warp 两两负责一个 atom（§11.2）。

### 5.5 UMMA 越界读的静态防护

第 93–94 行：

```cpp
static constexpr uint32_t UMMA_A_SIZE_PER_STAGE =
        math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);
DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages,
                 "Memory out of bound for UMMA");
```

因为 `UMMA_M` 恒为 128（或 256），当 `BLOCK_M = 32/64` 时 `LOAD_BLOCK_M < 128`，UMMA 仍会**按 128 行去读 A**，越过的部分落在后续 A stage、乃至 B ring 上。那些行算出来的 D 在 epilogue 里根本不会被读（只读 `STORE_BLOCK_M = BLOCK_M` 行），所以读到垃圾无害——但**必须落在本 CTA 已分配的动态 SMEM 内**，否则是非法访问。化简后（两边同除 `BLOCK_K * 2`）该断言等价于：

```
128 ≤ LOAD_BLOCK_M + LOAD_BLOCK_N * kNumStages
```

### 5.6 Worked Example：8192 × 8192 × 8192，BF16→BF16，B200（148 SM）

启发式选出的配置：`swap_ab=0, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, cluster=(2,1)`，`swizzle A/B/CD = 128`，`num_stages = 6`。

推导链：

```
kNumMulticast     = 2      (cluster_m=2, cluster_n=1)
kIsMulticastOnA   = false  (cluster_n == 1)
LOAD_BLOCK_M      = 128 / 1 = 128
LOAD_BLOCK_N      = 256 / 2 = 128
UMMA_M            = 128 * 2 = 256
UMMA_N            = BLOCK_N = 256        UMMA_K = 16
STORE_BLOCK_M     = min(128,128) = 128
STORE_BLOCK_N     = 128 / 2 = 64         kNumUMMAStoreThreads = 128
kDoMergeStages    = false  (num_stages = 6 < 8)
kMayHaveTailKBlock= false  (SHAPE_K = 8192 编译期常量，8192 % 64 == 0)
kNumAccumTmemCols = 2 * 256 = 512 → kNumTmemCols = 512   （恰好占满 TMEM）
kNum1DBlocksPerGroup: candidate 8 → 8*128 + 19*256 = 5888
                      candidate 16 → 16*128 + 10*256 = 4608   ⇒ 取 16
num_blocks = 64 * 32 = 2048,  num_waves = ceil(2048/148) = 14,  last_wave_util = 124
```

SMEM 字节表（单 CTA）：

| 偏移 | 大小 | 内容 |
| --- | --- | --- |
| 0 | 32 768 | C/D ring：2 × (128 行 × 128 B) |
| 32 768 | 98 304 | A ring：6 × (128 × 64 × 2 B = 16 KB) |
| 131 072 | 98 304 | B ring：6 × (128 × 64 × 2 B = 16 KB) |
| 229 376 | 184 | barriers：`3*6 + 5 = 23` 个 Barrier |
| 229 560 | 4 | `tmem_ptr_in_smem` |
| **device 合计** | **229 564** | |
| **host 申请** | **230 188** | `33 580 (extra) + 6 × 32 768` |

每个 k_block：单 CTA 载入 32 KB（A 16 KB + B 16 KB），CTA-pair 计算 `256 × 256 × 64 × 2 = 8.39 MFLOP`。若无 2-CTA 切分，每 CTA 需要自己存整块 `256 × 64` 的 B（32 KB），stage 数会从 6 掉到 4——这就是 `cta_group::2` 的直接收益。

---

## 6. Tensor Memory 布局

### 6.1 TMEM 与地址格式

SM100 每个 SM 有 256 KB 的 Tensor Memory，组织为 **128 行（datapath / lane）× 512 列 × 32 bit**。TMEM 地址是 32 位：

```
bit 31        16 15         0
┌──────────────┬─────────────┐
│  row (lane)  │   column    │
└──────────────┴─────────────┘
```

代码里两处直接操作这个格式：

- 累加器基址 `accum_stage_idx * UMMA_N` —— 纯列偏移，行号由硬件按 warp/lane 隐式定位（§3.2 的注释）。
- swap-AB epilogue 的 `cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, v4..v7)` —— `0x00100000` 是 `1 << 20`，即行域 `+16`，把第二次加载定位到第 16–31 行（第一次是 0–15 行，`16dp` = 16 datapath）。

### 6.2 分配 / 释放协议

```cpp
using Allocator = cute::conditional_t<kNumMulticast == 1,
                                      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

// prologue，warp 2 全体 32 lane（tcgen05.alloc 是 .sync.aligned，需整 warp）
Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);

// epilogue，warp 0 全体
Allocator().free(0, kNumTmemCols);
```

- 分配结果写进 SMEM 的 `tmem_ptr_in_smem`，再由 `__syncthreads()` / `cluster_sync` 广播给所有角色。
- `Allocator2Sm` 用于 `cta_group::2`：一次分配覆盖 CTA-pair 的两份 TMEM。
- 分配**之前**必须先 `cluster_sync_with_relaxed_arrive()`（第 102 行），注释：*Synchronize the cluster before 2-CTA TMEM allocation*。2-CTA 的 alloc 是 pair 级操作，两个 CTA 必须同时在场。
- 释放传 `0` 作为基址——正是 §3.2 那条断言保证的前提。
- `kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>()` 向上取整到 `{32, 64, 128, 256, 512}`，因为 `tcgen05.alloc` 只接受 2 的幂且 ≥ 32。

### 6.3 累加器双缓冲

```cpp
constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;   // 2 * UMMA_N
accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;         // 0,1,0,1,...
accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;   // 0,0,1,1,0,0,...
```

| 缓冲 | 列区间 | 内容 |
| --- | --- | --- |
| `acc[0]` | `[0, UMMA_N)` | 偶数 iter 的输出块 |
| `acc[1]` | `[UMMA_N, 2*UMMA_N)` | 奇数 iter 的输出块 |

MMA warp 与 epilogue warp **各自独立**用 `current_iter` 算出同一对 `(accum_stage_idx, accum_phase_idx)`，不需要任何显式传递。这正是 §4.1「复制式调度器 + `current_iter` 当逻辑时钟」的直接收益。

握手：MMA warp 在某块最后一个 k_block 上 `tcgen05.commit` → `tmem_full_barriers[accum_stage_idx]`；epilogue 读到寄存器后立刻 `arrive(0u)` → `tmem_empty_barriers[accum_stage_idx]`（§11.4 的「早释放」）。MMA warp 在开始下一块前 `tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1)`。

`UMMA_N = 256` 时 `kNumAccumTmemCols = 512`，双缓冲恰好占满 TMEM，这也是 host 侧 `2 * umma_n > 512 → 跳过候选` 的由来。

### 6.4 UMMA 的累加目标

`tcgen05.mma` 的第 0 个操作数 `[tmem_c]` 就是 `accum_stage_idx * UMMA_N`：

```cpp
mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
           kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
//                       ↑ scale_c 谓词
```

`scale_c` 在 PTX 里被翻成 `setp.ne.b32 p, %4, 0;` 后作为 `tcgen05.mma` 的尾随谓词：`p = false` 时 `D = A*B`（丢弃 TMEM 原值），`p = true` 时 `D = A*B + D`。所以一个输出块的**第一条** UMMA（`k_block_idx == 0 && kUMMAKIdx == 0`）传 0 完成清零，其余全部传 1 累加——省掉了单独 memset TMEM 的一步，也让上一轮遗留的脏数据自动失效。

---

## 7. 矩阵 Tiling 层次

### 7.1 六级 tiling

以 §5.6 的例子（`BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, cluster=2`）为例，从全局到指令共六级：

| 级别 | 尺度 | 承载者 | 说明 |
| --- | --- | --- | --- |
| L0 全局 | `M × N × K` | grid | persistent，`gridDim.x = kNumSMs` |
| L1 wave | `kNumSMs` 个输出块 | grid 的一轮 | `iter` 递增一次；L2 swizzle 在此层重排 |
| L2 cluster tile | `(2*BLOCK_M) × BLOCK_N = 256 × 256` | CTA-pair | 一条 `cta_group::2` UMMA 覆盖的范围 |
| L3 CTA tile | `BLOCK_M × BLOCK_N = 128 × 256` | 单 CTA | 落在本 CTA 的 TMEM（128 行 × 256 列 × 2 缓冲） |
| L4 k_block / stage | `BLOCK_M × BLOCK_K`（A）+ `LOAD_BLOCK_N × BLOCK_K`（B） | SMEM ring 的一格 | 流水的调度单位，`kNumStages` 格在飞 |
| L5 UMMA atom | `UMMA_M × UMMA_N × UMMA_K = 256 × 256 × 16` | 一条指令 | `BLOCK_K / UMMA_K = 4` 条 UMMA 消费一个 stage |
| L6 swizzle atom | `8 行 × 128 B` | SMEM 物理布局 | TMA 写入与 UMMA 读出共用的最小图案单位 |

K 方向的总迭代数：`num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K)`。注意用的是 `scheduler.current_shape_k` 而非 `shape_k`——k-grouped 变体里每个 group 的 K 长度不同，且这个值是**运行期**的，所以 K 循环不能整体展开，只能展开内层的 `BLOCK_K/UMMA_K` 次 UMMA。

### 7.2 2-CTA（`cta_group::2`）如何切分 A / B / D

`kNumMulticast == 2` 时，一对 CTA（cluster rank 0 = leader，rank 1 = peer）协作完成一个 `UMMA_M = 256` 的 MMA。切分方式取决于 `kIsMulticastOnA`：

**情形 A：`kIsMulticastOnA == false`（非 swap-AB，`cluster_m = 2`）**

```
           N ────────── BLOCK_N = 256 ──────────►
        M  ┌────────────────────────────────────┐
     CTA0  │  D[0:128,   0:256]  → CTA0 TMEM    │   A: CTA0 存 rows   0..127
        ▼  ├────────────────────────────────────┤       (LOAD_BLOCK_M = 128)
     CTA1  │  D[128:256, 0:256]  → CTA1 TMEM    │   B: CTA0 存 cols   0..127
           └────────────────────────────────────┘       CTA1 存 cols 128..255
                                                        (LOAD_BLOCK_N = 128)
```

- 调度器把**相邻的 `m_block_idx`** 分给 cluster 内的两个 CTA（`kIsMulticastOnA=false` → 组内 M 变化最快）。
- A 沿 M 对半切，各 CTA 存自己那 128 行；B 沿 N 对半切，各 CTA 存 128 列。
- 硬件跨 CTA-pair 读取 B，两个 SM 的 tensor core 合起来算出 `256 × 256`，各自把属于自己的 128 行写进本地 TMEM。
- Epilogue 时每个 CTA 用**自己的** `m_block_idx` 算 `base_m_idx`，独立存自己那 `128 × 256`。

**情形 B：`kIsMulticastOnA == true`（swap-AB，`cluster_n = 2`）**

角色互换：cluster 内两个 CTA 拿**相邻的 `n_block_idx`**（组内 N 变化最快），`LOAD_BLOCK_M = BLOCK_M / 2`、`LOAD_BLOCK_N = BLOCK_N = 128`，`UMMA_M = 256` 对应 GEMM 的 N 方向。

两种情形下 `full_barriers[i]->init(kNumMulticast)` 与 `arrive_and_expect_tx(bytes * kNumMulticast)` 的语义都是「pair 内两个 CTA 各自的 TMA 都到齐」。

### 7.3 索引计算：`get_global_idx` 的双模板开关

```cpp
uint32_t m_idx = scheduler.get_global_idx<(kGemmType == GemmType::MGroupedMasked), IndexType::MN>
                     (shape_m, BLOCK_M, m_block_idx);
uint32_t n_idx = scheduler.get_global_idx<(kMajorB == cute::UMMA::Major::K), IndexType::MN>
                     (shape_n, BLOCK_N, n_block_idx, m_block_idx);
uint32_t k_a_idx = scheduler.get_global_idx<(kMajorA == cute::UMMA::Major::MN), IndexType::K>
                     (shape_k, BLOCK_K, k_block_idx, m_block_idx);
uint32_t k_b_idx = scheduler.get_global_idx<(kMajorB == cute::UMMA::Major::MN), IndexType::K>
                     (shape_k, BLOCK_K, k_block_idx, m_block_idx);
```

第一个模板参数是 `kWithGroupOffset`，第二个是索引语义（`MN` / `K` / `SF_K`）。要点：

- `n_idx` 的 `kWithGroupOffset` 是 `kMajorB == K`。因为 B 的 group 维度**总是拼在外维**（见 `make_tma_b_desc` 的 `gmem_outer_dim * num_groups`）：K-major 时外维是 N，所以要加 `group * shape_n`；MN-major 时外维是 K，group 偏移由 `IndexType::K` 那条分支处理。
- `k_a_idx` / `k_b_idx` 的 `kWithGroupOffset` 是 `major == MN`：MN-major 时 K 是外维，k-grouped 的偏移 `current_k_cumsum` / `current_k_start` 加在 K 上；K-major 时 K 是内维，`k_idx = k_block_idx * BLOCK_K` 直接用。
- 源码注释：*`k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major*。
- 第 218 行的 `uint32_t k_idx = k_block_idx * BLOCK_K;` 实际上在后续代码里**没有被使用**（`tma::copy` 收到的是 `k_a_idx` / `k_b_idx`），是残留变量。

随后叠加 2-CTA 偏移（第 225–228 行）：

```cpp
if constexpr (kNumMulticast > 1) {
    m_idx += kIsMulticastOnA ? (block_rank_in_cluster() * load_block_m) : 0;
    n_idx += kIsMulticastOnA ? 0 : (block_rank_in_cluster() * LOAD_BLOCK_N);
}
```

注意 M 方向用的是**运行期**的 `load_block_m`（swap-AB 时来自 `get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast`，以适配 psum layout 的尾块），N 方向用编译期的 `LOAD_BLOCK_N`。

### 7.4 swap-AB：把「参差不齐的维度」放到 UMMA 的 N 上

这是本 kernel 最值得注意的一个架构决策。

`UMMA_M` 被 TMEM 的 datapath 数量钉死在 `128 × kNumMulticast`，**不能运行期改**；而 `UMMA_N` 只是 instruction descriptor 里的一个 5-bit 字段（`n_dim_ = umma_n >> 3`），**可以逐块改写**，粒度 8（非 swap）/ 16（swap，见 `get_aligned_effective_m_in_block` 里的 `UMMA_STEP_N = 16`）。

MoE / m-grouped 场景里每个 group 的有效 token 数（M）是动态且零碎的。若按常规方向做，M 必须向上取整到 `BLOCK_M`，padding 部分的 MMA 全部白算。swap-AB 之后：

```cpp
// 操作数对调
mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N, ...);

// 逐块动态改 UMMA_N = 有效 M
if constexpr (kSwapAB) {
    uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
    mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);   // desc.n_dim_ = umma_n >> 3;
}
```

于是无效 token 直接**不发射 MMA**。代价是 TMEM 里的累加器变成了 `D^T`（行 = n，列 = m），epilogue 必须转置，这就是 `sm100_store_cd_swap_ab.cuh` 用 `stmatrix...trans` 的原因。

### 7.5 Stage 合并：用更大的 `BLOCK_K` 摊薄 `umma_arrive`

第 43–50 行：

```cpp
constexpr bool kDoMergeStages =
    kNumStages_ >= 8 and kGemmType == GemmType::Normal and
    kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
constexpr uint32_t kNumMinStages       = 8;
constexpr uint32_t kNumStagesPerMerge  = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
constexpr uint32_t BLOCK_K             = BLOCK_K_ * kNumStagesPerMerge;   // 64 → 128/192/...
constexpr uint32_t kNumStages          = kNumStages_ / kNumStagesPerMerge;
```

注释说明动机：*this is for reducing the `umma_arrive()` overhead*。每个 k_block 结束时都要做一次 `tcgen05.commit`（+ 可能的 `tmem_full` commit）和一次 `full_barriers[stage]->wait()`，这些是 MMA warp 的**串行开销**。把 2 个 64-宽的 stage 合成 1 个 128-宽的 stage 后，同步次数减半、每次 UMMA 连发数从 4 增到 8，而**总的 SMEM 占用和流水深度（字节数）不变**。

合并只在 `kNumStages_ ≥ 8` 时触发，并保证合并后仍至少有 `kNumMinStages = 8` 个 stage（否则流水深度不足以掩盖延迟）。

合并后 SMEM 布局的关键点：一个 stage 内不再是「128 行 × 128 列」的单一 atom，而是 `kNumStagesPerMerge` 个 **K-atom（64 宽）沿「MN 外、K-atom 内」**排列。这一点在三处保持一致：

1. **TMA**：`tma::copy<BLOCK_K=128, LOAD_BLOCK_M, 128, bf16>` 内部 `BLOCK_INNER_ATOM = 128/2 = 64`，循环 2 次，第 i 次写到 `smem + i * LOAD_BLOCK_M * 64`。
2. **UMMA 描述符**：构造时用 `BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge = 64`（**不是** `BLOCK_K`），保证 `DG_STATIC_ASSERT(kSwizzleMode == BLOCK_ATOM_K * sizeof(dtype))` 即 `128 == 64*2` 仍成立；推进时 `kAtomKIdx = kUMMAKIdx * UMMA_K / BLOCK_ATOM_K`，偏移 `kAtomKIdx * LOAD_BLOCK_M * BLOCK_ATOM_K`。
3. **stage 步长**：`a_desc_lo` 用 `SMEM_A_SIZE_PER_STAGE`（按合并后的 `BLOCK_K` 算）作为 lane 间的步长。

举例：`kNumStages_ = 18` → `kNumStagesPerMerge = 2`、`BLOCK_K = 128`、`kNumStages = 9`；每个 k_block 发 `128/16 = 8` 条 UMMA，`kAtomKIdx ∈ {0,0,0,0,1,1,1,1}`，`kInnerKIdx ∈ {0,16,32,48,0,16,32,48}`。

### 7.6 `BLOCK_M < 128` 时的算力浪费（有意为之）

`UMMA_M = LAYOUT_AD_M * kNumMulticast` 恒为 128 或 256，**与 `BLOCK_M` 无关**。当启发式因 `desc.m ≤ 32` / `≤ 64` 选出 `BLOCK_M = 32` / `64` 时：

- UMMA 仍按 M=128 发射，读 128 行 A（其中 96/64 行是 SMEM 越界垃圾，§5.5），往 TMEM 写 128 行 D；
- epilogue 只读 `STORE_BLOCK_M = BLOCK_M` 行，`kNumUMMAStoreThreads = BLOCK_M` 个线程；
- 结果正确的部分只有前 `BLOCK_M` 行。

第 274–277 行的指令形状断言里虽然列了 `UMMA_M == 64` 的合法分支，但 `UMMA_M` 的推导式永远产生不出 64，那段是通用的形状合法性检查（注释也说明 *CUTLASS does not have such checks except the MMA traits, but we are not using these traits*）。

这是一个**明确的取舍**：小 M 场景本来就是访存/延迟受限（选小 `BLOCK_M` 的目的正是注释里的 *avoid TMA L2 OOB bound*，即不去 GMEM 白读 128 行），MMA 吞吐富余，用一条统一代码路径换掉「M=64 UMMA + 另一套描述符/断言」的复杂度是划算的。

### 7.7 Tail-K

```cpp
constexpr bool kMayHaveTailKBlock = ...;
for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
    ...
    if constexpr (kMayHaveTailKBlock) {
        auto issue_tail_k_block = [&](const uint32_t& remaining_k) {
            const auto num_valid_umma_k = math::ceil_div(remaining_k, UMMA_K);
            utils::for_each_static_prefix(std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(),
                                          num_valid_umma_k, issue_umma);
        };
        const auto is_last_k_block = k_block_idx == num_total_k_blocks - 1;
        if (is_last_k_block) {
            const auto remaining_k = scheduler.current_shape_k - k_block_idx * BLOCK_K;
            if (remaining_k < BLOCK_K) issue_tail_k_block(remaining_k);
            else                       issue_full_k_block();
        } else {
            issue_full_k_block();
        }
    } else {
        issue_full_k_block();
    }
}
```

三层设计：

1. **编译期消除**：`if constexpr (kMayHaveTailKBlock)` —— K 被编译进来且整除时整段不生成代码（§2.1）。
2. **运行期只在最后一个 k_block 判断**：`is_last_k_block` 之外的迭代走 `issue_full_k_block()`，把动态分支的代价压到 1/`num_total_k_blocks`。
3. **前缀展开**：`for_each_static_prefix` 在 `BLOCK_K/UMMA_K ≤ 4` 时用 `switch(num_valid)` 跳到对应的编译期前缀（注释：*Prefix expansion uses switch only for small cases to avoid long SASS*），`> 4` 时退化成运行期谓词的 fold expression。

`ceil_div(remaining_k, UMMA_K)` 向上取整意味着最后一条 UMMA 可能读到最多 15 个 padding 元素——它们由 TMA 的 OOB 零填充保证为 0（`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`），对结果无影响。而 k-grouped 路径有 `DG_STATIC_ASSERT(kKAlignment % UMMA_K == 0)`，`remaining_k` 必是 16 的倍数，`ceil_div` 不会真的向上取。

---

## 8. TMA 加载路径

### 8.1 调用形态

TMA warp（warp 0 的单个 lane）在每个 k_block 上按 `kMajorA/kMajorB` 四选一发射：

```cpp
if constexpr (kMajorA == cute::UMMA::Major::K)
    tma::copy<BLOCK_K,      LOAD_BLOCK_M, kSwizzleAMode, bf16, kIsBatchedMM>(
        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, kNumMulticast, batch_idx);
if constexpr (kMajorA == cute::UMMA::Major::MN)
    tma::copy<LOAD_BLOCK_M, BLOCK_K,      kSwizzleAMode, bf16, kIsBatchedMM>(
        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, kNumMulticast, batch_idx);
// B 同理
```

模板参数序是 `<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype, kIs3DTMA>`，函数参数序是 `(desc, barrier, smem_dst, inner_idx, outer_idx, num_multicast, batch_idx)`。**inner 恒为 SMEM 里连续的那一维**，所以 K-major 时 `(inner, outer) = (k, mn)`，MN-major 时 `(mn, k)`，两个 `if constexpr` 分支只是把实参顺序换了一下。

### 8.2 swizzle-atom 循环

```cpp
constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();
//   = kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t)

#pragma unroll
for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i)
    SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                           EVICT_NORMAL,
                           smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                           inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
```

TMA box 的内维被 host 压到 `swizzle/elem = 64` 个元素（128 B），所以一个逻辑上 `BLOCK_INNER` 宽的块要拆成 `BLOCK_INNER / 64` 条 TMA：

| 场景 | `BLOCK_INNER` | atom | TMA 条数 | SMEM 目的地址步进 |
| --- | --- | --- | --- | --- |
| K-major A，未合并（`BLOCK_K=64`） | 64 | 64 | 1 | — |
| K-major A，合并后（`BLOCK_K=128`） | 128 | 64 | 2 | `LOAD_BLOCK_M * 64` |
| MN-major A，`LOAD_BLOCK_M=128` | 128 | 64 | 2 | `BLOCK_K * 64` |

目的地址步进 `BLOCK_OUTER * BLOCK_INNER_ATOM` 正是 §5.3 描述的「atom 沿外维堆叠」布局，与 `make_umma_desc` 的 SBO/LBO 推导严格对偶。

### 8.3 三种 TMA 变体

```cpp
if (num_tma_multicast == 1) {
    cute::SM90_TMA_LOAD_2D::copy(...);                      // cp.async.bulk.tensor.2d...（单 CTA）
} else {
  #if __CUDA_ARCH__ >= 1000
    cute::SM100_TMA_2SM_LOAD_2D::copy(...);                 // 带 .cta_group::2
  #elif __CUDA_ARCH__ >= 900
    if (cute::block_rank_in_cluster() == 0)
        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(..., (1 << num_tma_multicast) - 1, ...);
  #endif
}
```

- **1-CTA**：`SM90_TMA_LOAD_2D`，Hopper 就有的 `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`。DeepGEMM 在 SM100 上复用它。
- **SM100 2-CTA**：`SM100_TMA_2SM_LOAD_2D`，即带 `.cta_group::2` 修饰的 bulk tensor copy。源码注释点明关键语义：*2-CTA function will send signals to the leader CTA only* —— 两个 CTA **各自发射**自己那一份 load，但 tx-count 只累加到 CTA-pair leader 的 mbarrier 上。这是 §10.3 里 `full_barriers` 计数推导的基础。
- **SM90 multicast**：只由 rank 0 发射一条带 CTA mask 的 multicast load，一份 GMEM 读同时写进 cluster 内所有 CTA 的 SMEM。SM100 路径不用它（2-CTA UMMA 不能动态关闭，见 §4.2）。

Cache hint 统一是 `EVICT_NORMAL`，并且开头有一条静态断言确保 SM90/SM100 两套枚举值一致。对比 [ptx/tma.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/tma.cuh) 里手写的 `tma_load_1d` 用的是 `EVICT_FIRST`（注释：*normally, the loaded part will be evicted soon*），而 store 用 `EVICT_NORMAL`（*the stored part will be used soon*）——本 kernel 的 A/B 走 cute 封装故为 NORMAL。

### 8.4 Descriptor prefetch

```cpp
if (warp_idx == 0) {
    cute::prefetch_tma_descriptor(&tensor_map_a);
    cute::prefetch_tma_descriptor(&tensor_map_b);
    cute::prefetch_tma_descriptor(&tensor_map_cd);
}
```

在 kernel 最开头、**任何同步之前**由 warp 0 全体执行（`prefetch.tensormap` 不是单线程指令）。descriptor 在常量内存里，首次 TMA 访问会有冷启动延迟，提前 prefetch 可以把它藏进 barrier 初始化与 TMEM 分配的时间里。

### 8.5 `expect_tx` 字节数

```cpp
constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
if (is_leader_cta) {
    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
} else {
    full_barriers[stage_idx]->arrive(0u);      // 远程 arrive 到 cluster rank 0
}
```

- `SMEM_*_SIZE_PER_STAGE` 用的是**合并后**的 `BLOCK_K`，与 TMA 实际搬运的字节数一致。
- `× kNumMulticast`：pair 内两个 CTA 各搬一份，tx 都记在 leader 的 barrier 上。
- peer CTA 只做一次「无 tx 的 arrive」，凑够 `init(kNumMulticast)` 的到达计数。
- **顺序**：TMA 先发射（第 233–244 行），`arrive_and_expect_tx` 后执行（第 248–252 行）。这是合法的——mbarrier 的 tx-count 是「期望值累加」，只要在该相位完成前把期望值补上即可；反过来（先 expect 后发 TMA）同样合法但会让 warp 更早阻塞在计数上。

### 8.6 完整的 TMA warp 循环

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast
                                      : LOAD_BLOCK_M;
    const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
        empty_barriers[stage_idx]->wait(phase ^ 1);      // ① 等消费者释放
        /* ② 算 m_idx / n_idx / k_a_idx / k_b_idx / batch_idx，叠加 2-CTA 偏移 */
        /* ③ 发 A、B 的 TMA（各 1~2 条指令） */
        /* ④ arrive_and_expect_tx（leader）/ arrive(0u)（peer） */
    }
}
```

整个 TMA warp 就是这四步的无限重复，没有任何计算。它的推进速度只受 `empty_barriers` 的释放节奏限制，因此可以超前 MMA 多达 `kNumStages` 个 k_block。

---

## 9. UMMA 发射路径

### 9.1 Instruction descriptor

```cpp
auto instr_desc = kSwapAB
    ? cute::UMMA::make_instr_desc<bf16, bf16, float, UMMA_M, UMMA_N, kMajorB, kMajorA>()
    : cute::UMMA::make_instr_desc<bf16, bf16, float, UMMA_M, UMMA_N, kMajorA, kMajorB>();
...
const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
```

- 编码了 A/B 数据类型（BF16）、累加类型（FP32）、MMA 形状（`UMMA_M × UMMA_N`）、A/B 的 major 模式。swap-AB 时把两个 major 也对调。
- `make_runtime_instr_desc` 把这个 32-bit 描述符左移 32 位打包成 `uint64_t`，与 PTX 里的取用方式对应：`"r"(static_cast<uint32_t>(desc >> 32))` —— 只取高 32 位作为 `tcgen05.mma` 的第 4 个操作数。
- swap-AB 时 `update_instr_desc_with_umma_n(instr_desc, umma_n)` 改的是 `desc.n_dim_ = umma_n >> 3`，而 `runtime_instr_desc` 在 k_block 循环**内部**每轮重新计算（第 321 行），所以逐块的 `umma_n` 变化能立刻生效。

### 9.2 SMEM descriptor（`SmemDescriptor`）

`make_smem_desc()` 填充 7 个字段：

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `version_` | `1` | SM100 版本标记 |
| `lbo_mode_` | `0` | legacy 模式 |
| `layout_type_` | `to_umma_layout_type<...>()` | `SWIZZLE_NONE / 32B / 64B / 128B / 128B_BASE32B` |
| `start_address_` | `cast_smem_ptr_to_uint(p) >> 4` | SMEM 地址，**16 B 为单位** |
| `base_offset_` | `0` | — |
| `stride_byte_offset_` (SBO) | `stride_byte_offset >> 4` | atom 间在某一维上的字节步长 |
| `leading_byte_offset_` (LBO) | `leading_byte_offset >> 4` | atom 间在另一维上的字节步长 |

`to_umma_layout_type()` 有一个特例：`dtype == float && major == MN`，或显式 `kUseBase32`，返回 `SWIZZLE_128B_BASE32B`；对应 `get_atom_base()` 返回 32 而非 16，进而 `num_non_contiguous = 128 / 32 = 4`（常规是 `128 / 16 = 8`）。BF16 路径永远走常规分支，`num_non_contiguous = 8`。

**K-major** 的 SBO/LBO 推导：

```cpp
DG_STATIC_ASSERT(kSwizzleMode * kPackFactor == BLOCK_K * sizeof(dtype_t));   // 128 == 64 * 2
const uint32_t stride_byte_offset  = num_non_contiguous * BLOCK_K * sizeof(dtype_t) / kPackFactor;  // 8*64*2 = 1024
const uint32_t leading_byte_offset = 0;
```

注释解释了为什么 LBO 是 0：*on K, there is only 1 atom as asserted previously*——那条静态断言保证每个 block 在 K 方向恰好只有一个 swizzle atom，于是「K 方向 atom 间步长」无意义。SBO = 一个 atom 的字节大小 = `8 行 × 128 B = 1024 B`，即 MN 方向相邻 atom 的步长。

**MN-major**：

```cpp
constexpr uint32_t BLOCK_MN_ATOM = tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();  // 64
DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);        // 不允许 atom 内的 MN 偏移
uint32_t stride_byte_offset  = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);   // 8*64*2 = 1024
uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
if constexpr (kSwizzleMode == 16) math::swap(stride_byte_offset, leading_byte_offset);
```

注释给出了 SBO/LBO 的语义约定：swizzle 时 `{SBO, LBO}` 是 atom 在 `{K, MN}` 上的步长；非 swizzle（`kSwizzleMode == 16`，*means non-swizzling but interleaving*）时是 `{MN, K}`，所以要 swap。

`kPackFactor` 只对 packed FP4 是 2（*Packed FP4 stores two logical elements per byte in SMEM*），BF16 恒为 1，并有 `DG_STATIC_ASSERT(kPackFactor == 1 or sizeof(dtype_t) == 1)` 兜底。

### 9.3 用 warp 的 32 个 lane 存 per-stage 描述符

```cpp
DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
...
const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
```

这是一个很巧的优化：`kNumStages` 个 stage 的描述符低 32 位（含 `start_address_`）**分散存放在 MMA warp 的 lane 0 … lane `kNumStages-1` 的寄存器里**，需要哪个 stage 就 `__shfl_sync` 广播出来。

- 省掉了 SMEM 里的描述符表（以及随之而来的 `ld.shared` 延迟与 bank 压力）；
- 省掉了每 stage 重算 `make_umma_desc` 的指令；
- `/ 16` 与 `start_address_` 的 16-B 粒度一致，`SMEM_*_SIZE_PER_STAGE` 是 1024 的倍数（§5.1 断言）所以整除无损；
- 代价是 `kNumStages ≤ 32` 这条硬约束（与 host 的 `kNumMaxStages = 32` 对应）。

`a_desc` / `b_desc` 这两个 64-bit 结构体被 `issue_umma` lambda 按引用捕获，循环里**只改 `.lo`**——`.hi`（SBO / base_offset / layout_type / version）是 stage 无关的常量。

### 9.4 K 内层展开与描述符推进

```cpp
auto issue_umma = [&]<uint32_t kUMMAKIdx>() {
    constexpr uint32_t kAtomKIdx  = kUMMAKIdx * UMMA_K / BLOCK_ATOM_K;      // 第几个 64-宽 K atom
    constexpr uint32_t kInnerKIdx = kUMMAKIdx * UMMA_K % BLOCK_ATOM_K;      // atom 内 K 偏移
    a_desc.lo = advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, bf16>(
                    a_desc_base_lo, kAtomKIdx * LOAD_BLOCK_M * BLOCK_ATOM_K, kInnerKIdx);
    b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, bf16>(
                    b_desc_base_lo, kAtomKIdx * LOAD_BLOCK_N * BLOCK_ATOM_K, kInnerKIdx);
    kSwapAB ? mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N, kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc)
            : mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N, kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
};
auto issue_full_k_block = [&]() {
    utils::for_each_static_until<BLOCK_K / UMMA_K>(
        std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), issue_umma);
};
```

`for_each_static_until` 是一个 fold expression：`((kIdx < kNumValid ? func.template operator()<kIdx>() : void()), ...)`。`issue_umma` 是带显式模板参数列表的 generic lambda（C++20），所以 `kUMMAKIdx` 是**编译期常量**，`kAtomKIdx` / `kInnerKIdx` / 元素偏移全部常量折叠，`BLOCK_K/UMMA_K` 条 UMMA 完全展开且各自的描述符增量是立即数。

`advance_umma_desc_lo` 的算式：

```cpp
return base + (((offset + k_idx * stride_k) * sizeof(dtype_t)) >> 4u);
// stride_k = (major == K) ? 1 : get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>()
```

- **K-major**：`offset` 是 K-atom 的元素偏移（`kAtomKIdx * LOAD_BLOCK_M * 64`，因为 atom 沿 MN 外维堆叠），`k_idx` 是 atom 内的 K 元素偏移，`stride_k = 1`（K 在 atom 内连续）。合计 `× 2 B >> 4` 换成 16-B 单位。未合并时 `kAtomKIdx` 恒为 0，`kInnerKIdx ∈ {0,16,32,48}` → `.lo` 依次 `+0, +2, +4, +6`（每条 UMMA 消耗 `16 × 2 B = 32 B = 2` 个 16-B 单位）。
- **MN-major**：`stride_k = BLOCK_MN_ATOM = 64`，因为 MN-major 下 K 是**外维**，K 前进 1 要跨过一整个 64 宽的 MN atom。

### 9.5 PTX 指令

```cpp
using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_F16BF16_SS,
                                                      ptx::SM100_MMA_F16BF16_2x1SM_SS>;
```

[ptx/tcgen05.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh) 里的实现：

```cpp
struct SM100_MMA_F16BF16_SS {
    fma(desc_a, desc_b, tmem_c, scale_c, desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
               "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};
// 2x1SM 版本仅把 cta_group::1 换成 cta_group::2
```

操作数语义：

| 位置 | 约束 | 含义 |
| --- | --- | --- |
| `[%0]` | `"r"` | TMEM 累加器地址（列偏移 `accum_stage_idx * UMMA_N`） |
| `%1` | `"l"` | A 的 SMEM 描述符（64-bit） |
| `%2` | `"l"` | B 的 SMEM 描述符（64-bit） |
| `%3` | `"r"` | instruction descriptor（`desc >> 32`） |
| `p` | 由 `%4` 生成 | scale-D 谓词：0 → 覆写，1 → 累加 |

后缀 `_SS` = 两个操作数都来自 Shared memory（对比 `_RS` 变体的 A 来自寄存器）。BF16 走 `kind::f16`；同文件还有 `mxf8f6f4` / `f8f6f4` / `mxf4`（带 `.block_scale` 与额外的 `[tmem_sfa], [tmem_sfb]` 操作数）以及 `tcgen05.mma.ws`（weight-stationary）变体，本 kernel 都不用。

`asm volatile` 且无输出操作数——防止编译器把「重复」的 MMA 指令合并或重排。

### 9.6 MMA warp 的完整循环

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    accum_stage_idx = current_iter % 2;  accum_phase_idx = (current_iter / 2) & 1;
    tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);   // ① 等 epilogue 放掉累加器
    ptx::tcgen05_after_thread_sync();                                  // ② tcgen05.fence::after_thread_sync
    /* 定义 umma_arrive / empty_barrier_arrive lambda；swap-AB 时改 instr_desc 的 n_dim */
    for (k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
        full_barriers[stage_idx]->wait(phase);                         // ③ 等 TMA 到货
        ptx::tcgen05_after_thread_sync();
        if (elect_one_sync()) { /* ④ 发 BLOCK_K/UMMA_K 条 UMMA */ }
        __syncwarp();
        empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);   // ⑤ tcgen05.commit
        /* ⑥ 可选的 tensor core 降速（§9.7） */
    }
}
```

```cpp
auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
    if (do_tmem_full_arrive)
        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
    __syncwarp();
};
auto umma_arrive = [](const uint64_t* barrier) {
    if constexpr (kNumMulticast == 1) cutlass::arch::umma_arrive(barrier);
    else                              cutlass::arch::umma_arrive_multicast_2x1SM(barrier, (1 << kNumMulticast) - 1);
};
```

`tcgen05.commit` 的作用是：让指定的 mbarrier **跟踪此前由本 warp 发起的所有 `tcgen05.mma`**，当它们全部完成时自动产生一次 arrive。这是异步 MMA 与 mbarrier 之间唯一的桥梁——软件完全不需要轮询 MMA 状态。

两个 commit 的分工：

- `empty_barriers[stage_idx]`：**每个** k_block 都 commit，通知 TMA warp「这个 SMEM stage 已被读完，可以覆盖」。
- `tmem_full_barriers[accum_stage_idx]`：**只在最后一个** k_block commit（`do_tmem_full_arrive = (k_block_idx == num_total_k_blocks - 1)`），通知 epilogue「整块累加完成」。因为 commit 跟踪的是「此前全部 MMA」，最后一次 commit 天然覆盖了整块的所有 UMMA。

`umma_arrive_multicast_2x1SM` 的 CTA mask 是 `(1 << kNumMulticast) - 1 = 0b11`，一次 commit 同时向 pair 内两个 CTA 的同名 barrier 各投递一次 arrive——peer CTA 的 TMA warp 也需要知道 leader 的 UMMA 读完了自己那份 SMEM。

第 367 行的注释是一个重要的正确性说明：*No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`*。而 epilogue 侧读 TMEM 后通知 `tmem_empty` 时，**必须**显式调用 `ptx::tcgen05_before_thread_sync()`（因为那里用的是普通 `mbarrier.arrive`，没有 commit 帮忙）。

### 9.7 Tensor Core 利用率控制（防掉频）

```cpp
DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
if constexpr (kTensorCoreUtilControl < 100) {
    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));
    __syncwarp();
    tensor_core_full_barrier->wait(tensor_core_phase);
    tensor_core_phase ^= 1;

    constexpr static uint64_t kNumUMMACycles  = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
    const auto start_clock = clock64();
    if (cute::elect_one_sync())
        while (clock64() - start_clock < kNumDummyCycles) {}
    __syncwarp();
}
```

注释：*Let tensor cores relax for lower possibility of frequency drop*。这是一个**主动降速**机制：

- `kNumUMMACycles = 2·M·N·K / 8192` —— 一个 k_block 的浮点运算量除以 8192 FLOP/cycle（该 kernel 假定的 tensor core 峰值吞吐）得到理论执行周期。
- `kNumDummyCycles = (100 - util)/util × kNumUMMACycles` —— 要让占空比降到 `util%`，需要在每 `kNumUMMACycles` 的实际计算后插入的空转周期。`util = 50` → 空转与计算等长。
- 空转前先用 `tcgen05.commit` + `wait` **确认上一批 UMMA 真的做完了**，否则量出来的占空比不准。
- 由 `device_runtime->set_tc_util()` / Python `deep_gemm.set_tc_util()` 控制，默认 100（整段 `if constexpr` 不生成代码，`tensor_core_full_barrier` 也不 `init`）。

用途是在功耗受限的集群上做可复现的性能对比、或验证「降频是否是真凶」。对纯性能跑分应始终保持 100。

---

## 10. 生产者-消费者同步

### 10.1 barrier 全景

全部同步对象都是 `cutlass::arch::ClusterTransactionBarrier`（8 B SMEM 驻留的 mbarrier），共 5 类：

| barrier | 个数 | `init` 计数 | 等待者 | 到达者 | 语义 |
| --- | --- | --- | --- | --- | --- |
| `full_barriers[s]` | `kNumStages` | `kNumMulticast` | MMA warp（leader） | leader：`arrive_and_expect_tx(bytes×mc)`<br>peer：`arrive(0u)`<br>＋TMA 硬件的 tx 完成 | SMEM stage `s` 已装好，A/B 可读 |
| `empty_barriers[s]` | `kNumStages` | `1` | TMA warp（**每个** CTA） | `tcgen05.commit`（`umma_arrive`，2-CTA 时 multicast 到 pair 两边） | stage `s` 已被 UMMA 读完，可覆写 |
| `tmem_full_barriers[a]` | 2 | `1` | Epilogue warps（**每个** CTA） | 最后一个 k_block 上的 `tcgen05.commit` | TMEM 累加器 `a` 已写满 |
| `tmem_empty_barriers[a]` | 2 | `kNumMulticast × kNumUMMAStoreThreads` | MMA warp（**仅** leader） | 每个 epilogue 线程的 `arrive(0u)`（远程投递到 cluster rank 0） | 累加器 `a` 已读完，可重新累加 |
| `tensor_core_full_barrier` | 1 | `1`（仅 `tc_util < 100`） | MMA warp | `tcgen05.commit` | 上一批 UMMA 已完成（用于算占空比） |

两条流水嵌套关系：

```
【A/B SMEM 环】  生产者 = TMA warp        消费者 = MMA warp
     full  : TMA → MMA        empty : MMA → TMA

【TMEM 累加器环】 生产者 = MMA warp       消费者 = Epilogue warps
     tmem_full  : MMA → Epilogue      tmem_empty : Epilogue → MMA

【C/D SMEM 环】  生产者 = Epilogue(STSM)  消费者 = TMA store 引擎
     不用 mbarrier，而用 cp.async.bulk.wait_group.read + NamedBarrier
```

MMA warp 同时是上游的消费者和下游的生产者，是整条链的**唯一串行点**——这也是为什么它只用一个 warp（甚至一个 lane）：UMMA 是异步指令，发射本身极便宜，真正的瓶颈在 `tcgen05.commit` 的次数（§7.5 的 stage 合并就是为了摊薄它）。

### 10.2 parity 相位约定与「首轮免等」

`Barrier::wait(p)` 对应 `mbarrier.try_wait.parity.shared::cta.b64 P, [bar], p`，含义是「等待奇偶性为 `p` 的那个相位**完成**」。两类角色的相位推进方式不同：

**① A/B 环：显式相位变量**

```cpp
uint32_t stage_idx = 0, phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++ k_block_idx;
    stage_idx = (stage_idx + 1) % kNumStages;
    phase ^= stage_idx == 0;          // 只在回绕时翻转
};

// 生产者（TMA）：empty_barriers[stage_idx]->wait(phase ^ 1);
// 消费者（MMA） ：full_barriers[stage_idx]->wait(phase);
```

第一轮（`phase = 0`）：

- TMA 等 `empty` 的 parity **1**。新建的 mbarrier 处于 phase 0 且未完成，对 parity 1 的 `try_wait` 立即成功——这就是「缓冲初始为空，生产者前 `kNumStages` 轮直接穿过」的标准手法。
- MMA 等 `full` 的 parity **0**，必须等第一次 tx 完成。

回绕一次后 `phase = 1`，两边等待的 parity 同时翻转，协议自洽。

**② TMEM 环：从 `current_iter` 直接推导**

```cpp
accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;      // 0,1,0,1,…
accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1; // 0,0,1,1,0,0,…

// 生产者（MMA）入口 ：tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
// 消费者（Epilogue）入口：tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
```

同样地，`iter = 0` 时 MMA 等 parity 1 → 立即穿过（累加器初始可用）。好处是**无需维护额外的状态变量**，两个角色各自从同一个 `current_iter` 算出完全一致的 `(stage, phase)`。

注意 `tensor_core_phase` 用的是手动 `^= 1`（每轮翻转），因为它只有一个 barrier、不区分 stage。

### 10.3 arrive 计数逐条推导

**`full_barriers[s]->init(kNumMulticast)`**（源码注释：*Arrive only at the leader CTA*）

- leader CTA：`arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast)` —— 本地 arrive 1 次 + 设置期望 tx。
- peer CTA：`arrive(0u)` —— `mapa.shared::cluster` 把地址映射到 cluster rank 0 后 `mbarrier.arrive.shared::cluster`，即远程 arrive 到 **leader** 的那份 barrier。
- TMA 硬件：`SM100_TMA_2SM_LOAD_2D` 的 `.cta_group::2` 语义使两个 CTA 各自 load 的 tx 都只记到 leader 的 barrier（*2-CTA function will send signals to the leader CTA only*）。
- 合计：`kNumMulticast` 次 arrive + `kNumMulticast × (A+B)` 字节 tx → 相位完成。只有 leader 的 MMA warp 需要等（UMMA 只由 leader 发射），peer 本地那份 `full_barriers` 无人等待。

**`empty_barriers[s]->init(1)`**（注释：*Arrive at all CTAs*）

- `kNumMulticast == 1`：`cutlass::arch::umma_arrive` → `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64` → 本地 1 次 arrive。
- `kNumMulticast == 2`：`umma_arrive_multicast_2x1SM(bar, 0b11)` → `tcgen05.commit.cta_group::2.…multicast::cluster.b64 [bar], mask` → **pair 内每个 CTA 的同名 barrier 各收 1 次 arrive**。
- 所以两个 CTA 各自的 `empty_barriers[s]` 都是「计数 1、由 leader 的 commit-multicast 填满」，而两个 CTA 的 TMA warp 各自等自己本地的那份。注释 *Arrive at all CTAs* 正是这个意思。

**`tcgen05.commit` 是 warp 级指令（重要）**

第 364–368 行的结构是：

```cpp
if (cute::elect_one_sync()) { /* 只有 1 个 lane 发 UMMA */ }
__syncwarp();
empty_barrier_arrive(...);      // ← 在 elect 块之外，warp 1 的 32 个 lane 全部执行 umma_arrive
```

而 `empty_barriers[s]->init(1)`。若每个 lane 都产生一次 arrive，相位会被多推 31 次，parity 协议立即崩。因此可反推出：**`tcgen05.commit` 是 warp 级（收敛）指令，整 warp 执行只产生一次 arrive**，它代表的是「本 warp 此前发起的全部 `tcgen05.mma` 完成」这一事件，而非每线程事件。

旁证：`sm100_bmk_bnk_mn.cuh` 里同样是 `empty_barriers[i]->init(1)` + 在 warp 作用域直接调 `cutlass::arch::umma_arrive(...)`；`sm100_fp8_fp4_gemm_1d1d.cuh` 也是同样的 `empty_barrier_arrive` 结构。三个 SM100 kernel 一致，说明这是有意的写法。这也解释了 `empty_barrier_arrive` 末尾那个 `__syncwarp()`：保证 commit 在 warp 收敛状态下发出，并与下一轮 `stage_idx` 的使用隔开。

**`tmem_full_barriers[a]->init(1)`**

注释：*the tensor memory accumulator pipeline has nothing to do with multicasting*——意思是这个环不属于 A/B 的 multicast 切分体系，但它仍然用同一个 `umma_arrive` 封装，因此 2-CTA 时同样 multicast 到两个 CTA，每边各 1 次 arrive，而两个 CTA 的 epilogue warp 各自等本地那份（因为两个 CTA 的 TMEM 各存自己那 128 行，都需要被读走）。

**`tmem_empty_barriers[a]->init(kNumMulticast * kNumUMMAStoreThreads)`**

- epilogue 里 `tmem_empty_barrier->arrive(0u)` 没有 `elect_one_sync()` 包裹，所以**每个参与线程都 arrive 一次** → 单 CTA `kNumUMMAStoreThreads` 次。
- `arrive(0u)` 的目标固定是 cluster rank 0，所以 pair 内两个 CTA 的到达全部汇到 leader → `× kNumMulticast`。
- 只有 leader 的 MMA warp 等它（因为只有 leader 发 UMMA）。peer CTA 本地的 `tmem_empty_barriers` 是死对象。

### 10.4 fence 序列

异步代理（TMA / tensor core）与普通线程之间的可见性需要专用 fence，本 kernel 共四处：

| 位置 | 指令 | 作用 |
| --- | --- | --- |
| barrier 初始化后（第 167 行） | `cutlass::arch::fence_barrier_init()`<br>= `fence.mbarrier_init.release.cluster` | 让初始化后的 mbarrier 对 **async proxy** 可见（TMA/UMMA 会直接读写它们），cluster 作用域保证 peer CTA 也看得到 |
| 每次等完 barrier、发 tcgen05 指令前（第 285/317 行） | `ptx::tcgen05_after_thread_sync()`<br>= `tcgen05.fence::after_thread_sync` | 把「同步点之后」的 TMEM/MMA 操作与同步点正确排序，防止 tensor core 异步管线越过 barrier 提前取数 |
| epilogue 读完 TMEM、通知 `tmem_empty` 前（`sm100_store_cd.cuh:113`） | `ptx::tcgen05_before_thread_sync()`<br>= `tcgen05.fence::before_thread_sync` | 保证所有 `tcgen05.ld` 已完成并排序在 arrive 之前。**MMA 侧不需要**这条，因为 `tcgen05.commit` 隐含了它（源码第 367 行注释） |
| TMEM load 之后、用寄存器值之前（`sm100_store_cd.cuh:91/99`） | `cutlass::arch::fence_view_async_tmem_load()`<br>= `tcgen05.wait::ld.sync.aligned` | `tcgen05.ld` 也是异步的，必须等它真正写回寄存器 |
| STSM 写完 SMEM、发 TMA store 前（`sm100_store_cd.cuh:118`） | `cute::tma_store_fence()`<br>= `fence.proxy.async.shared::cta` | generic proxy（`st.shared` / `stmatrix`）→ async proxy（TMA）的 SMEM 可见性 |

这五条 fence 是 SM100 异步编程模型里最容易遗漏、且遗漏后只在特定时序下出错的部分。

### 10.5 稳态时序（一个输出块的完整生命周期）

以 `kNumStages = 6`、`num_total_k_blocks = 128`（K=8192, BLOCK_K=64）为例，纵轴是时间：

```
TMA warp  │ empty.wait │ TMA(s=0) │ empty.wait │ TMA(s=1) │ … │ TMA(s=5) │ empty.wait(阻塞) │ TMA(s=0) │ …
              │ expect_tx │              │ expect_tx │                                        ▲
              ▼            ▼              ▼            ▼                                        │
full[0] ────●───────────────────────────────────────────────────────────────────┐      │
full[1] ──────────●──────────────────────────────────────────────────────────────┼─┐    │
                 │ wait(phase=0)                                                    │ │    │
MMA warp  │ tmem_empty.wait(穿过) │ full[0].wait │ 4×UMMA │ commit→empty[0] │ full[1].wait │ 4×UMMA │ commit→empty[0]…
                                          │                                                    │
                                          └──── 最后一个 k_block 额外 commit → tmem_full[a] ────┘
                                                                             │
Epilogue                                                                 tmem_full.wait(a)
                                                                             │
                                                ┌────────────────────────────┘
                                                ▼
                          for each (w,s) store stage:
                            tma_store_wait<1> → NamedBarrier.sync
                            → tcgen05.ld ×8 → fence(wait::ld) → st.shared.v4 ×8
                            → 【最后一次】tcgen05.fence::before_thread_sync + tmem_empty.arrive(0u)  ← 早释放
                            → tma_store_fence → NamedBarrier.sync → TMA store + commit_group
```

三个重叠关系：

1. **TMA 超前 MMA 最多 6 个 k_block**（受 `kNumStages` 限）。
2. **MMA 超前 Epilogue 最多 1 个输出块**（受 TMEM 双缓冲限）。块 i 的 epilogue 与块 i+1 的全部 128 个 k_block 重叠。
3. **STSM 超前 TMA store 最多 1 个 store stage**（受 `kNumTMAStoreStages = 2` 与 `wait_group.read 1` 限）。

### 10.6 退出协议

```cpp
// MMA warp（仅 leader CTA），循环结束后：
const auto iter_idx = scheduler.current_iter - 1;
if (kNumMulticast > 1 and iter_idx >= 0) {
    const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
    tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
}

// 所有 warp：
kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();   // 第 452 行
if (warp_idx == 0) Allocator().free(0, kNumTmemCols);
```

源码注释把第 392–397 行称为 *another round of waits*，理由是 *To safely deconstruct barriers*。具体危险：peer CTA 的 epilogue 线程通过 `arrive(0u)` **远程**写 leader CTA SMEM 里的 mbarrier。如果 leader 已经跑完并退出（SMEM 释放），peer 那次远程 arrive 就是一次非法访问。因此 leader 必须等到**最后一次** `tmem_empty` 到达（因为 barrier 按序使用，等最后一个就隐含前面的都已到达）。

注意这里用的是 `wait(accum_phase_idx)` 而不是循环里的 `wait(accum_phase_idx ^ 1)`——相位不取反，因为现在要等的是「该相位**已完成**」（即 epilogue 真的 arrive 了），而不是循环入口那个「上一轮已释放」的语义。

随后的 `cluster_sync_with_relaxed_arrive()`（= `cluster_arrive_relaxed()` + `cluster_wait()`，注释说明比 `cute::cluster_sync` 略快但内存序保证更弱）保证两个 CTA 都到达后才释放 TMEM。第 451 行的 `// TODO: Remove redundant synchronization` 说明作者也意识到这里的同步可能多余。

---

## 11. Epilogue

### 11.1 入口

```cpp
} else if (warp_idx >= kNumNonEpilogueThreads / 32 and
           warp_idx <  (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
    const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);
    uint32_t tma_stage_idx = 0;                       // 跳块共享
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        accum_stage_idx / accum_phase_idx ← current_iter
        tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
        ptx::tcgen05_after_thread_sync();
        tmem_base_addr = accum_stage_idx * UMMA_N;
        base_m_idx = scheduler.get_global_idx<(not is_m_grouped_contiguous(kGemmType)), MN>(shape_m, BLOCK_M, m_block_idx);
        base_n_idx = n_block_idx * BLOCK_N;
        kSwapAB ? sm100_store_cd_swap_ab<...>(...) : sm100_store_cd<...>(...);
    }
}
```

两个细节：

- `base_m_idx` 的 `kWithGroupOffset` 是 `not is_m_grouped_contiguous(...)`——m-grouped-contiguous 的 group 偏移已经由 `make_tma_a_desc` 把 `m * num_groups` 拼进了 gmem 外维，所以**不能**再加；而 masked / psum 变体需要加 `current_group_idx * shape_m`。
- `base_n_idx` 直接用 `n_block_idx * BLOCK_N`，不走 `get_global_idx`。B 的 group 偏移在加载侧已经处理，存储侧的 `tensor_map_cd` 对 m-grouped 把 group 拼在 M 维上（`make_tma_cd_desc(d, m, n, …, num_groups, …)` → `gmem_outer = m * num_groups`）。
- `tma_stage_idx` 声明在块循环外，注释 *Share store pipeline between blocks*：C/D 环也跳块连续，与 A/B 环同理。

### 11.2 非 swap-AB：TMEM → RF → SMEM 的 swizzle 逐行推导

```cpp
constexpr uint32_t kNumBankGroupBytes     = 16;
constexpr uint32_t kNumElemsPerBankGroup  = 16 / sizeof(cd_dtype_t);   // bf16→8, fp32→4
constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;                    // 本 kernel 恒为 1
constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;                // = BLOCK_N / (128/sizeof)

for (w = 0; w < kNumMWaves; ++w)
  for (s = 0; s < kNumStores; ++s, advance_store_pipeline()) {
    smem_base_ptr = smem_cd[tma_stage_idx];
    if (epilogue_warp_idx == 0) cute::tma_store_wait<kNumTMAStoreStages - 1>();   // wait_group.read 1
    NamedBarrier::sync(kNumUMMAStoreThreads, 0);

    for (i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i) {
        auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);   // i + lane*8
        constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;       // 128B swizzle → true
        auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
        auto col = kHasShortcut ? (i)                : (bank_group_index % 8);
        col ^= row % (kSwizzleCDMode / 16);                                              // col ^= row % 8

        uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
        auto smem_ptr = smem_base_ptr + epilogue_warp_idx * 32 * kSwizzleCDMode
                                    + row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
        uint32_t values[kNumElemsPerBankGroup];
        // fp32：SM100_TMEM_LOAD_32dp32b4x → 4 个值 → st.shared.v4.f32
        // bf16：SM100_TMEM_LOAD_32dp32b8x → 8 个值 → cast_into_bf16_and_pack ×4 → st.shared.v4.u32
    }
    …
  }
```

逐步解读：

1. **行划分**：`epilogue_warp_idx * 32 * kSwizzleCDMode` 把 warp `w` 定位到 SMEM 的第 `32w` 行。`kNumUMMAStoreThreads = STORE_BLOCK_M` 个线程恰好覆盖 `STORE_BLOCK_M` 行，**一线程一行**。
2. **TMEM 读取形状**：`32dp32b{4,8}x` = 32 个 datapath（对应 warp 的 32 个 lane）× 32-bit × 4/8 次重复。一个 warp 一条指令读走 `32 行 × {4,8} 列` 的 FP32 累加器，正好是 `kNumElemsPerBankGroup` 列 = 一个 16-B bank group 的宽度。
3. **`i` 循环**：跑 `STORE_BLOCK_N / kNumElemsPerBankGroup` 次（bf16：`64/8 = 8`；fp32：`32/4 = 8`），恰好覆盖 128-B 行内的 8 个 bank group。所以两种 dtype 下都是 **8 次 TMEM load + 8 次 128-bit `st.shared`**。
4. **swizzle 计算**：`kHasShortcut` 分支在 `kSwizzleCDMode == 128` 时成立，`row = i/8 + lane_idx = lane_idx`（因为 `i < 8`），`col = i ^ (lane_idx % 8)`。通用分支先把 `(i, lane)` 线性化成 `bank_group_index` 再除/模 8——两者在 128B swizzle 下等价，前者省掉一次除法。
5. **每次写入量**：一次 `st.shared.v4.u32` = 16 B = 一个 bank group。一个 warp 一轮写完 `32 行 × 128 B = 4 KB`，8 轮写完整个 stage（`STORE_BLOCK_M × 128 B`）。
6. **无 bank conflict**：同一轮里 32 个 lane 写 32 个**不同行**的同一逻辑 bank group，而 `col ^= row % 8` 把它们打散到 8 个不同的物理 bank group。

### 11.3 swap-AB：靠 `stmatrix.trans` 做转置

swap-AB 下 TMEM 里存的是 `D^T`（行 = n，列 = m），而 GMEM 里的 D 是行主序的 `M × N`，必须转置：

```cpp
DG_STATIC_ASSERT(STORE_BLOCK_N == 128, "STORE_BLOCK_N must be 128 to match TMEM rows");
DG_STATIC_ASSERT(kSwizzleCDMode == 128, "TMA D must be 128B swizzled");
constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);   // bf16 → 64
constexpr uint32_t kNumSwizzleAtomRows = 8;

const auto num_stores = effective_m / STORE_BLOCK_M;        // 运行期！effective_m 可达 BLOCK_M/16 = 16 轮
for (s = 0; s < num_stores; ++s, advance_store_pipeline()) {
    for (i = 0; i < STORE_BLOCK_M / kNumSwizzleAtomRows; ++i) {          // 16/8 = 2
        tmem_addr = tmem_base_addr + s * STORE_BLOCK_M + i * kNumSwizzleAtomRows;
        constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;    // 64/32 = 2
        outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
        inner_atom_offset = i * kNumSwizzleAtomRows * kSwizzleCDMode;

        // bf16：两次 16dp256b 加载（第二次 tmem_addr | 0x00100000 即行 +16）→ 8 个值
        //       → cast_into_bf16_and_pack ×4 → SM90_U32x4_STSM_T<int>::copy(...)  ← .trans
        // fp32：一次 32dp32b8x → 8 个值 → 逐行 st.shared.u32，row = lane%8、col = (warp%2)*4 + lane/8
    }
}
```

- **行覆盖**：`16dp256b` 一次只覆盖 16 个 datapath，所以用 `tmem_addr | 0x00100000`（行域 `+16`）发第二次，两次合起来 32 行×8 列 = 32 lane × 8 值。
- **warp 分工**：`kNumWarpsPerAtom = 2`，warp 0/1 写 atom 0（n 方向 0–63）、warp 2/3 写 atom 1（n 方向 64–127），每 atom `STORE_BLOCK_M(16) × 128 B = 2 KB`，四 warp 共 4 KB = 一个 stage。
- **转置本体**：`stmatrix.sync.aligned.x4.m8n8.shared.b16.trans`（`SM90_U32x4_STSM_T<int>`）——四个 lane 组各自提供 4×32-bit（= 8 个 bf16），硬件在写入 SMEM 时完成 8×8 转置。这是避免在寄存器里做显式 shuffle 转置的关键。
- **`num_stores` 是运行期的**：`effective_m = get_aligned_effective_m_in_block(m_block_idx)`，psum layout 的尾块可以小于 `BLOCK_M`，因此循环不能展开——这正是 swap-AB 省算力的地方（§7.4）。
- **TMA store 拆成 `STORE_BLOCK_N / STORE_BLOCK_N_ATOM = 2` 条**，每条 box = `64 × 16`，坐标 `(n_idx = base_n_idx + i*64, m_idx = base_m_idx + s*16)`。

### 11.4 TMEM 早释放（关键优化）

```cpp
// Notify tensor memory empty (only at the leader CTA) arrival ASAP
// NOTES: only the last stage needs to do this
if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
    ptx::tcgen05_before_thread_sync();
    tmem_empty_barrier->arrive(0u);
}

// 之后才是：
cute::tma_store_fence();
NamedBarrier::sync(kNumUMMAStoreThreads, 0);
if (epilogue_warp_idx == 0 and cute::elect_one_sync()) { …TMA store…; cute::tma_store_arrive(); }
```

释放点被刻意放在「所有 `tcgen05.ld` 已完成、但 TMA store 尚未发出」之间。合法性依据：到达此处时，`i` 循环已经把整个 `BLOCK_M × BLOCK_N` 累加器全部读进了寄存器并写入了 SMEM（`fence_view_async_tmem_load()` 保证 `tcgen05.ld` 真的完成），TMEM 不再需要。而后续的 `tma_store_fence` + NamedBarrier + TMA store 只涉 SMEM。

效果：MMA warp 可以**立刻**开始下一个输出块（往另一个 TMEM 缓冲累加），而 epilogue 还在慢慢把数据搬到 GMEM。若把 arrive 放到函数末尾，TMEM 双缓冲的收益会被 TMA store 的延迟吃掉一大半。

### 11.5 C/D 环：不用 mbarrier，用 `wait_group.read`

```cpp
if (epilogue_warp_idx == 0) cute::tma_store_wait<kNumTMAStoreStages - 1>();   // = wait_group.read 1
cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
…写 SMEM…
cute::tma_store_fence();
cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
    SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);      // 或 SM90_TMA_REDUCE_ADD_2D
    cute::tma_store_arrive();                                                  // cp.async.bulk.commit_group
}
__syncwarp();
```

- **两级同步分工**：`NamedBarrier`（`barrier.sync.aligned id, num_threads`，id = 0，只绑 `kNumUMMAStoreThreads` 个线程）把「TMA 读完 SMEM」的信息从 warp 0 广播给全体；写完后再用一次 NamedBarrier 确保所有人的 `st.shared` 都可见，才让 warp 0 的单一 lane 发 TMA。
- **为什么只 warp 0 等**：`cp.async.bulk.wait_group` 是**每线程**的计数，而 `commit_group` 只由 warp 0 的 elected lane 发出，所以只有它持有非零的 group 计数，也只有它能等。
- **`.read` 后缀的语义差别**：cute 的 `tma_store_wait<N>()` 展开为 `cp.async.bulk.wait_group.read N`，它只等到「TMA 引擎不再需要读这块 SMEM」，**不**等数据真正落到 GMEM。这正是覆写 stage 所需的最弱条件，比无 `.read` 的版本（等全局可见）早很多。对比 [ptx/tma.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/ptx/tma.cuh) 里手写的 `ptx::tma_store_wait<N>()`，它特意注释 *this function does not have `.read`*——本 epilogue 用的是 cute 那个带 `.read` 的版本。
- **`kWithAccumulation`**：当调用方传了 `c`（`with_accumulation = c.has_value()`），TMA 指令从 `SM90_TMA_STORE_*` 换成 `SM90_TMA_REDUCE_ADD_*`（`cp.reduce.async.bulk.tensor.…add`），在**写回路径上做 GMEM 原子累加**，实现 `D += A@B` 而不需要先把 C 读进来。Batched 时同理换成 3D 变体。
- **`epilogue_type_t::apply_index_n<STORE_BLOCK_N>(n_idx)`**：本 kernel 固定传 `EpilogueIdentity`（恒等）。另一个可选的 `EpilogueHeadSplits<kLeft,kMid,kRight>` 用于 attention 场景，把 Q/K/V 三段拼在一起的 N 轴索引跳过中间的 head 段，要求三段都能被 `STORE_BLOCK_N` 整除。

### 11.6 dtype 转换

```cpp
// fp32 输出：直接存
cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, v0, v1, v2, v3);
fence_view_async_tmem_load();
ptx::st_shared(smem_ptr, v0, v1, v2, v3);                       // st.shared.v4.f32

// bf16 输出：读 8 个 FP32，两两 pack
DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and is_same_v<cd_dtype_t, bfloat16_t>);
cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, v0..v7);
fence_view_async_tmem_load();
ptx::st_shared(smem_ptr, cast_into_bf16_and_pack(v0,v1), cast_into_bf16_and_pack(v2,v3),
                          cast_into_bf16_and_pack(v4,v5), cast_into_bf16_and_pack(v6,v7));   // st.shared.v4.u32
```

`cast_into_bf16_and_pack` 的实现是 `__float22bfloat162_rn({x, y})` 后重解释为 `int`——**一次指令完成两个 FP32 → BF16 的舍入与拼接**（round-to-nearest-even）。所以 8 个 FP32 累加器 → 4 个 32-bit → 一条 128-bit `st.shared.v4.u32`，寄存器压力与指令数都是最优的。

累加器永远是 FP32（`make_instr_desc<bf16, bf16, float, …>`），BF16 只出现在输入与最终输出，不存在中间累加降精度。

---

## 12. 其他重要机制

### 12.1 PDL：把 prologue 藏进前驱 kernel 的尾巴

Programmatic Dependent Launch 允许后一个 kernel 在前一个 kernel **尚未执行完**时就被派发到 SM 上跑，只要它不碰前驱的输出。DeepGEMM 的用法是把 `cudaGridDependencySynchronize()`（PTX: `griddepcontrol.wait`）**刻意推迟**到 prologue 的最后一步：

```cpp
line 102  kNumMulticast > 1 ? cluster_sync_with_relaxed_arrive() : void();   // 2-CTA TMEM alloc 前对齐
line 110  if (warp_idx == 0) { prefetch_tma_descriptor(&tensor_map_a/b/cd); } // tensormap 进 L2
line 117  shape_* = SHAPE_* != 0 ? SHAPE_* : shape_*;                         // 编译期常量覆写
line 122  extern __shared__ __align__(1024) uint8_t smem_buffer[];            // SMEM 指针算术
line 148  if (warp_idx == 1 && elect_one_sync()) { …32 个 mbarrier init…; fence_barrier_init(); }
line 168  else if (warp_idx == 2) { Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem); }
line 172  cluster_sync / __syncthreads();
line 175  cudaGridDependencySynchronize();     ◄── 真正的依赖点
line 194  …角色分派，第一条 TMA load…
```

被提到依赖点之前的四件事，全都**只读写本 kernel 自己的资源**，与前驱的数据无因果关系：

| prologue 工作 | 为什么可以越过 PDL 边界 |
| --- | --- |
| TMA descriptor prefetch | 三个 `tensor_map_*` 是 `__grid_constant__` 按值传入的 kernel 参数，住在常量内存里，由 host 在 launch 前填好，不是前驱的 GMEM 输出 |
| mbarrier init（`3S + 4` 个） | 纯 SMEM 写，SMEM 随 CTA 分配，天然私有 |
| `tcgen05.alloc` | TMEM 是 SM 上的独立资源池，分配走硬件仲裁，不依赖任何 GMEM 状态 |
| cluster sync / `fence_barrier_init` | 只是 CTA 之间的汇合与可见性，不涉及数据 |

而第一条 `cp.async.bulk.tensor` 要读的 A/B 张量**很可能就是前驱 kernel（如 RMSNorm、量化、all-gather）刚写出来的**，因此必须留在 `cudaGridDependencySynchronize()` 之后。

收益量化：冷启动路径上，32 个 mbarrier 的逐个 `mbarrier.init` + `fence.mbarrier_init.release.cluster`、`tcgen05.alloc.sync.aligned`（2-CTA 时还要等 peer CTA 一起），以及三个 tensormap 的 L2 miss，加起来通常是数百到上千周期。把它们与前驱 kernel 的 drain 阶段重叠，等于把 GEMM 的「首字节延迟」压掉一截——对小 K、块数少的 shape 尤其明显。

三个使用注意点：

1. **默认关闭**。`DeviceRuntime::enable_pdl = false`（[device_runtime.hpp](../third_party/DeepGEMM/csrc/jit/device_runtime.hpp) 第 16 行），需要 `deep_gemm.set_pdl(True)` 才生效。`LaunchArgs` 构造函数的默认值虽然是 `enable_pdl = true`，但 `KernelRuntime::launch()` 会在发射前无条件覆写：

   ```cpp
   // Allow runtime override from Python.
   // NOTES: the default is enabled.
   launch_args.enable_pdl = device_runtime->get_pdl();     // kernel_runtime.hpp:146
   ```

   注释里的「the default is enabled」指的是 `LaunchArgs` 那个默认实参，而**实际生效的是全局开关**，它默认是关的。这处注释与代码的不一致很容易误读。设 `DG_JIT_DEBUG=1` 可以在 launch 日志里直接看到 `pdl: 0/1`。
2. **不开 PDL 也正确**。`enable_pdl == false` 时不加 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`，grid 按流序在前驱完全结束后才启动，此时 `cudaGridDependencySynchronize()` 的依赖已天然满足、立即返回，所以这一行不需要条件编译。
3. 全仓库**没有任何一处调用 `cudaTriggerProgrammaticLaunchCompletion()`**（`griddepcontrol.launch_dependents`）。也就是说 DeepGEMM 只做「等待方」，不做「提前放行方」——它依赖前驱 kernel 自然结束来放行自己，而不主动帮后继 kernel 提前启动。这在 GEMM 之间级联时意味着 PDL 的收益是单向的。

### 12.2 `cluster_sync_with_relaxed_arrive()` 与它的安全性前提

```cpp
// comm/barrier.cuh
CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();   // barrier.cluster.arrive.relaxed
    cute::cluster_wait();             // barrier.cluster.wait
}
```

`cute::cluster_sync()` 展开的 arrive 带默认（release）内存序，会把「arrive 之前本 CTA 的所有写」提升到 cluster 作用域可见。relaxed 版省掉这一层，只做**控制流的汇合**，不承诺任何内存可见性——快一点，但用错就是隐蔽的竞态。

kernel 里三处使用，各自的正确性依据并不相同，值得逐条看清：

| 位置 | 时机 | 为什么 relaxed 够用 |
| --- | --- | --- |
| 第 102 行 | `tcgen05.alloc` **之前** | 2-CTA 的 TMEM 分配是硬件级成对操作，只要求两个 CTA 都到达该点；此处**尚无任何需要跨 CTA 可见的写**（barrier 还没 init），所以连 release 都是多余的 |
| 第 172 行 | barrier init + TMEM alloc **之后** | 这一处的跨 CTA 可见性**是必需的**（peer CTA 会通过 `umma_arrive_multicast` 与 `arrive(0u)` 远程投递到本 CTA 的 mbarrier）。但承担 release 的不是 cluster arrive，而是它上面第 167 行的 `cutlass::arch::fence_barrier_init()`（`fence.mbarrier_init.release.cluster`）——这条 fence 专门为「mbarrier 初始化对 cluster 内 async proxy 可见」设计，比通用 release 更精确也更便宜。cluster arrive 于是退化为纯汇合，可以安全 relaxed |
| 第 452 行 | 所有角色退出之后 | 配合第 392–397 行的「额外一轮 `tmem_empty_barriers` wait」，确保 peer CTA 不会再向本 CTA 发远程 arrive，然后才能 `Allocator().free(0, kNumTmemCols)`。此处同样只需汇合语义 |

`kNumMulticast == 1` 时三处分别退化为 `void()` / `__syncthreads()` / `__syncthreads()`——注意第一处是 `void()` 而非 `__syncthreads()`，因为单 CTA 下 `tcgen05.alloc` 本来就不需要跨 CTA 对齐，而 CTA 内部此时也还没有任何需要全 block 可见的状态。

### 12.3 三层断言防御体系

[common/exception.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/exception.cuh) 定义了三个宏，分工非常清楚：

```cpp
#define DG_STATIC_ASSERT(cond, ...)        static_assert(cond, __VA_ARGS__)         // 编译期，零成本
#define DG_DEVICE_ASSERT(cond)             do { if (not (cond)) { printf(…); asm("trap;"); } } while (0)
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond)   do { if (not (cond)) asm("trap;"); } while (0)
```

`DG_DEVICE_ASSERT` 会生成 `printf` 的完整调用序列（占用寄存器、拉进 vprintf 的常量字符串），在热路径上代价可观；`DG_TRAP_ONLY_DEVICE_ASSERT` 只留一条条件 `trap`，几乎免费但出错时只知道「哪次 launch 挂了」，不知道具体条件。

本 kernel 内三者的分布：

| 断言 | 行 | 类别 | 检查内容 |
| --- | --- | --- | --- |
| `cd_dtype_t` 是 `float` / `bfloat16_t` | 56 | static | 输出 dtype 白名单 |
| `BLOCK_K_ == 64` | 65 | static | 一个 swizzle atom 的 K 字节数必须是 128 B |
| `BLOCK_K % UMMA_K == 0`、`kKAlignment % UMMA_K == 0` | 66–67 | static | K 方向能被 UMMA 原子（16）整除 |
| `kNumMulticast ∈ {1,2}` | 68 | static | cluster 最多 2 CTA |
| `(kSwapAB and BLOCK_N == LAYOUT_AD_M) or …` | 69 | static | swap-AB 时 `BLOCK_N` 必须恰好 128 |
| `kNumUMMAStoreThreads % 32 == 0` | 81 | static | epilogue 必须以整 warp 参与 |
| 三个 SMEM 区都是 1024 B 的倍数 | 88 | static | `__align__(1024)` + swizzle-128B 的前提 |
| `kNumTMAStoreStages >= 1` | 90 | static | C/D 环非空 |
| `UMMA_A_SIZE_PER_STAGE <= SMEM_A + SMEM_B * kNumStages` | 94 | static | UMMA 的越界读不会冲出 A/B 区（见 §5.5） |
| `32 <= kNumTmemCols <= 512` | 99, 145 | static | TMEM 分配的硬件合法区间 |
| `kGemmType` 与 `kMajorA` 的组合合法 | 216 | static | m-grouped 必须 A K-major |
| `kNumStages <= 32` | 264 | static | lane 寄存器存描述符的前提（见 §9.3） |
| UMMA shape 合法（`M∈{64,128,256}` 且 N 的粒度/范围匹配） | 274–277 | static | 替代 CUTLASS MMA traits 的手写校验 |
| `kTensorCoreUtilControl > 0` | 371 | static | 利用率控制的分母非零 |
| `ptx::ld_shared(tmem_ptr_in_smem) == 0` | 405 | **trap-only** | TMEM 分配基址列号必须为 0（禁止 2 CTA 共 SM） |
| `false and "This kernel only support sm_100f"` | 460 | device | 位于 `#else`（非 SM100）分支，SM100 cubin 里不存在 |

一个值得注意的结论：**在真正编出来的 SM100 cubin 里，`DG_DEVICE_ASSERT` 一条都没有**，运行期检查只剩第 405 行那一条 trap。所有配置合法性都在 JIT 编译阶段就被 `static_assert` 拦住了——这也意味着 host 端启发式（§2.2）如果生成了非法配置，失败形式是**编译报错**而不是运行期错误，调试信息直接指向那条 `static_assert` 的字符串。

第 405 行选 trap-only 而非 printf 版，是因为它位于 epilogue warp 的入口、每次 launch 只执行一次，但作者仍希望它不影响这一带的寄存器分配（printf 版本会引入额外的活跃值）。运行期索引合法性检查（依赖运行期 shape 的那些）被下沉到库里，例如 `mma::sm100::make_umma_desc` 的 `DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0)`。

### 12.4 编译期常量折叠的收益链

JIT 全量模板特化不是「代码洁癖」，它触发一串连锁的死代码消除。以典型调用（`compiled_dims = "nk"`）为例：

```
SHAPE_K != 0
  └─► line 119: shape_k 变成编译期常量
        └─► line 313: kMayHaveTailKBlock = (SHAPE_K == 0 or SHAPE_K % BLOCK_K != 0) = constexpr false
              └─► line 344–359: 整个 tail-K 分支不生成 SASS
                    ├─ for_each_static_prefix 的 fold expression 展开消失
                    ├─ 为「≤4 用 switch」准备的跳转表消失
                    └─ issue_tail_k_block lambda 消失
        └─► num_total_k_blocks = ceil_div(shape_k, BLOCK_K) 成为编译期常量
              └─► K 主循环可完全展开，循环边界比较折叠
                    └─► BLOCK_K/UMMA_K = 4 条 UMMA 全部直线排布
                          └─► 每条的 a_desc.lo / b_desc.lo 增量是立即数
                                （SMEM_A_SIZE_PER_STAGE/16、advance_umma_desc_lo 的 offset）
                                └─► 描述符推进折叠成一条 IADD3
```

同时被 `if constexpr` 彻底消除的分支维度：`kMajorA`（2）× `kMajorB`（2）× `kSwapAB`（2）× `kNumMulticast`（2）× `kWithAccumulation`（2）× `kIsBatchedMM`（2）× `kGemmType`（7）× `cd_dtype_t`（2）。**一份 cubin 里只存在一条完全直线化的路径**，没有任何「运行期再决定走哪边」的开销，指令 cache 压力也最小。

反过来说，`SHAPE_M` 默认**不**编译进来是有意的取舍：推理场景下 M（token 数）几乎每次都变，若把 M 也特化，JIT 缓存会爆炸式增长。代价是 M 方向的 tail 处理必须保留为运行期逻辑——`get_aligned_effective_m_in_block()`、`kEnsureZeroPadding`、swap-AB 下的动态 `load_block_m`、epilogue 里运行期的 `num_stores = effective_m / STORE_BLOCK_M`，全都是为这个「M 未知」付出的成本。

JIT 缓存的组织方式也值得一提：每个 (shape, config) 对应一个独立目录，`load_kernel()` 会校验 `cuLibraryGetKernelCount == 1`，否则打印 *Corrupted JIT cache directory … please run `rm -rf <dir>`* 并断言失败（[handle.hpp](../third_party/DeepGEMM/csrc/jit/handle.hpp) 第 144–150 行）。一个目录一个 kernel 的约定让缓存失效的判断非常简单。

### 12.5 `kEnsureZeroPadding` 的真实作用范围

这个模板参数名字很唬人，但它**只在一个地方被读取**——[scheduler/gemm.cuh](../third_party/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh) 第 192 行：

```cpp
CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(const uint32_t& m_block_idx) const {
    constexpr uint32_t UMMA_STEP_N = 16;
    DG_STATIC_ASSERT(BLOCK_M % UMMA_STEP_N == 0, "Invalid alignment");
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout and not kEnsureZeroPadding)
        return math::align(…当前 psum 块的实际剩余行数…, UMMA_STEP_N);
    return BLOCK_M;
}
```

即：只有在 **psum layout 的 m-grouped contiguous** 且显式关掉 zero-padding 时，最后一个 M 块的有效行数才会被收缩到实际值（对齐到 UMMA 的 N 步长 16）。其余所有情形恒返回 `BLOCK_M`——整个块照算，越界行的 A 由 TMA 的 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` 零填充，于是 D 的 padding 行自然是 0。

两条路径的差别在于「padding 区是否被写」：收缩路径下 epilogue 的 `num_stores` 变小，padding 行的 D **完全不写**（保留 GMEM 原值）；零填充路径下整个 `BLOCK_M × BLOCK_N` 都会被 TMA store 覆盖成 0。Python API 默认 `ensure_zero_padding = true`。对 BF16 Normal GEMM 而言这个参数是纯 dead code，读代码时不必为它费神。

---

## 13. 不变式、限制与已知坑

### 13.1 必须成立的不变式

下表是「改动这个 kernel 或其 host 启发式时不能破坏」的硬约束。左列任一被打破，结果要么是编译失败，要么是静默的数值错误。

| 不变式 | 由谁保证 | 破坏后的表现 |
| --- | --- | --- |
| `BLOCK_K_ == 64` | host `block_k = 128 / element_size` | `static_assert`（第 65 行） |
| `kSwizzleMode == BLOCK_ATOM_K × sizeof(dtype)`（K-major） | `get_swizzle_mode()` | `make_umma_desc` 里的 `DG_STATIC_ASSERT("Unexpected value")`；SBO/LBO 全错 |
| `kNumStages <= 32` | host `num_stages = min(…, 32)` | `static_assert`（第 264 行）。真正的原因是 lane 寄存器只有 32 个，用来存 per-stage 描述符低位 |
| 三个 SMEM 区各自 1024 B 对齐且尺寸为 1024 的倍数 | `SMEM_*_SIZE_PER_STAGE` 的构造 | `static_assert`（第 88 行）；swizzle-128B 的 atom 跨边界 → 数据错乱 |
| `2 × UMMA_N <= 512` | host TMEM 容量过滤 | host 直接跳过该候选；若绕过则 `tcgen05.alloc` 失败 |
| `32 <= kNumTmemCols <= 512` | `get_num_aligned_tmem_cols` | `static_assert`（第 99/145 行） |
| UMMA base address 必须是 TMEM 第 0 列 | 1 CTA/SM + `__launch_bounds__(256,1)` | 第 405 行 `trap`。所有 TMEM 地址都省掉了 base 偏移 |
| `UMMA_M == LAYOUT_AD_M × kNumMulticast`，`LAYOUT_AD_M == 128` | TMEM datapath 固定 128 行 | UMMA shape 非法（第 274 行） |
| cluster 只能沿 layout A/D 方向 | host 过滤：`swap_ab && cluster_m > 1` 跳过；`!swap_ab && cluster_n > 1` 跳过 | `cta_group::2` 的 M 维拼接方向错 |
| `kNumSMs % cluster_size == 0` | host 过滤 | cluster 跨 grid 边界，launch 失败 |
| `ceil_div(m, BLOCK_M) % cluster_m == 0`（N 同理） | host 过滤 | cluster 内两 CTA 落到不同边界外，`arrive` 计数永远凑不齐 → 死锁 |
| `kNum1DBlocksPerGroup % kNumMulticast == 0` | host `get_num_1d_blocks_per_group()` | cluster 内两 CTA 分到不同 L2 组，multicast 的收益归零 |
| `128 <= LOAD_BLOCK_M + LOAD_BLOCK_N × kNumStages` | 第 94 行断言的化简形式 | UMMA 读 A 时越过 B 区末尾，读到未初始化 SMEM |
| swap-AB ⇒ `BLOCK_N == 128` 且 `kSwizzleCDMode == 128` | host 过滤 + epilogue 的 `static_assert` | `STORE_BLOCK_N` 与 TMEM 行数不匹配 |
| m-grouped ⇒ A 必须 K-major | host `DG_HOST_ASSERT` + 第 216 行 `static_assert` | `get_global_idx` 的 group 偏移公式失效 |
| k-grouped contiguous ⇒ A/B 都必须 MN-major | host `DG_HOST_ASSERT`（impls 第 266 行） | K 方向的 group cumsum 索引算错 |
| `kNumEpilogueStages == 2` | 硬编码 | 退出协议（§10.6）与 `accum_phase_idx` 的 `& 1` 推导全部失效 |
| 每个 barrier 的 `init` 计数与 arrive 次数严格相等 | §10.3 的逐条推导 | 计数多 → 永久等待（死锁）；计数少 → parity 提前翻转（数据竞争） |

### 13.2 功能与平台限制

- **仅 SM100**。`#if __CUDA_ARCH__ >= 1000` 之外只有 `DG_DEVICE_ASSERT(false and "This kernel only support sm_100f")`。SM90 走的是同仓库另一套 kernel（`sm90_*`），两者的流水组织完全不同（SM90 是 warpgroup MMA + `wgmma.mma_async`，没有 TMEM）。
- **dtype 硬编码 BF16**。`smem_a` / `smem_b` 一律 cast 成 `cutlass::bfloat16_t*`，`make_instr_desc<bfloat16_t, bfloat16_t, float, …>`。FP8/FP4 走 `sm100_fp8_fp4_gemm_1d1d.cuh`（需要 scale factor，流水结构不同）。输出 `cd_dtype_t` 只有 `float` 与 `bfloat16_t` 两种，没有 FP16。
- **cluster 最多 2 CTA**（`kNumMulticast ∈ {1,2}`），不支持 SM100 理论上的更大 cluster 做 multicast。
- **无 split-K**。K 方向由单个 CTA 串行跑完整个 `num_total_k_blocks`，K 很大而 M/N 很小的瘦长 shape 无法靠增加并行度填满 SM。`kWithAccumulation`（`cp.reduce.async.bulk.tensor…add`）提供了「多次调用累加到同一块 D」的能力，算是 host 层面的手工 split-K，但每次调用之间没有 kernel 内同步。
- **persistent 调度是静态的**。`next_block_idx = (++current_iter) * kNumSMs + blockIdx.x`，纯算术映射，没有原子操作也没有工作窃取。好处是零同步开销、每个 CTA 都能独立推算出自己的块序列；代价是**块间代价不均时无法再平衡**——`MGroupedMasked` 下各 group 的 `masked_m` 差异很大时，末 wave 的长尾会直接暴露在关键路径上（host 的 `compare()` 里 `last_wave_util` 那一项就是在缓解它，但只能缓解不能消除）。
- **D 必须 N-major**（`check_major_type_cd`）。想要列主序输出只能在 host 侧转置。
- **`gridDim.x == num_sms`**，且 `deep_gemm.set_num_sms()` 会同时影响 grid 大小与调度器的 `kNumSMs`；如果设置成小于物理 SM 数，会有 SM 闲置；两者必须一致，否则块分配会漏。

### 13.3 已知坑与代码瑕疵

按「踩到的概率 × 排查成本」排序：

1. **`kTensorCoreUtilControl < 100` 会主动降速**（§9.7）。这是给功耗受限集群做可复现对比用的旋钮，通过 `deep_gemm.set_tc_util(n)` 打开，默认 100。跑分或做性能回归前必须确认它是 100——否则 kernel 会在每个 stage 后 `clock64()` 自旋 `kNumDummyCycles`，测出来的 TFLOPS 毫无意义，而且现象很像「kernel 莫名变慢」，很难往这个方向想。

2. **`BLOCK_M ∈ {32, 64}` 时 MMA 仍按 M=128 发射**（§7.6）。`UMMA_M = LAYOUT_AD_M × kNumMulticast` 与 `BLOCK_M` 无关，所以小 `BLOCK_M` 的配置存在**有意的算力浪费**：TMEM 的 128 个 datapath 里只有 `BLOCK_M` 个装着有用数据。换来的是单一代码路径 + 避免小 M 块的 TMA L2 OOB。看 profiler 里「MMA 指令数 / 有效 FLOP」比值异常时，先确认 `BLOCK_M`。

3. **barrier 区有 `kNumStages` 大小的空洞**（§5.2）。`tensor_core_full_barrier` 的索引是 `3S + 4` 而不是 `2S + 4`，中间空出的 S 个 `Barrier` 是 FP8/FP4 1D1D kernel 的「每 stage 三组 barrier」约定（第三组是 with-SF full barriers）的遗留占位。host 侧 `smem_barriers = 32*8*3 + 2*8*2 + 8` 与它精确呼应，所以 SMEM 预留量对 `kNumStages <= 32` 恰好是**紧上界**。如果有人「优化」掉这个空洞但忘了同步改 host 的 `3`，在 `kNumStages` 接近 32 时就会写穿 SMEM。

4. **第 218 行 `uint32_t k_idx = k_block_idx * BLOCK_K;` 是未使用的残留变量**。真正传给 TMA 的是 `k_a_idx` / `k_b_idx`（由 `get_global_idx` 按 major 分别算出）。读代码时容易误以为 `k_idx` 参与了地址计算。

5. **第 451 行 `// TODO: Remove redundant synchronization`**。退出前的那次 cluster sync 作者自己也怀疑多余。但它目前是 `Allocator().free(0, kNumTmemCols)` 的唯一安全网（保证 peer CTA 不再引用本 CTA 的 TMEM / barrier），不要贸然删除。

6. **peer CTA 上有一批死对象**。`block_rank_in_cluster() != 0` 的 CTA 里：warp 1 完全不执行 MMA 分支（但它 prologue 里 init 的那些 `full_barriers` / `empty_barriers` 是**会被 leader CTA 远程 arrive 的**，不是死的）；真正死掉的是 `tmem_full_barriers`——它 `init(1)`，只由本 CTA 的 `umma_arrive` 触发，而 peer CTA 从不发 UMMA。用 nsys / compute-sanitizer 观察 barrier 计数时，不要把这些恒零的对象当成 bug。

7. **host 的 `store_block_n` ≠ kernel 的 `STORE_BLOCK_N`**（§2.3）。host 填 `layout.block_n`，kernel 非 swap 分支算的是 `kSwizzleCDMode / sizeof(cd_dtype_t)`。两者最终一致只是因为 `make_tma_2d_desc()` 里 `smem_inner_dim = swizzle_mode / elem_size` 又把 host 的值覆盖了一遍。对不上号时先想到这层覆盖。

8. **大量 warp 在空转**。256 线程 / 8 warp 里，真正干活的是：TMA 1 lane + MMA 1 lane（`commit` 时整 warp）+ TMEM alloc 时 32 lane（一次性）+ epilogue `kNumUMMAStoreThreads`。warp 3 恒空转；`BLOCK_M = 32` 时 warp 5–7 也空转。极端配置下 256 个线程里稳态活跃的只有约 34 个。这不是 bug（warp specialization 的固有形态），但会严重误导「占用率」类的性能分析——具体该看哪些指标、为何 `sm__warps_active` 在此无意义，见 **§3.4.6**。

9. **`__shfl_sync(0xffffffff, …)` 要求整 warp 收敛**。第 324 行的 `elect_one_sync()` 与 `__shfl_sync` 都在 `full_barriers[...]->wait()` 之后、warp-uniform 的分支内部，所以合法。若将来有人在 MMA warp 里引入依赖 lane 的分支（例如按 `lane_idx` 走不同路径），这两处会立刻变成未定义行为。

10. **`tcgen05.commit` 是 warp 级收敛指令**（§10.3）。`umma_arrive` 在 `elect` 块**之外**被整个 warp 1 调用，而 `empty_barriers[i]->init(1)`——这只有在「整 warp 执行 commit 只产生一次 arrival」的前提下才成立。任何试图把 `umma_arrive` 挪进 `elect_one_sync()` 块、或把 `init(1)` 改成 `init(32)` 的「修正」都会破坏协议。

---

## 14. 小结

这份 kernel 的核心思想可以压缩成三句话：

1. **把同步全部外化成 mbarrier 的 parity 相位**。没有 `__syncthreads()` 出现在稳态路径上，五类 barrier（`full` / `empty` / `tmem_full` / `tmem_empty` / `tensor_core_full`）各自表达一条单向的生产-消费边，每个角色只等自己需要的那一条，因此三级流水（SMEM ring → TMEM 双缓冲 → C/D SMEM ring）能各自独立超前。
2. **把能变成编译期常量的东西全部变成编译期常量**。JIT 全量特化换来的是内层 K 循环完全展开、描述符增量成即数、tail-K 整段消失、所有 `if constexpr` 分支塌缩成一条直线路径——这才是 DeepGEMM 相对通用库的真正护城河，比任何单个 PTX 技巧都重要。
3. **让专用硬件做它最擅长的事**。TMA 负责所有 GMEM↔SMEM 搬运（含 swizzle、越界零填充、multicast、reduce-add），`tcgen05.mma` 负责所有乘加（操作数直接来自 SMEM，累加器直接在 TMEM），`tcgen05.ld` / `stmatrix` 负责 TMEM→RF→SMEM，通用寄存器与 `__syncthreads()` 只承担控制流。整个数据通路上，通用 SIMT 部分几乎不碰数据。

理解这三点之后，其余所有细节——为什么 `UMMA_M` 恒为 128、为什么用 lane 寄存器存描述符、为什么 TMEM 要早释放、为什么 barrier 区有个空洞、为什么 `cudaGridDependencySynchronize()` 放在第 175 行——都是它们的自然推论。

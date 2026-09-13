# InfiniTrain

[![CI](https://github.com/InfiniTensor/InfiniTrain/actions/workflows/format-check.yaml/badge.svg)](
https://github.com/InfiniTensor/InfiniTrain/actions
)
[![Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniTrain)](
https://github.com/InfiniTensor/InfiniTrain/issues
)
[![PR](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniTrain)](
https://github.com/InfiniTensor/InfiniTrain/pulls
)
[![License](https://img.shields.io/github/license/InfiniTensor/InfiniTrain)](
https://github.com/InfiniTensor/InfiniTrain/blob/master/LICENSE
)

A from-scratch C++ training framework for large-scale models with multi-dimensional distributed parallelism.

## 🚀 Quick Start

### System Requirements

#### Hardware Requirements

- **Recommended**: NVIDIA Ampere-class GPUs (A100/A800) or newer

#### Software Requirements

- **CUDA / NCCL**: Latest stable versions
- **gcc / g++**: Version **13+**
- **CMake**: Version **3.13+**

### Installation

```bash
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON
make -j
```

Build Options:

- `USE_CUDA=ON`

  Enable CUDA backend support.

- `USE_NCCL=ON`

  Enable NCCL-based distributed communication.

> Both options are optional and can be disabled for CPU-only builds.

## ✨ InfiniTrain Overview

### ✔ Support Matrix

| Category                  | Feature                         | Description                                          | Status         |
| ------------------------- | ------------------------------- | ---------------------------------------------------- | -------------- |
| Model Support             | GPT-2                           | Decoder-only Transformer language model              | ✔ Supported    |
|                           | LLaMA 3                         | Modern LLaMA-family Transformer architecture         | ✔ Supported    |
|                           | Qwen3-8B                        | Qwen3 8B language model                              | 🗓 Planned     |
|                           | DeepSeek-V3                     | Large-scale MoE-based language model                 | 🗓 Planned     |
| Precision                 | Multiple Data Type              | FP32, BF16                                           | ✔ Supported    |
|                           | Mixed Precision                 | Autocast-based BF16 compute with FP32 accumulation   | ✔ Supported    |
| Distributed Training      | Data Parallel (DP)              | Parameter-server-style data parallelism              | ✔ Supported    |
|                           | Distributed Data Parallel (DDP) | Collective-based data parallelism                    | ✔ Supported    |
|                           | Tensor Parallelism (TP)         | Intra-layer tensor sharding                          | ✔ Supported    |
|                           | Sequence Parallelism (SP)       | Sequence dimension sharding                          | ✔ Supported    |
|                           | Pipeline Parallelism (PP)       | GPipe, 1F1B scheduling, Virtual Pipeline (vPP)       | ✔ Supported    |
|                           | Hybrid Parallelism              | Arbitrary combination of DDP + TP + SP + PP          | ✔ Supported    |
| Core Components           | Multi-backend                   | CPU and CUDA execution backends                      | ✔ Supported    |
|                           | Multi-node Distributed Training | Distributed execution across multiple nodes          | ✔ Supported    |
|                           | Transformer Abstraction         | Generic Transformer structure abstraction            | ✔ Supported    |
|                           | Backend Registries              | Device / CCL / dtype abstraction and registration    | ✔ Supported    |
|                           | Kernel Dispatcher               | Kernel registration and dynamic dispatch mechanism   | ✔ Supported    |
|                           | Autograd                        | Automatic differentiation engine                     | ✔ Supported    |
|                           | Autocast                        | Automatic mixed precision runtime                    | ✔ Supported    |
|                           | Checkpointing                   | Training checkpoint save and restore                 | 🗓 Planned     |
| Fine-tuning               | LoRA                            | Memory-efficient fine-tuning with merge / unmerge    | ✔ Supported    |
| Memory Optimizations      | ZeRO Stage-1                    | Sharded optimizer states for DDP                     | ✔ Supported    |
|                           | ZeRO Stage-2                    | Sharded gradients across DDP ranks                   | ✔ Supported    |
|                           | Activation Recomputation        | Recompute activations to reduce memory usage         | 🗓 Planned     |
| Performance Optimizations | Compute–Comm Overlap            | Explicit scheduling to hide communication latency    | ✔ Supported    |
|                           | DDP Gradient Bucketing          | Deferred and bucketed gradient synchronization       | ✔ Supported    |
| Execution Mode            | Training Mode                   | Full forward–backward training with autograd         | ✔ Supported    |
|                           | `no_grad` Inference             | Forward-only execution without gradient tracking     | ✔ Supported    |
| Debugging & Tooling       | Built-in Profiler               | Kernel-level performance profiling                   | ✔ Supported    |
|                           | Precision Alignment Checker     | Function / Module precision checks and E2E loss diff | ✔ Supported    |
|                           | CTest + GTest Infrastructure    | Automated unit tests with CTest integration          | ✔ Supported    |
|                           | Automated Benchmarking          | One-click execution, log analysis and Feishu export  | ✔ Supported    |

## 🏋️ Training

Each model in the `example/` directory is compiled into an independent executable.  
For example, the `llama3` example produces a binary named `llama3`.

To view available runtime options:

```bash
./build/llama3 --help
```

### Getting Started

#### Prepare Datasets and Weights

Run the asset preparation script from the repository root. Prepared files are
written to `data/` by default.

```bash
# MNIST dataset
./scripts/assets/prepare-infinitrain-assets.sh mnist

# GPT-2 124M weights, tokenizer, and tokenized TinyShakespeare data
./scripts/assets/prepare-infinitrain-assets.sh gpt2

# LLaMA 3.2 1B weights and tokenized TinyShakespeare data
HF_TOKEN=hf_xxx ./scripts/assets/prepare-infinitrain-assets.sh llama3

# Same flow through ModelScope
MODEL_SOURCE=modelscope MODEL_REPO_ID=LLM-Research/Meta-Llama-3.2-1B \
  ./scripts/assets/prepare-infinitrain-assets.sh llama3
```

Preparing LLaMA requires access to the gated
`meta-llama/Llama-3.2-1B` repository. Accept its license on Hugging Face and
provide `HF_TOKEN`, or authenticate with `hf auth login`, before running the
command. If the Hugging Face download is blocked, set `MODEL_SOURCE=modelscope`
and optionally override `MODEL_REPO_ID` to the mirror you have access to.
The complete LLaMA preparation requires approximately 8.5 GB of free disk
space, including the downloaded checkpoint and converted FP32 weights.

Use `DATA_DIR` to write the assets elsewhere, or prepare all supported assets
in one invocation:

```bash
DATA_DIR=/path/to/data \
HF_TOKEN=hf_xxx \
./scripts/assets/prepare-infinitrain-assets.sh all
```

#### Model Examples

The generated files can be passed directly to the corresponding executables:

##### MNIST

```bash
./build/mnist \
  --device cpu \
  --dataset data/mnist
```

##### GPT-2 124M

```bash
./build/gpt2 \
  --device cuda \
  --input_bin data/gpt2/tiny_shakespeare_train.bin \
  --input_val_bin data/gpt2/tiny_shakespeare_val.bin \
  --tokenizer_bin data/gpt2/gpt2_tokenizer.bin \
  --llmc_filepath data/gpt2/gpt2_124M.bin \
  --num_iteration 10
```

##### LLaMA 3.2 1B

```bash
./build/llama3 \
  --device cuda \
  --input_bin data/llama3/tiny_shakespeare_train.bin \
  --input_val_bin data/llama3/tiny_shakespeare_val.bin \
  --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
  --num_iteration 10
```

### Launch Modes

GPT-2 and LLaMA training support both thread-based and process-based launches.
The examples below use LLaMA, but the same launch modes also apply to GPT-2.

#### Direct Launch

Running a model executable directly uses one process and one device by default.
Set `--nthread_per_process` to use multiple execution threads and devices in the
same process:

```bash
./build/llama3 \
  --device cuda \
  --input_bin data/llama3/tiny_shakespeare_train.bin \
  --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
  --nthread_per_process 8 \
  --num_iteration 10
```

#### Single-node Multi-process Launch

Use `infini_run` to start multiple training processes on one node. Each process
uses one execution thread by default:

```bash
./build/infini_run \
  --nnodes=1 \
  --nproc_per_node=8 \
  ./build/llama3 \
    --device cuda \
    --input_bin data/llama3/tiny_shakespeare_train.bin \
    --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
    --num_iteration 10
```

#### Multi-node Multi-process Launch

Run the following command on every node with the same rendezvous settings and
a distinct `node_rank`:

```bash
./build/infini_run \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=[rank_id] \
  --rdzv_endpoint=[master_addr]:29500 \
  --rdzv_id=[job_id] \
  ./build/llama3 \
    --device cuda \
    --input_bin data/llama3/tiny_shakespeare_train.bin \
    --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
    --num_iteration 10 \
    --tensor_parallel 2 \
    --pipeline_parallel 2 \
    --sequence_parallel
```

`--nproc_per_node` and `--nthread_per_process` can be combined. The total
training world size is:

```text
world_size = nnodes × nproc_per_node × nthread_per_process
```

### Parallelism Strategies

#### Distributed Data Parallelism (DDP)

For a direct launch with TP and PP disabled, the following starts eight
data-parallel workers in one process:

```bash
--nthread_per_process 8  # 8-way DDP when TP=1 and PP=1
```

For all launch modes, the data-parallel size is derived from the total world
size after accounting for tensor and pipeline parallelism:

```text
data_parallel_size = world_size / (tensor_parallel × pipeline_parallel)
```

#### Tensor Parallelism (TP)

```bash
--tensor_parallel 4        # 4-way tensor parallelism
--sequence_parallel        # Enable sequence parallelism (requires TP > 1)
```

#### Pipeline Parallelism (PP)

```bash
--pipeline_parallel 8     		# 8 pipeline stages
--virtual_pipeline_parallel 4  	# Virtual pipeline for better load balancing
```

#### Combining Parallelism Strategies

Multiple parallelism strategies (DDP, TP, SP, PP) can be freely combined to scale training across devices and nodes.

## 🔍 Profiling with nsys

InfiniTrain emits NVTX ranges for every training step and every major phase
(`Forward`, `Backward`, `Optimizer`, `LossReadback`, ...) when built with
`-DNVTX_MODE=ON`. Combine this with `nsys` to locate GPU bottlenecks without
adding any CUDA synchronization.

### Build with NVTX

```bash
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DNVTX_MODE=ON
make -j llama3
```

### Profile LLaMA 3 for 5 iterations

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=nsys/out/llama3_5iter_nvtx \
  --export=sqlite \
  ./build/llama3 \
    --device cuda \
    --dtype bfloat16 \
    --input_bin data/llama3/tiny_shakespeare_train.bin \
    --input_val_bin data/llama3/tiny_shakespeare_val.bin \
    --llmc_filepath data/llama3/llama3.2_1B_fp32.bin \
    --num_iteration 5 \
    --batch_size 4 \
    --sequence_length 64 \
    --total_batch_size 256
```

Flags worth noting:

- `--trace=cuda,nvtx,osrt,cudnn,cublas` collects CUDA runtime/driver, NVTX
  ranges, OS runtime, cuDNN and cuBLAS events. Add `nccl` for multi-GPU runs.
- `--sample=none --cpuctxsw=none` disables CPU sampling and context-switch
  tracing to keep the capture small and the overhead low.
- `--export=sqlite` produces a `.sqlite` file alongside the `.nsys-rep` so the
  results can be queried with SQL or Python.
- `--force-overwrite=true` lets repeated runs reuse the same output name.

### Generate summary reports

```bash
nsys stats \
  --force-export=true \
  --report cuda_gpu_kern_sum \
  --report cuda_gpu_sum \
  --report cuda_api_sum \
  --report nvtx_sum \
  --report nvtx_pushpop_sum \
  --report cuda_gpu_mem_time_sum \
  --report cuda_gpu_mem_size_sum \
  --format table \
  --output nsys/out/llama3_5iter_stats \
  nsys/out/llama3_5iter_nvtx.nsys-rep
```

This writes one `.txt` per report next to the capture. Open the `.nsys-rep` in
the Nsight Systems GUI for the full timeline.

### Per-step Python breakdown

`nsys/out/analyze_steps.py` queries the exported sqlite and prints per-step
wall time, GPU busy ratio, the hot kernels inside the first steady-state step,
and the memcpy breakdown:

```bash
python3 nsys/out/analyze_steps.py
```

### Reference result (A100-SXM4-40GB, LLaMA 3.2 1B, BF16 autocast, 5 iters)

The run above uses `--dtype bfloat16`, which turns on **autocast** mixed
precision: `Matmul`/`Linear` run on BF16 Tensor Cores while the master weights
and the Adam optimizer stay in FP32 (see `infini_train/include/autocast.h`).

| step   | wall (ms) | Σ kernel (ms) | kernels | GPU util |
| ------ | --------- | ------------- | ------- | -------- |
| Step_0 | 273.031   | 54.854        | 3483    | 21.14%   |
| Step_1 | 118.333   | 83.362        | 3569    | 73.13%   |
| Step_2 | 114.822   | 76.690        | 3574    | 69.33%   |
| Step_3 | 112.973   | 73.618        | 3574    | 67.52%   |
| Step_4 | 112.404   | 73.628        | 3574    | 68.05%   |

Step_0 is warm-up (CUDA context init, cuBLAS BF16 kernel selection, first-time
cast-buffer growth); it is host-bound, so GPU util is only ~21%. From Step_1
onward the GPU is busy ~68–73% of the wall time — noticeably lower than FP32's
~93%, because BF16 compute is much lighter and the step becomes
**launch/cast-bound** on a single CUDA stream (no kernel overlap).

Top steady-state (Step_1) kernels:

| rank | kernel                                            | calls | Σ (ms)    | share     |
| ---- | ------------------------------------------------- | ----- | --------- | --------- |
| 1    | `AdamAccumulateGradKernel<float>`                 | 110   | 31.7      | 38.1%     |
| 2    | `CastKernel<__nv_bfloat16, float>` (f32→bf16)     | 356   | 9.5       | 11.4%     |
| 3    | `ampere_s16816gemm_bf16_128x128_..._nt`           | 49    | 4.6       | 5.5%      |
| 4    | `ampere_bf16_s16816gemm_bf16_128x256_..._f2f_tn`  | 49    | 4.5       | 5.4%      |
| 5    | `BinaryBackwardKernel<float>` (Mul)               | 194   | 3.7       | 4.4%      |
| 6    | `FillKernel<float>`                               | 678   | 3.2       | 3.8%      |
| 7    | `BinaryForwardKernel<float>` (Mul)                | 194   | 2.9       | 3.5%      |
| 8    | `TransposeForwardKernel<float>`                   | 128   | 2.5       | 3.0%      |
| –    | **all GEMM kernels combined (BF16 Tensor Core)**  | 339   | **16.8**  | **20.0%** |
| –    | **all Cast kernels combined (autocast)**          | 661   | **10.7**  | **12.8%** |

Sub-phase breakdown of Step_1 (118.3 ms wall; Σ kernel = GPU time executing
inside each NVTX window):

| phase                     | wall (ms) | Σ kernel (ms) | util  |
| ------------------------- | --------- | ------------- | ----- |
| `Forward`                 | 61.9      | 53.1          | 85.7% |
| └ `CrossEntropyForward`   | 6.5       | 5.4           | 82.6% |
| `Backward`                | 53.6      | 28.2          | 52.7% |
| `LossReadback`            | 0.8       | 0.0           | –     |
| `Optimizer`               | 1.1       | 1.3           | –     |

> The `Forward` window shows 53.1 ms of GPU execution but only 23.4 ms was
> actually launched in-phase — the ~30 ms gap is Step_0's Adam optimizer
> draining on the GPU at the start of Step_1. `nsys/out/llama3_kernel_timeline.md`
> uses host-launch attribution (via `correlationId`) to correct for this and
> gives the per-phase totals: Forward 1343 kernels / 23.4 ms, Backward 2116 /
> 28.2 ms, Optimizer 115 / 32.3 ms.

FP32 → BF16 comparison (same 5-iteration workload, Step_1 steady state):

| metric                       | FP32          | BF16 (autocast) |
| ---------------------------- | ------------- | --------------- |
| Step_1 wall                  | 181.8 ms      | 118.3 ms (1.54×)|
| steady-state throughput      | ~1 424 tok/s  | ~2 260 tok/s    |
| GPU util (steady)            | ~93%          | ~68–73%         |
| all GEMM (339 calls)         | 113.9 ms (68.5%) | 16.8 ms (20.0%) — **6.8× faster** |
| fp32 `sgemm` calls           | 339           | ≈0 (all Tensor Core) |
| autocast Cast                | —             | 661 calls / 10.7 ms |
| Adam optimizer               | 31.9 ms (19.2%) | 32.3 ms (38.5%) |
| `CrossEntropyForward` phase  | 32.5 ms       | 6.5 ms          |
| `LossReadback` phase         | 29.3 ms       | 0.8 ms          |

Key observations:

- **BF16 Tensor Core GEMM (6.8× faster).** All 339 GEMMs run on BF16
  `s16816`/`s161616` tensor cores (fp32 `sgemm` ≈ 0), cutting GEMM time from
  113.9 ms to 16.8 ms. The `lm_head` GEMM (vocab = 128 256) alone drops from
  ~7.3 ms to ~0.70 ms — which is why `CrossEntropyForward` shrinks 32.5 → 6.5 ms
  and `LossReadback` (the sync that waits on the backward tail) 29.3 → 0.8 ms.
- **Autocast Cast is the main new cost (12.8%).** 661 `CastKernel` launches /
  10.7 ms per step. The FP32→BF16 **weight** casts dominate: each MLP weight
  (2048×8192) cast costs ~100–102 μs, *more* than the BF16 GEMM it feeds
  (~73–89 μs). Caching the BF16 weight copy across fwd+bwd within a step would
  recover most of this.
- **Adam optimizer is now the #1 cost.** `AdamAccumulateGradKernel<float>`
  (110 calls, 31.7 ms, 38.1%) is untouched by autocast (master weights /
  optimizer stay FP32), so with GEMM 6.8× faster it becomes the largest single
  contributor.
- **Launch/cast-bound.** GPU util fell 93% → ~68–73%, and `cudaLaunchKernel`
  rose to ~3 526 per step (from 2 929) due to the extra casts (~27.9 ms
  host-side). CUDA Graphs, or fusing Cast into the GEMM prologue, are the
  natural next step.
- **`FillKernel`** is still launched ~678–710 times per step (~3.2 ms);
  merging these into fewer larger fills would cut launch overhead.
- **Model load** (before Step_0) still transfers 5.99 GB HtoD (~0.82 s): the
  checkpoint is FP32 and master weights stay FP32, so BF16 does not change this
  one-shot cost.

Steady-state throughput: **~2 260 tok/s** (256 tokens / ~113 ms) on a single
A100-SXM4-40GB at BF16 autocast — about **1.6×** the FP32 ~1 424 tok/s.

For the full per-kernel timeline (methodology, layer-0 launch sequence,
kernel→architecture mapping, and the FP32→BF16 breakdown) see
[`nsys/out/llama3_kernel_timeline.md`](nsys/out/llama3_kernel_timeline.md).

## 🗺 Roadmap

- **2025/03/10** — InfiniTrain **v0.1.0**

  Initial framework prototype with MNIST CPU training.

- **2025/04/30** — InfiniTrain **v0.3.0**

  Added Autograd support and GPT-2 training on CPU/CUDA.

- **2025/07/09** — InfiniTrain **v0.4.0**

  Introduced kernel registration, LLaMA training on CPU/CUDA, BF16 precision, and Data Parallelism.

- **2025/12/31** — InfiniTrain **v0.5.0**

  Added Autocast, multi-dimensional distributed parallelism
   (DDP, TP, SP, PP with GPipe / 1F1B / vPP),
   multi-node training, `no_grad` mode,
   and communication–computation overlap with bucketed gradient synchronization.

- **2026/06/08** — InfiniTrain **v0.6.0**

  Added loss alignment tooling for Function / Module level precision checks
   and end-to-end loss comparison, with a unified hook mechanism.

  Added memory optimizations for DDP training and Autograd execution.
   ZeRO Stage-1 shards optimizer states across DDP ranks, while ZeRO Stage-2
   further shards gradients. Autograd Tensor release timing was also optimized
   to reduce peak memory usage.

  Introduced LoRA fine-tuning with `merge` / `unmerge` support for efficient
   training and inference-time weight merging.

  Refactored core backend abstractions around device, communication, and
   low-precision dtype registration. The framework layer now uses
   `DeviceGuard`, `CclGroupGuard`, and backend-registered FP16 / BF16 native
   types to avoid hardware-specialized framework code.

  Introduced a generic Transformer structure abstraction backed by
   `TransformerConfig`, providing a common foundation for GPT-2 and LLaMA 3
   style model construction.

  Improved BF16 training performance through autocast and elementwise kernel
   optimizations.

  Integrated a CTest + GTest based testing infrastructure to strengthen the
   framework's automated test workflow.

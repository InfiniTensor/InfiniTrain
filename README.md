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
./llama3 --help
```

### Getting Started

The following examples demonstrate **LLaMA 3 supervised fine-tuning (SFT)** using InfiniTrain.

#### Single-node Training Example

```bash
./llama3 \
  --device cuda \
  --input_bin [training_data_path] \
  --llmc_filepath [model_path] \
  --num_iteration 10

```

#### Multi-nodes Training Example (3D parallel)

```bash
./infini_run \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=[rank_id] \
  -- ./llama3 \
     --device cuda \
     --input_bin [training_data_path] \
     --llmc_filepath [model_path] \
     --num_iteration 10 \
     --nthread_per_process 8 \
     --batch_size 40 \
     --total_batch_size 10240 \
     --tensor_parallel 2 \
     --pipeline_parallel 2 \
     --sequence_parallel
```

### Parallelism Strategies

#### Distributed Data Parallelism (DDP)

```bash
--nthread_per_process 8 	# ddp_size = nthread_per_process / (tensor_parallel × pipeline_parallel)
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
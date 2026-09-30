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
```

Preparing LLaMA requires access to the gated
`meta-llama/Llama-3.2-1B` repository. Accept its license on Hugging Face and
provide `HF_TOKEN`, or authenticate with `hf auth login`, before running the
command. The complete LLaMA preparation requires approximately 8.5 GB of free
disk space, including the downloaded checkpoint and converted FP32 weights.

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

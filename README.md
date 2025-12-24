# InfiniTrain

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

### ✔ Model Support

- GPT-2
- LLaMA 3

### ✔ Precision Support

- Full-precision training
  - FP32
  - BF16
- Mixed-precision training
  - BF16 computation with FP32 accumulation
  - Autocast-based precision control

### ✔ Distributed Training

InfiniTrain provides a flexible distributed execution model with multiple parallelism strategies.

- Data Parallelism
  - Parameter-server-style Data Parallel (DP)
  - Collective-based Data Parallel (DDP)
- Model Parallelism
  - Tensor Parallelism (TP)
  - Sequence Parallelism (SP)
  - Pipeline Parallelism (PP)
    - GPipe scheduling
    - 1F1B scheduling
    - Virtual Pipeline Parallelism (vPP)
- Arbitrary combination of **DDP + TP + SP + PP**

### ✔ Core Components

- Multi-backend support
  - CPU
  - CUDA
- Multi-node distributed training
- Kernel registration and dispatch mechanism
- Automatic differentiation (Autograd)
- Automatic mixed precision (Autocast)

### ✔ Performance Optimizations

- Computation-communication overlap
  - Explicit scheduling to hide communication latency
- DDP gradient bucketing
  - Gradient synchronization is deferred and bucketed
  - Reducer-controlled bucket scheduling for overlap optimization

### ✔ Training & Execution Modes

- Standard training mode
- `no_grad` inference mode
  - Forward-only execution without gradient tracking

### ✔ Debugging & Tooling

- Built-in profiler
  - Native framework profiler for kernel-level time analysis
- Automated performance testing
  - One-click execution of all benchmark cases
  - Automatic log analysis
  - Performance metrics aggregated and reported to Feishu spreadsheets

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
--sequence_parallel        # Enable sequence parallelism (recommended with TP)
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

- **2025/12/xx** — InfiniTrain **v0.5.0**

  Added Autocast, multi-dimensional distributed parallelism
   (DDP, TP, SP, PP with GPipe / 1F1B / vPP),
   multi-node training, `no_grad` mode,
   and communication–computation overlap with bucketed gradient synchronization.
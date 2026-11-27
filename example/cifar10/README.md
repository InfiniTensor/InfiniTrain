# InfiniTrain CIFAR-10 Vision Training (`ResNet-18` & `MobileNetV2`)

End-to-end small vision model training support for InfiniTrain on CIFAR-10, covering CPU, single-GPU CUDA, optional cuDNN acceleration, and multi-GPU DistributedDataParallel (DDP) training.

## Implemented Framework Components

- **Vision Operators (CPU + CUDA, Forward + Backward, FP32)**
  - `Conv2d` (`stride`, `padding`, `groups` including depthwise `groups == in_channels`, optional `bias`)
    - Zero-dependency custom CUDA backend (`USE_CUDNN=OFF`): specialized direct depthwise convolution kernel, zero-`im2col` 1×1 strided-batched GEMM, chunked `im2col + cuBLAS SGEMM`, deterministic gather `col2im`, 32-bit indexing, and per-thread/per-device reusable workspace.
    - Optional cuDNN backend (`USE_CUDNN=ON`, `-DCUDNN_ROOT=/path/to/cudnn`): per-device handle/descriptor cache, strict FP32 (`CUDNN_FMA_MATH`), and reusable workspace.
  - `BatchNorm2d` (separate `Train()` / `Eval()` behavior, unbiased `running_var` update, FP64 statistics accumulation on CPU & CUDA)
  - `ReLU`, `ReLU6`
  - `MaxPool2d` (with argmax index saving), `AdaptiveAvgPool2d` (general + fast `1x1` global average pooling)
  - `nn::Flatten`
- **Module & Training Infrastructure**
  - Recursive `Module::Train(bool)` / `Module::Eval()` / `is_training()`, plus buffer accessors (`has_buffer`, `mutable_buffer`, `buffer`).
  - `nn::init::KaimingNormal` initialization.
  - `optimizers::SGD` with PyTorch-exact `momentum`, `weight_decay`, fused CPU/CUDA `SGDAccumulateGrad` kernel, and full `StateDict()` / `LoadStateDict()` (returning empty state when `momentum == 0` for backward compatibility).
  - Portable full training checkpoints (model weights + BN running buffers + SGD momentum buffers + epoch/step/LR metadata).
  - Deterministic non-overlapping DDP dataset sharding with per-rank coverage verification.

## Build Instructions

### Standard Zero-Dependency CUDA + NCCL Build

```bash
# Set CMAKE_CUDA_ARCHITECTURES for your target GPU:
#   NVIDIA A40: 86 | RTX 4090: 89 | H100: 90 | RTX 5090 (Blackwell): 120
cmake -S . -B build \
  -DUSE_CUDA=ON \
  -DUSE_NCCL=ON \
  -DUSE_CUDNN=OFF \
  -DBUILD_TEST=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc) --target cifar10 test_autograd_cpu test_autograd_cuda
```

### Optional cuDNN-Accelerated Build

```bash
cmake -S . -B build_cudnn \
  -DUSE_CUDA=ON \
  -DUSE_NCCL=ON \
  -DUSE_CUDNN=ON \
  -DCUDNN_ROOT=/path/to/cudnn \
  -DBUILD_TEST=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build_cudnn -j$(nproc) --target cifar10 test_autograd_cuda
```

### CPU-Only Build

```bash
cmake -S . -B build_cpu \
  -DUSE_CUDA=OFF \
  -DUSE_NCCL=OFF \
  -DBUILD_TEST=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu -j$(nproc)
ctest --test-dir build_cpu --output-on-failure
```

## Dataset Preparation

Download and extract the official CIFAR-10 binary distribution (`cifar-10-binary.tar.gz`) so that `cifar-10-batches-bin/data_batch_{1..5}.bin` and `test_batch.bin` reside under `<DATASET_DIR>`:

```bash
mkdir -p data/cifar10 && cd data/cifar10
curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
```

## Training Examples

### Single-GPU Training (20 Epochs, Baseline Augmentation)

```bash
# ResNet-18 (CIFAR-10 3x3 stride-1 stem)
./build/cifar10 \
  --dataset=data/cifar10 \
  --model=resnet18 \
  --device=cuda \
  --bs=128 --epochs=20 \
  --lr=0.1 --momentum=0.9 --weight_decay=5e-4 \
  --augment=true \
  --save_ckpt=ckpts/resnet18_e20.bin

# MobileNetV2 (width_mult=1.0, CIFAR-10 adapted strides)
./build/cifar10 \
  --dataset=data/cifar10 \
  --model=mobilenetv2 \
  --device=cuda \
  --bs=128 --epochs=20 \
  --lr=0.1 --momentum=0.9 --weight_decay=5e-4 \
  --augment=true \
  --save_ckpt=ckpts/mobilenetv2_e20.bin
```

### 2-GPU DDP Training

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/cifar10 \
  --dataset=data/cifar10 \
  --model=resnet18 \
  --device=cuda \
  --nthread_per_process=2 \
  --bs=128 --epochs=20 \
  --lr=0.1 --momentum=0.9 --weight_decay=5e-4 \
  --augment=true \
  --save_ckpt=ckpts/resnet18_ddp2_e20.bin
```

### Checkpoint Resume & Single-CPU Smoke Run

```bash
# Resume training from saved checkpoint
./build/cifar10 --dataset=data/cifar10 --model=resnet18 --device=cuda \
  --load_ckpt=ckpts/resnet18_e20.bin --epochs=25

# CPU smoke test
./build/cifar10 --dataset=data/cifar10 --model=resnet18 --device=cpu --bs=16 --epochs=1
```

## Numerical Alignment & Throughput Verification

```bash
# Run operator unit tests (CPU + CUDA)
./build/tests/autograd/test_autograd_cpu --gtest_filter='*AutogradCVTest*'
./build/tests/autograd/test_autograd_cuda --gtest_filter='*AutogradCVTest*'

# Run end-to-end PyTorch alignment, checkpoint resume, 2-GPU DDP grad check, and throughput benchmark
python3 example/cifar10/verify_pytorch_alignment.py --binary=./build/cifar10

# (Optional) Export standalone PyTorch reference weights to example/cifar10/weights/
python3 example/cifar10/export_alignment_weights.py
```

## Verified Results (NVIDIA A40 46GB, FP32, TF32=OFF)

### 20-Epoch CIFAR-10 Convergence (`bs=128`, SGD `momentum=0.9`, `wd=5e-4`, Cosine LR `0.1`, Baseline Augment)

| Model | Config | Final Test Acc (Epoch 20) | Peak Test Acc | Target |
| --- | --- | --- | --- | --- |
| **ResNet-18** | 1× A40 | **91.94%** (9194/10000) | **92.02%** | ≥ 80% |
| **ResNet-18** | 2× A40 DDP | **91.12%** (9112/10000) | **91.12%** | ≥ 80% |
| **MobileNetV2** | 1× A40 | **91.38%** (9138/10000) | **91.38%** | ≥ 70% |
| **MobileNetV2** | 2× A40 DDP | **90.44%** (9044/10000) | **90.44%** | ≥ 70% |

### Single-GPU CUDA Throughput vs PyTorch (FP32, `3x32x32`, deterministic cuDNN)

| Model | Batch Size | PyTorch (cuDNN) | InfiniTrain (`USE_CUDNN=OFF`) | InfiniTrain (`USE_CUDNN=ON`, deterministic) |
| --- | --- | --- | --- | --- |
| **ResNet-18** | 32 | 2,800.0 img/s | 1,727.4 img/s (61.7%) | **1,988.7 img/s (71.0%)** |
| **ResNet-18** | 64 | 3,269.1 img/s | 1,794.1 img/s (54.9%) | **2,350.3 img/s (71.9%)** |
| **ResNet-18** | 128 | 3,651.5 img/s | 1,828.2 img/s (50.1%) | **2,816.8 img/s (77.1%)** |
| **ResNet-18** | 256 | 3,732.7 img/s | 1,839.8 img/s (49.3%) | **2,854.5 img/s (76.5%)** |
| **MobileNetV2** | 32 | 2,385.5 img/s | 1,839.0 img/s (77.1%) | **1,352.8 img/s (56.7%)** |
| **MobileNetV2** | 64 | 2,896.8 img/s | 1,997.4 img/s (69.0%) | **1,826.2 img/s (63.0%)** |
| **MobileNetV2** | 128 | 3,394.3 img/s | 2,129.5 img/s (62.7%) | **2,165.9 img/s (63.8%)** |
| **MobileNetV2** | 256 | 3,491.3 img/s | 2,179.1 img/s (62.4%) | **2,419.3 img/s (69.3%)** |

*(With non-deterministic cuDNN backward algorithms enabled, single-GPU ResNet-18 reaches up to 3,144.7 img/s / 91.6% of PyTorch and 2-GPU DDP reaches 6,039.3 img/s; deterministic cuDNN is selected by default so checkpoint resume is bitwise-identical.)*

### 2-GPU DDP Scaling (`bs_per_rank=128`, FP32)

| Model | Backend | 1-GPU | 2-GPU DDP | Speedup | DDP Grad Max Diff |
| --- | --- | --- | --- | --- | --- |
| **ResNet-18** | Custom CUDA (`USE_CUDNN=OFF`) | 1,828.2 img/s | 3,476.1 img/s | **1.90×** | `0.00e+00` |
| **ResNet-18** | cuDNN (`USE_CUDNN=ON`, det.) | 2,792.9 img/s | **5,378.3 img/s** | **1.93×** | `0.00e+00` |
| **MobileNetV2** | Custom CUDA (`USE_CUDNN=OFF`) | 2,129.5 img/s | 4,152.2 img/s | **1.95×** | `0.00e+00` |
| **MobileNetV2** | cuDNN (`USE_CUDNN=ON`, det.) | 2,166.6 img/s | **4,095.0 img/s** | **1.89×** | `0.00e+00` |

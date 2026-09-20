# MNIST CNN 端到端训练

本文记录任务第三步的可复现配置与实测结果。训练入口完整执行
`Dataset → DataLoader → Forward → CrossEntropyLoss → Backward → SGD → Evaluation`，
并输出 epoch、step、training loss、test loss 与 test accuracy。

## 数据与模型

使用官方 MNIST 的 60,000 张训练图片和 10,000 张测试图片。程序读取解压后的
IDX 文件，将像素转换为 FP32 并除以 255。服务器实测文件的 SHA256 如下：

| 文件 | SHA256 |
| --- | --- |
| `train-images-idx3-ubyte` | `ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db` |
| `train-labels-idx1-ubyte` | `65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5` |
| `t10k-images-idx3-ubyte` | `0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7` |
| `t10k-labels-idx1-ubyte` | `ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2` |

网络为 `Conv2d(1,16,3) → ReLU → Conv2d(16,32,3) → ReLU → Flatten →
Linear(18432,10)`，共 189,130 个可训练参数。详细结构见
[MNIST CNN 网络](mnist_cnn.md)。

## 训练配置

| 参数 | 值 |
| --- | --- |
| dtype | FP32 |
| optimizer | SGD，momentum=0，weight_decay=0 |
| learning rate | 0.05 |
| batch size | 64 |
| epochs | 3 |
| seed | 42 |
| shuffle | 每个 epoch 使用 `seed + epoch` 确定性打乱训练集 |
| evaluation | 训练前一次，每个 epoch 后一次 |
| checkpoint | 第 3 个 epoch 后保存，并重新加载复验 |

实测环境为 NVIDIA GeForce RTX 4090（24564 MiB）、驱动 570.124.06、
CUDA 12.8、GCC 13.3.0 和 CMake 3.31.4。CUDA 与 CPU 使用相同初始化、样本顺序和超参数。

## 训练结果

### CUDA

| epoch | train loss | train accuracy | test loss | test accuracy | train seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | — | — | 2.302524 | 10.21% | — |
| 1 | 0.297060 | 91.173% | 0.111381 | 96.73% | 8.331 |
| 2 | 0.099873 | 97.100% | 0.069132 | 97.88% | 8.306 |
| 3 | 0.069905 | 97.882% | 0.069654 | 97.88% | 8.253 |

### CPU

| epoch | train loss | train accuracy | test loss | test accuracy | train seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | — | — | 2.302522 | 10.21% | — |
| 1 | 0.297042 | 91.175% | 0.111398 | 96.73% | 103.777 |
| 2 | 0.099882 | 97.100% | 0.069135 | 97.88% | 103.192 |
| 3 | 0.069932 | 97.885% | 0.069621 | 97.88% | 103.341 |

两种设备上，测试 loss 均由约 2.3025 降至约 0.0696，测试 accuracy 由
10.21% 提升至 97.88%。CUDA 与 CPU 的最终 accuracy 相同；最终 test loss
相差约 3.3e-5，符合不同 kernel 浮点归约顺序带来的预期差异。两个最终
checkpoint 重新加载后均复现各自的最终 test loss 和 97.88% accuracy。

## 单元测试

| 测试组 | 通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: |
| CNN 基础能力 CPU | 8 | 0 | 0 |
| CNN 基础能力 CUDA | 7 | 1 | 0 |
| MNIST CNN CPU | 7 | 0 | 0 |
| MNIST CNN CUDA | 6 | 1 | 0 |

两个 CUDA 跳过项只检查非法参数和非法输入；校验逻辑与 CPU 共用，已由对应 CPU
测试覆盖。CUDA 的 Forward、Backward、SGD、数据读取、末批处理、指标统计和确定性
初始化路径均实际执行并通过。

## 构建和运行

CPU 构建：

```bash
cmake -S . -B build-cpu -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
  -DUSE_OMP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target mnist -j 4
./build-cpu/mnist --dataset=data/mnist --device=cpu --bs=64 \
  --num_epoch=3 --lr=0.05 --seed=42 --shuffle=true --threads=16 \
  --log_interval=100 --output_dir=runs/mnist-cpu-seed42
```

CUDA 构建：

```bash
cmake -S . -B build-cuda -DUSE_CUDA=ON -DUSE_NCCL=OFF \
  -DUSE_OMP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda --target mnist -j 4
./build-cuda/mnist --dataset=data/mnist --device=cuda --bs=64 \
  --num_epoch=3 --lr=0.05 --seed=42 --shuffle=true --threads=8 \
  --log_interval=100 --output_dir=runs/mnist-cuda-seed42
```

每个输出目录包含 `config.txt`、`metrics.jsonl` 和 `checkpoint/`。程序拒绝写入
非空输出目录，避免覆盖已有结果。用保存的 checkpoint 仅评估：

```bash
./build-cuda/mnist --dataset=data/mnist --device=cuda --bs=64 \
  --eval_only=true --checkpoint=runs/mnist-cuda-seed42/checkpoint \
  --output_dir=runs/mnist-cuda-checkpoint-eval
```

`--dataset` 必须包含四个解压后的 IDX 文件；`--device=cuda` 要求以
`USE_CUDA=ON` 构建。batch size、epoch、learning rate、线程数和日志间隔必须为正值。

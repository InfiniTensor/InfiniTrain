# MNIST CNN 单机多卡 DDP

在已有 CPU/CUDA CNN 上复用框架的 `DistributedDataParallel`、`ProcessGroup`、
NCCL 和 `infini_run`，采用一进程一卡，支持 bucketed 和逐参数两条梯度归约路径。
本 Demo 支持单机多卡；不支持多节点、ZeRO、混合精度或训练中断续训。
checkpoint 可在单卡或双卡上仅评估。

## 构建与启动

需要 CUDA、NCCL 开发库和至少两张可见 GPU。RTX 5090 使用 CUDA 12.8 以上，
`CMAKE_CUDA_ARCHITECTURES=120`；其他 GPU 可设置为 `native`。

```bash
cmake -S . -B build-ddp -DUSE_CUDA=ON -DUSE_NCCL=ON \
  -DBUILD_TEST=ON -DBUILD_MNIST_DDP_TESTS=ON -DUSE_OMP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-ddp --target mnist mnist_align infini_run \
  test_mnist_ddp test_mnist_cpu test_mnist_cuda test_autograd_cpu test_autograd_cuda -j 8

# --bs 是每个 rank 的 batch；此处 global batch = 32 * 2 = 64。
OMP_NUM_THREADS=2 ./build-ddp/infini_run --nproc_per_node=2 ./build-ddp/mnist \
  --ddp=true --device=cuda --dataset=data/mnist --bs=32 \
  --num_epoch=3 --lr=0.05 --seed=42 --threads=8 --output_dir=runs/mnist-ddp

# 与上述训练使用相同的全局 batch、初始权重、样本集合和学习率。
./build-ddp/mnist --device=cuda --dataset=data/mnist --bs=64 \
  --num_epoch=3 --lr=0.05 --seed=42 --threads=8 --output_dir=runs/mnist-single
```

使用独立 NCCL 安装时设置 `-DNCCL_ROOT=/path/to/nccl`，该目录包含 `include/nccl.h`
和 `lib/libnccl.so`。切换已有 build 的 NCCL 版本时先清除 CMake 缓存中的旧路径：
`cmake -S . -B build-ddp -U NCCL_INCLUDE_DIR -U NCCL_LIBRARY -DNCCL_ROOT=/path/to/nccl`。
运行 PyTorch 参考脚本时也将对应 `lib` 目录放在 `LD_LIBRARY_PATH` 首位。

## 训练语义

- `--ddp=false` 默认保持原有 CPU/单卡行为。多进程启动必须显式传 `--ddp=true`。
- `--ddp_buckets=true` 默认复用框架 bucket reducer；设为 `false` 使用逐参数 AllReduce。
- `infini_run` 提供 `RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE` 和独立 rendezvous ID。
  `LOCAL_RANK` 映射可见 GPU，进程组使用 DP 轴，不创建 TP/PP 组。
- 先在 CPU 确定性初始化，迁移到本卡，再将 rank 0 的全部参数广播到所有 rank。
- 每个 epoch 使用同一 `seed+epoch` 生成全局排列，rank r 读取位置 r、r+world_size 等样本。
  训练只丢弃总样本数对 world_size 的余数，每个 rank 样本数相等。
  不足 `--bs` 的本地尾批仍执行；各卡尾批大小相同，NCCL 平均本地 mean-loss 梯度，
  因而等价于全局 batch 的 mean-loss 梯度，学习率不额外乘除卡数。
- MNIST 60,000 个训练样本在双卡上不丢弃也不重复；每卡 30,000 个，最后本地 batch=16，
  全局尾批=32。每个 epoch 938 次 SGD，3 epochs 共 2,814 次。
- 测试集不截断、不补齐；允许不等长或空评估分片。FP64 AllReduce 汇总 loss_sum、correct、
  samples，再计算全局均值，避免平均每卡 accuracy/loss 引入偏差。
- 只由 rank 0 输出全局 JSONL 与 checkpoint；每个 rank 输出独立配置。结束前广播参数副本，
  校验所有 189,130 个参数有限且与 rank 0 逐字节相等，写入 `rank*/replica_check.txt`。
  所有 rank 在保存完成后同步退出。非空输出目录会报错。

## 测试

```bash
OMP_NUM_THREADS=2 ctest --test-dir build-ddp -R '^mnist_ddp_' --output-on-failure -V
./build-ddp/tests/autograd/test_autograd_cpu --gtest_filter='CPU/CnnTest.*'
./build-ddp/tests/autograd/test_autograd_cuda --gtest_filter='CUDA/CnnTest.*'
./build-ddp/tests/mnist/test_mnist_cpu --gtest_filter='CPU/MnistCnnTest.*:MnistDistributedSampler.*'
./build-ddp/tests/mnist/test_mnist_cuda --gtest_filter='CUDA/MnistCnnTest.*'
```

两个 DDP CTest 均实际启动两个进程/两张 GPU，分别验证 bucket 与逐参数路径。
测试故意用不同 seed 初始化两个 rank，验证参数广播；使用 batch=6、2、6 连续三步，
检查 logits、全局 loss、输入梯度、六组参数梯度、SGD 后参数，并覆盖不均匀/空评估分片。
数值判据为 `abs(candidate-reference) <= atol + rtol*abs(reference)`：logits 为
`1e-5/1e-4`，梯度为 `1e-6/1e-3`，更新参数为 `1e-6/1e-5`。
采样器单测验证确定性、分片互斥、截断策略、尾批及全评估集覆盖。
这些测试仅在 `BUILD_MNIST_DDP_TESTS=ON` 时注册（默认 OFF，避免影响单 GPU 的普通 CTest）。
无双卡环境应只运行 CPU/单卡测试，不将其当作已验证的 DDP。

项目报告随附的 `validate_ddp.py`（PR 外）可一键运行上述测试、3 类固定 fixture 的
PyTorch 单卡/DDP 数值对齐、MNIST 全量训练、双卡重复训练与 checkpoint 重新评估。
输入梯度没有 DDP 参数 hook，因此导出时除以 world_size，转换为全局 mean-loss 的导数。
完整训练不要求与单卡逐位相同：不同矩阵 batch 形状和归约次序存在 FP32 舍入差异。
固定短轨迹的阈值对齐与相同配置的独立重复运行分别检验正确性与可复现性。

```bash
./build-ddp/infini_run --nproc_per_node=2 ./build-ddp/mnist \
  --device=cuda --ddp=true --dataset=data/mnist --bs=32 --eval_only=true \
  --checkpoint=runs/mnist-ddp/checkpoint --output_dir=runs/mnist-ddp-eval
```

常见错误：缺少 `USE_NCCL` 时重建；`LOCAL_RANK` 超出 GPU 数量时检查 `CUDA_VISIBLE_DEVICES`；
训练样本少于卡数时减少卡数；出现 NCCL 错误时开启 `NCCL_DEBUG=INFO` 并核对动态库实际版本。
不要用跳过多卡测试或禁用梯度同步替代修复通信环境。

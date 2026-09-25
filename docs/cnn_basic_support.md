# CNN 基础训练能力（任务第一步）

依据：[项目要求：小模型训练支持](https://gxtctab8no8.feishu.cn/docx/QiMHdE5w4omePTx9KBicX6gZnSd)，任务拆解第 1 项。

第二步的网络搭建与 MNIST 示例接入见 [MNIST CNN 网络说明](mnist_cnn.md)。

## 已实现接口

| 能力 | 接口 | 范围 |
| --- | --- | --- |
| 卷积模块 | `nn::Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=true, device=Device())` | 方形卷积核，整数步长、对称零填充，可选偏置 |
| 卷积函数 | `nn::function::Conv2d(input, weight, bias=nullptr, stride=1, padding=0)` | 权重为 OIHW；函数也支持矩形卷积核 |
| 激活 | `nn::ReLU`、`nn::function::ReLU(input)` | 非原地操作，零点导数为 0 |
| 展平模块 | `nn::Flatten(start_dim=1, end_dim=-1)` | 支持负维度，复用 `Tensor::Flatten` 的变形与自动求导 |

卷积与 ReLU 支持 CPU/CUDA 上的 FP32。卷积输入为连续的 `[N,C,H,W]`，权重为 `[C_out,C_in,K_h,K_w]`，执行与 PyTorch Conv2d 相同的互相关计算。当前不提供 groups、dilation、其他 padding 模式、混合精度或非连续张量接口。卷积要求正的 batch、通道与空间尺寸；Flatten 要求合法的非标量维度区间。

输出高度为 `(H + 2 * padding - K_h) / stride + 1`，采用整数向下取整，宽度同理。输入 rank、通道、dtype、设备、bias 形状、stride/padding 和可用空间均在计算前检查。

## 框架接入

- `nn::Conv2d` 注册可训练的 `weight` 和可选 `bias`，复用 Kaiming/Uniform 初始化、参数枚举、设备迁移和优化器。
- `autograd::Conv2d` 保存反向所需张量，通过 Dispatcher 调用后端，计算输入、权重和偏置梯度，并跳过不需要的梯度输出。
- CPU 使用直接卷积，前向、输入梯度和权重梯度可使用 OpenMP；CUDA 每个线程负责一个输出或梯度元素，使用框架当前 stream、DeviceGuard 和 launch 错误检查。无 CPU 回退和浮点原子累加。
- CPU/CUDA 共享标量索引计算，单元测试使用独立的双精度参考实现验证；该实现优先保证正确性，尚未做 im2col/GEMM 或 cuDNN 性能优化。
- 新源码和测试接入项目原有的 CMake 源码收集规则。新增文件后需要重新运行 CMake 配置。
- CUDA 使用 C++20，允许通过 `CMAKE_CUDA_ARCHITECTURES` 指定 GPU 架构；未指定时保留原默认值 `75;80;90`。

必要头文件：

```cpp
#include "infini_train/include/nn/modules/convolution.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/flatten.h"
#include "infini_train/include/nn/functional.h"
```

```cpp
auto conv = std::make_shared<nn::Conv2d>(1, 16, 3);
auto relu = std::make_shared<nn::ReLU>();
auto flatten = std::make_shared<nn::Flatten>();  // 保留 batch 维
auto features = (*flatten)((*relu)((*conv)({input})))[0];
// [N,1,28,28] -> [N,16,26,26] -> [N,10816]
```

与框架现有模块保持一致，参与设备迁移或参数枚举的模块使用 `std::shared_ptr` 管理。

## 测试与复现

测试源文件：`tests/autograd/test_autograd_cnn.cc`，测试组：`CnnTest`。

覆盖内容：

1. 手算卷积前向、输入/权重/bias 梯度及非均匀上游梯度。
2. 多 batch、多输入/输出通道、矩形输入/卷积核、stride/padding、有无 bias。前向与独立双精度参考计算比较；三类梯度逐元素做中心有限差分检查。
3. 全部七种可训练输入/权重/bias 组合，无 bias 模块、1×1 卷积。
4. ReLU 的正负数、零点导数、非原地行为和 NaN/无穷值前向。
5. Flatten 的默认/负维度/局部/全维展平、数值顺序与反向形状恢复。
6. 两层 Conv2d + ReLU + Flatten + Linear + CrossEntropyLoss 的四步 SGD。检查六组参数的梯度非零、有限，每个元素满足 SGD 更新公式，loss 下降且 ZeroGrad 生效。
7. NoGrad 推理、重复反向的梯度累积，以及非法参数失败检查。

前向及参数更新比较使用绝对误差 `1e-5`，有限差分步长 `1e-4`、梯度绝对误差 `2e-5`。测试数据采用可精确表示的二进制小数以降低参考输入转换误差。

在 Linux/WSL 中，从项目根目录运行：

```bash
# 若使用 Git checkout，先补齐仓库声明的 third_party 子模块。
git submodule update --init --recursive

cmake -S . -B build-cpu -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
  -DUSE_OMP=ON -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target test_autograd_cpu -j 6
OMP_NUM_THREADS=2 ./build-cpu/tests/autograd/test_autograd_cpu --gtest_filter='*CnnTest*'

cmake -S . -B build-cuda -DUSE_CUDA=ON -DUSE_NCCL=OFF \
  -DUSE_OMP=ON -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda --target test_autograd_cuda -j 6
OMP_NUM_THREADS=2 ./build-cuda/tests/autograd/test_autograd_cuda --gtest_filter='CUDA/CnnTest.*'
```

CMake 4.x 配合较早版本的第三方依赖时，可增加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`。未在 PATH 中的 nvcc 可用 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` 指定。CUDA 参数错误测试复用 CPU 检查并在 CUDA 测试组中跳过，避免 fork 已初始化的 CUDA context。

本次工作止于基础算子、模块与训练链路单测；MNIST 网络改造、完整数据集训练和 PyTorch 端到端对齐属于文档后续任务。

## 本机验证记录

验证环境：WSL Ubuntu 26.04、GCC 15.2、CMake 4.2.3、CUDA 13.3，GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU（编译参数 `CMAKE_CUDA_ARCHITECTURES=120`），启用 OpenMP。

原压缩包中的第三方目录为空，本机从原依赖上游补齐了 glog 0.7.1、gflags 2.2.2、GoogleTest 1.17.0 和 InfiniTensor/eigen-mirror（5.0.1-dev），未修改这些依赖的源码。

- CPU 独立构建成功，新增 CNN 测试 8/8 通过。
- CPU 自动求导回归：141 项中 140 项通过，1 项原有 BF16 测试按现有条件跳过，无失败。
- CUDA 构建和链接成功，包含新增 Conv2d/ReLU 内核及测试程序。
- 本机曾因 Windows 驱动与 CUDA 13.3 运行库不兼容而无法执行 GPU 测试。最终验收改在驱动 570.124.06、CUDA 12.8、RTX 4090 服务器上完成。
- 服务器实测：CPU 的 8 项 `CnnTest` 全部通过；CUDA 的 7 项计算/训练测试全部通过，1 项非法参数测试因与 CPU 共用校验逻辑按设计跳过。没有失败项。

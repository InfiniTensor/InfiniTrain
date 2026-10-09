# MNIST CNN 网络（任务第二步）

依据：[小模型训练支持项目要求](https://gxtctab8no8.feishu.cn/docx/QiMHdE5w4omePTx9KBicX6gZnSd)，任务拆解第 2 项。网络直接采用文档中的两层卷积结构，并复用第一步的基础能力。

## 网络结构

| 模块名 | 配置 | 输出形状 |
| --- | --- | --- |
| 输入 | FP32，像素除以 255 | `[N,1,28,28]` |
| `conv1` | Conv2d(1,16,3)，stride=1，padding=0，bias=true | `[N,16,26,26]` |
| `relu1` | ReLU | `[N,16,26,26]` |
| `conv2` | Conv2d(16,32,3)，stride=1，padding=0，bias=true | `[N,32,24,24]` |
| `relu2` | ReLU | `[N,32,24,24]` |
| `flatten` | Flatten(1,-1) | `[N,18432]` |
| `fc` | Linear(18432,10)，bias=true | `[N,10]` |

共 **189,130** 个可训练参数，分为 `conv1.weight/bias`、`conv2.weight/bias`、`fc.weight/bias` 六组。输出为原始 logits，直接交给现有 CrossEntropyLoss；网络末尾不额外使用 Softmax。预测时取 logits 最大值对应的类别。

`example/mnist/net.h` / `net.cc` 中的 `MNIST` 类保留原名称，替换原 MLP。网络接受两种批量输入：

- `[N,784]`：适配现有 DataLoader，在模型入口通过可求导的 View 恢复为 NCHW。
- `[N,1,28,28]`：直接用于图像输入和后续参考实现对齐。

其他形状、非 FP32 输入和空 batch 会明确报错。batch 大小不固定，支持末批数量不足的情况。参数初始化沿用 Conv2d/Linear 的默认实现。

## 示例接入与必要修复

- `example/mnist/main.cc` 使用 `shared_ptr<MNIST>` 管理网络，使参数枚举、设备迁移与优化器正常工作；通过模块调用接口执行前向。
- 测试集前向使用 `NoGradGuard`。读取设备回传的指标前同步，避免异步拷贝后提前访问数据。
- 修复 `MNISTDataset` 的图片切片步长：IDX 中每张图片占 784 字节，但归一化成 FP32 后占 3136 字节。以前仍按 784 字节偏移读取，会错误读取第二张及后续图片；现在按实际 FP32 大小切片。
- 保留通用 DataLoader 的原有行为，将图像形状适配局限于 MNIST 网络入口。

使用示例：

```cpp
auto network = std::make_shared<MNIST>();
network->To(device);
infini_train::optimizers::SGD optimizer(network->Parameters(), 0.001f);
infini_train::nn::CrossEntropyLoss loss_fn;
optimizer.ZeroGrad();
auto logits = (*network)({images})[0];
auto loss = loss_fn({logits, labels})[0];
loss->Backward();
optimizer.Step();
```

这里的 `images` 是已归一化的 FP32 Tensor，`labels` 是类别索引 Tensor，两者已位于所选设备；构建网络之后先迁移设备，再创建优化器。

## 验证与运行

新增 `tests/mnist/test_mnist.cc`，CMake 注册 `test_mnist_cpu` / `test_mnist_cuda`。测试覆盖逐层输出尺寸、六组参数名称及总量、batch=1/2/3、两种输入布局的结果一致性、原始 logits、无梯度推理、完整模型的 Loss/Backward/SGD、非法输入，以及真实 IDX 格式的合成文件与末批数据接入。

```bash
cmake -S . -B build-cpu -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
  -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target mnist test_mnist_cpu -j 6
OMP_NUM_THREADS=2 ctest --test-dir build-cpu -R MnistCnnTest --output-on-failure

# 准备好真实 MNIST 数据后，可继续使用原示例入口。
OMP_NUM_THREADS=2 ./build-cpu/mnist --device cpu --dataset data/mnist --bs 64 --num_epoch 1 --lr 0.01
```

完整数据集训练采用 `batch_size=64`、`epochs=3`、`lr=0.05`、`seed=42`，配置与结果见 [MNIST 端到端训练](mnist_training.md)。CMake 4.x 配合较旧依赖时需额外传入 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`。

CUDA 构建及环境要求见 [CNN 基础能力说明](cnn_basic_support.md)。已有 CUDA 构建目录可执行：

```bash
cmake -S . -B build-cuda
cmake --build build-cuda --target mnist test_mnist_cuda -j 6
OMP_NUM_THREADS=2 ./build-cuda/tests/mnist/test_mnist_cuda --gtest_filter='CUDA/*'
```

真实 MNIST 已在 CPU 和 CUDA 上完成端到端训练；PyTorch 数值对齐也已在同一 CUDA 环境完成。第二步的合成数据检查仍只用于验证网络接入，不作为准确率结果。

## 本机验证记录

验证环境沿用第一步的 WSL / GCC / CUDA 环境。CPU 版 `mnist` 已成功构建，并用三张合成 IDX 图片、batch=2、1 epoch、lr=0.001 跑完训练和评估，覆盖末批大小为 1 的情况，loss 有限、程序正常退出。该检查只证明入口可运行，不代表真实手写数字识别精度。

CPU 的 5 项 `MnistCnnTest` 均经 CTest 运行通过：逐层尺寸/参数、输入布局/批量大小、完整网络反向与 SGD、IDX 读取/末批接入、非法输入检查。反向测试确认全部六组参数梯度有限且非零，更新逐元素符合 SGD 公式，更新后的同批次 loss 下降。

最终在驱动 570.124.06、CUDA 12.8、RTX 4090 服务器上完成 GPU 验收。CPU 的 7 项 `MnistCnnTest` 全部通过；CUDA 的 6 项计算/训练测试通过，1 项非法输入测试因与 CPU 共用校验逻辑按设计跳过。相同 seed 和超参数下，CPU/CUDA 都完成 3 个 epoch 的真实 MNIST 训练并达到 97.88% 测试准确率，详见 [MNIST 端到端训练](mnist_training.md)。

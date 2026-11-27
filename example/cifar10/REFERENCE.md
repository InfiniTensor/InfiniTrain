# 参考资源说明（REFERENCE.md）

本次赛题（2026 秋季启元人工智能大赛 — 赛道二 赛题二：InfiniTrain 小型视觉模型训练支持）提交的全部 C++/CUDA 算子实现、Autograd 封装、Module/Optimizer 扩展、CIFAR-10 数据集加载与 DDP 训练示例、Checkpoint 读写及数值对齐验证脚本均由参赛选手 **闫超美（启忆智存 / `yanchaomei`）** 独立设计与编写。

实现过程中参考的公开学术论文、官方框架数学定义及标准数据集规范如下：

1. **PyTorch & TorchVision 官方文档与数学定义**
   - **链接**：https://github.com/pytorch/pytorch 、https://github.com/pytorch/vision
   - **开源协议**：BSD-3-Clause
   - **参考内容**：`nn.Conv2d`、`nn.BatchNorm2d`（含无偏方差更新 `running_var` 与 train/eval 行为）、`nn.ReLU`、`nn.ReLU6`、`nn.MaxPool2d`、`nn.AdaptiveAvgPool2d`、`nn.Flatten`、`nn.init.kaiming_normal_` 及 `torch.optim.SGD`（含 `momentum` 与 `weight_decay` 更新公式）的公开数学定义与接口语义；`ResNet-18`（CIFAR-10 3×3 stride-1 stem 变体）与 `MobileNetV2`（CIFAR-10 适配 stride）的层结构与权重命名规范，用于确保与 PyTorch 参考模型的 1:1 数值对齐验证。
   - **独立开发与优化说明**：全部 CPU/CUDA 前反向内核、InfiniTrain Autograd/Module/Dispatcher 注册、`Conv2d` 零依赖自研多路径 CUDA 内核（Depthwise 专用 kernel、1×1 零 im2col Strided-Batched GEMM、分块 im2col + cuBLAS SGEMM、确定性 gather col2im、线程局部可复用显存工作区）与可选确定性 cuDNN 后端、FP64 统计累加 `BatchNorm2d`、融合 `SGDAccumulateGrad` 内核及完整 DDP 训练/Checkpoint 续训流程均为独立从零实现。

2. **Deep Residual Learning for Image Recognition (ResNet) & MobileNetV2 原始论文**
   - **链接**：https://arxiv.org/abs/1512.03385 、https://arxiv.org/abs/1801.04381
   - **参考内容**：残差块（BasicBlock）与倒残差块（InvertedResidual）结构定义及超参数配置。
   - **独立开发说明**：基于 InfiniTrain `nn::CloneableModule` 与 `nn::Sequential` 独立构建 C++ 模型实现。

3. **CIFAR-10 官方二进制数据集格式**
   - **链接**：https://www.cs.toronto.edu/~kriz/cifar.html
   - **参考内容**：`data_batch_{1..5}.bin` / `test_batch.bin` 每条样本 `1 byte label + 3072 bytes CHW uint8` 的标准二进制格式说明。
   - **独立开发说明**：在 `example/cifar10/dataset.{h,cc}` 中独立实现 OpenMP 批构建、数据增强与无重叠确定性 DDP 分片逻辑。

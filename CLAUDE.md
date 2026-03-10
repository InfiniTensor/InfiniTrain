# 任务名：在example目录下将gpt2和llama3的推理中的attention部分换为实现好了的flash-atten

## 背景介绍
在之前，你已经实现好了flash-attenion的forward和backward, 并且在框架内可以正常调用，你实现的可以在infini_train/src/kernels/cuda/flash_attention.cu 和 infini_train/src/autograd/flash_attention.cc里面看到，以及你在example目录中的每个模型中的net已经加上了flash_attention

## 任务具体介绍
任务可以拆解为：
- **命令行参数接入**: 在各自模型的 main.cc 中添加 --flash 参数。在训练时通过该参数决定是否启用 FlashAttention。

- **网络结构修改**：在各自模型的 `net.cc` 中 `CausalSelfAttention::Forward` 增加逻辑分支:
    - `--flash=false` -> 走原始小算子拼接版本。
    - `--flash=true` -> 调用 FlashAttention 融合算子。

- **测试与验证**:
    - 运行`scripts/run_models_and_profile.bash`,对所有端到端实验进行验证并输出 log。
    - 与原始实现对比:
        -  确认训练精度一致（也可选择与 PyTorch 开启 FlashAttention 的训练 loss 对比，允许浮点差异）。
        -  提供性能对比数据，验证加速效果。

## 当前你的问题:
- 在flash attention下，你跑的模型loss会出现NAN, 即使在测试代码中`test/test_flash_attention.cc`,forward和backward的过程应该是对的

## 问题修复建议:
- 你可以在训练gpt2或llama3中对比flash 和 non-flash每一层的输出，看看是否是forward出了问题，定位到是哪一个layer出了问题
- 如果forward没有问题，那么你应该看看backward的哪一个layer出了问题，梯度是否计算正确，对于参数是否更新正确

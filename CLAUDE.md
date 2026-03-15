# 任务目标：将实现的flash attention算子接入此训练框架中

## 相关信息
- 自己实现的flash-attention在`my-flash-attention`目录下面
- 你需要使用此训练框架的数据结构, 而在`my-flash-attention`中的算子为了验证正确性，使用了torch cpp扩展，你需要酌情修改或者扩展

## 任务拆解:
你需要在`example`目录下的`gpt2`和`llama3`接入此flash-attention, 具体而言
1. 命令行参数接入
   * 在各自模型的 `main.cc` 中添加 `--flash` 参数。
   * 在训练时通过该参数决定是否启用 FlashAttention。

2. 网络结构修改
   * 在各自模型的 `net.cc` 中 `CausalSelfAttention::Forward` 增加逻辑分支：
     * `--flash=false` -> 走原始小算子拼接版本。
     * `--flash=true` -> 调用 FlashAttention 融合算子。

3. 算子接口定义
   * 在 `infini_train/src/nn/functional.cc` 中添加接口：
     
     ```cpp
     std::shared_ptr<Tensor> ScaledDotProductAttention(const std::shared_ptr<Tensor> &qu
         // TODO: call autograd function
     }
     ```
   * 参数语义与 PyTorch `scaled_dot_product_attention` 保持一致。

4. Autograd Function 封装
   * 在 `infini_train/include/autograd` 与 `infini_train/src/autograd` 下定义并实现 ScaledDotProductAttention：
     * `Forward` : 调用 FlashAttention forward CUDA kernel。
     * `SetupContext` : 【可选】存储必要信息用于反向传播。
     * `Backward` : 调用 FlashAttention backward CUDA kernel，支持梯度回传。

5. 正确性检验:
    * 与原版的attentio相比，在接入了flash-attention后，训练loss的的精度大致应该相同
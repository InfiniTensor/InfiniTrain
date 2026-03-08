下面我把 **FlashAttention-2 backward pass 的完整执行流程**整理成一份可直接使用的 **Markdown 文档**。内容会分成两层：

1. **论文级算法流程**：对应 FlashAttention-2 的标准 backward 思路。
2. **CUDA 实现级执行流程**：结合你当前的 kernel 设计
   `grid_dims(num_heads_kv, batch_size, Tc), block_dims(Tc, Tr)`
   说明一个 block 实际在做什么。

论文中 backward 的核心点包括：将 (Q) 按行块、(K/V) 按列块切分；先计算 (D = \mathrm{rowsum}(dO \circ O))；随后固定一个 (K_j,V_j) 列块，遍历所有 (Q_i) 行块，重算 (S_{ij})、(P_{ij})、(dP_{ij})、(dS_{ij})，并更新 (dV_j,dK_j,dQ_i)。另外，FlashAttention-2 在 backward 中只保存 **row-wise logsumexp (L)**，而不是同时保存 row-max 和 row-sum。 

---

# FlashAttention-2 Backward Pass 执行流程

## 1. 问题定义

给定前向中已经得到的：

* (Q \in \mathbb{R}^{N_q \times d})
* (K \in \mathbb{R}^{N_k \times d})
* (V \in \mathbb{R}^{N_k \times d})
* (O \in \mathbb{R}^{N_q \times d})
* (L \in \mathbb{R}^{N_q})，其中 (L) 是每一行 softmax 的 `logsumexp`
* 上游梯度 (dO \in \mathbb{R}^{N_q \times d})

要求计算：

* (dQ \in \mathbb{R}^{N_q \times d})
* (dK \in \mathbb{R}^{N_k \times d})
* (dV \in \mathbb{R}^{N_k \times d})

FlashAttention-2 backward 的基本思想不是存整张 attention matrix，而是 **按 tile 重算** 中间量，从而降低 HBM 读写。论文的 Algorithm 2 先把 (Q,O,dO,L) 切成行块，把 (K,V) 切成列块，然后以列块为外循环执行 backward。

---

## 2. 数学公式

### 2.1 前向关系

对任意一个 tile ((i,j))：

[
S_{ij} = Q_i K_j^T
]

若启用缩放：

[
S_{ij} \leftarrow \text{scale} \cdot S_{ij}
]

softmax 概率：

[
P_{ij} = \exp(S_{ij} - L_i)
]

这里 (L_i) 是该行块对应的 row-wise logsumexp。FlashAttention-2 backward 只依赖 (L)，这是它相对早期实现的一个简化。

输出满足：

[
O_i = \sum_j P_{ij} V_j
]

---

### 2.2 backward 关键中间量

先定义：

[
D = \mathrm{rowsum}(dO \circ O)
]

即对每一行做逐元素乘法后求和。论文明确把这一步放在主 backward 循环之前，并写回 HBM。

对任意 tile ((i,j))：

1. 先计算
   [
   dP_{ij} = dO_i V_j^T
   ]

2. 再做 softmax backward
   [
   dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)
   ]

3. 然后得到三个输入梯度的 tile 贡献：

[
dV_j \mathrel{+}= P_{ij}^T dO_i
]

[
dK_j \mathrel{+}= dS_{ij}^T Q_i
]

[
dQ_i \mathrel{+}= dS_{ij} K_j
]

这些正是论文 Algorithm 2 的核心更新过程。

---

## 3. Tile 划分

设：

* 行块大小为 (B_r)
* 列块大小为 (B_c)

则：

* (Q) 被分成 (T_r = \lceil N_q / B_r \rceil) 个行块
* (K,V) 被分成 (T_c = \lceil N_k / B_c \rceil) 个列块

论文中明确写出：

* (Q_1,\dots,Q_{T_r})
* (K_1,\dots,K_{T_c})
* (V_1,\dots,V_{T_c})
* (O_i,dO_i,L_i,D_i,dQ_i) 也都按行块划分
* (dK_j,dV_j) 按列块划分。 

---

## 4. 论文级 backward 总流程

## Step 0：输入准备

从前向保留下来：

* `Q, K, V, O`
* `dO`
* `L = logsumexp`
* 以及可选的 `dropout_seed`

初始化输出：

* `dQ = 0`
* `dK = 0`
* `dV = 0`

---

## Step 1：预计算 (D)

对每一行计算：

[
D_i = \sum_{m=1}^{d} dO_{i,m} \cdot O_{i,m}
]

并写回全局内存。

### 作用

这是 softmax backward 所需的行级归约项，用来避免显式构造整张 Jacobian。论文把它作为 backward 前处理步骤。

---

## Step 2：外层遍历列块 (j)

对每个列块 (j = 1,\dots,T_c)：

1. 从 HBM 载入 (K_j, V_j) 到 on-chip SRAM / shared memory
2. 在 shared memory 或寄存器中初始化本列块的局部梯度：

   * (dK_j = 0)
   * (dV_j = 0)

论文 Algorithm 2 就是以列块为外层循环，并在每个 (j) 开始时把 `K_j, V_j` 加载到 SRAM，然后初始化 `dK_j, dV_j`。

---

## Step 3：内层遍历行块 (i)

对每个行块 (i = 1,\dots,T_r)：

### 3.1 载入当前行块数据

从 HBM 载入：

* (Q_i)
* (O_i)（若需要本 kernel 内使用）
* (dO_i)
* (L_i)
* (D_i)

---

### 3.2 重算 score tile

重算当前 tile 的 attention score：

[
S_{ij} = Q_i K_j^T
]

若设置 `scale`：

[
S_{ij} \leftarrow \text{scale} \cdot S_{ij}
]

若启用 `causal mask`，则对非法位置做 mask：

[
S_{ij}(r,c) = -\infty
\quad \text{if } k_idx > q_idx
]

其中：

* `q_idx = i * B_r + r`
* `k_idx = j * B_c + c`

### 说明

这一步没有从全局内存读取旧的 attention matrix，而是 **重算**。这是 FlashAttention backward 相比普通实现最关键的节省显存策略。

---

### 3.3 重算概率 tile

根据保存的 `logsumexp`：

[
P_{ij} = \exp(S_{ij} - L_i)
]

若启用 `dropout`，则要重建前向时相同的 dropout mask，并应用：

[
P_{ij}^{drop} = \frac{M_{ij} \circ P_{ij}}{1-p}
]

其中：

* (M_{ij}) 为基于同一随机种子重建的二值 mask
* (p) 为 dropout probability

如果没有 dropout，则：

[
P_{ij}^{drop} = P_{ij}
]

---

### 3.4 计算 (dV_j) 的当前 tile 贡献

[
dV_j \mathrel{+}= (P_{ij}^{drop})^T dO_i
]

### 解释

因为前向输出是 (O_i = \sum_j P_{ij}V_j)，所以对 (V_j) 的梯度就是概率矩阵转置乘以上游梯度。

---

### 3.5 计算 (dP_{ij})

[
dP_{ij} = dO_i V_j^T
]

若启用 dropout，则 backward 中也要乘上同样的 dropout mask，并带上缩放因子：

[
dP_{ij} \leftarrow \frac{M_{ij} \circ dP_{ij}}{1-p}
]

---

### 3.6 计算 softmax backward 的 (dS_{ij})

[
dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)
]

注意这里使用的是 **未 dropout 前的 softmax 概率** (P_{ij})，而 `dP` 则需要反映 dropout 后的链式法则。

这一步是整套 backward 中最核心的公式，因为 softmax 的 Jacobian 已经被压缩成逐元素乘法加一个行级标量 (D_i)。

---

### 3.7 计算 (dK_j) 的当前 tile 贡献

[
dK_j \mathrel{+}= dS_{ij}^T Q_i
]

若前向有 `scale`，则这里还要乘上相同的缩放因子对梯度链路进行修正。更准确地说，如果前向做的是：

[
S = \text{scale} \cdot QK^T
]

那么由 (S) 向 (Q,K) 回传的梯度都要乘 `scale`。

---

### 3.8 计算 (dQ_i) 的当前 tile 贡献

[
dQ_i \mathrel{+}= dS_{ij} K_j
]

这一步会累加来自所有列块 (j) 的贡献，因此：

* 若一个 block 只处理单个列块，那么 `dQ_i` 往往需要原子加到全局内存
* 或者先在 block 内做局部累加，再统一写回

---

## Step 4：写回本列块的 (dK_j,dV_j)

当所有行块 (i) 都遍历完后，本列块 (j) 的局部梯度已经累加完成：

* 写回 `dK_j`
* 写回 `dV_j`

由于一个列块 (j) 通常只由一个 block 负责，所以这两者一般不需要跨 block 再归约；但若你在 GQA 下让多个 query-head block 共同更新同一组 kv-head，也可能需要额外归约或原子加。

---

## Step 5：结束条件

当所有列块 (j) 处理完成后：

* `dQ`
* `dK`
* `dV`

全部得到。

---

# 5. 结合你当前 CUDA 设计的执行流程

你当前 kernel 为：

```cpp
FlashAttentionBackwardKernel<T><<<grid_dims, block_dims, shared_mem_size, cuda_stream>>>(
    grad_query_ptr, grad_key_ptr, grad_value_ptr,
    query_ptr, key_ptr, value_ptr, output_ptr, grad_output_ptr,
    logsumexp_ptr, dropout_seed_value, attn_mask_ptr,
    scale, is_causal, dropout_p, enable_gqa,
    batch_size, seq_len_q, seq_len_k, num_heads, num_heads_kv, head_dim);
```

其中：

* `grid_dims(num_heads_kv, batch_size, Tc)`
* `block_dims(Tc, Tr)`

这表示：

* `blockIdx.x`：对应一个 `kv head`
* `blockIdx.y`：对应一个 `batch`
* `blockIdx.z`：对应一个列块 (j)

也就是说，**一个 block 固定负责某个 batch、某个 kv-head、某个列块 (j)**。

---

## 5.1 一个 block 的职责

对给定的：

* `batch = blockIdx.y`
* `kv_head = blockIdx.x`
* `col_tile = blockIdx.z`

该 block 需要：

1. 加载当前 `K_j, V_j`
2. 初始化局部 `dK_j, dV_j`
3. 遍历所有相关的 query row tiles (i)
4. 对每个 (i)：

   * 找到对应的 query-head（若启用 GQA，要从 q-head 映射到 kv-head）
   * 载入 `Q_i, dO_i, L_i, D_i`
   * 重算 `S_ij`
   * 应用 causal/mask/dropout
   * 计算 `P_ij, dP_ij, dS_ij`
   * 累加到 `dV_j, dK_j`
   * 把对 `dQ_i` 的贡献写回全局
5. 行块遍历结束后，写回 `dK_j, dV_j`

---

## 5.2 GQA 下的 head 映射

如果启用了 GQA：

* `num_heads` 是 query heads 数
* `num_heads_kv` 是 kv heads 数
* 每个 kv head 会对应若干个 query heads

常见映射方式为：

[
q_head_start = kv_head \times \frac{num_heads}{num_heads_kv}
]

[
q_head_end = (kv_head+1) \times \frac{num_heads}{num_heads_kv}
]

于是当前 block 除了遍历 row tiles (i)，还需要遍历属于这个 kv-head 组的所有 q-head。这样同一个 `K_j,V_j` 会被多个 q-head 共享使用，而 `dK_j,dV_j` 也要把这些 q-head 的贡献全部累加起来。

这与论文中提到的 MQA/GQA backward 需要对共享的 K/V head 梯度求和是一致的。

---

## 5.3 causal mask 的执行方式

若 `is_causal = true`：

对于 tile 内每个元素 ((r,c))，若其全局位置满足：

[
k_global > q_global
]

则该位置无效。

实现上通常有两层优化：

1. **整块跳过**：
   若当前列块完全在当前行块右侧，则整个 tile 不必计算

2. **边界块细粒度判断**：
   若 tile 横跨对角线，则对元素级做 mask

这样可以减少无效 matmul 和 exp 计算。

---

## 5.4 dropout 的执行方式

若 `dropout_p > 0`：

1. 使用 `dropout_seed_value` 和 tile 内元素索引重建与 forward 一致的随机数序列
2. 得到二值 mask
3. 对 `P_ij` 和 `dP_ij` 执行同样的 mask/缩放

这样 backward 就与前向使用的是同一组保留位置。

---

## 5.5 scale 的执行方式

若 `scale != 1`：

前向 score 为：

[
S = \text{scale} \cdot QK^T
]

那么 backward 中：

* 重算 `S_ij` 时要乘 `scale`
* 从 `dS` 回传到 `dQ,dK` 时也要体现这条链路

通常最直接的实现是：

* 先在重算 `S_ij` 时乘 `scale`
* 在 `dQ = dS K`、`dK = dS^T Q` 前再统一乘 `scale`
* 或者直接把 `scale` 融进 `dS`

两种写法本质等价，但要保证链式法则只乘一次，不要漏乘或重复乘。

---

# 6. 一个推荐的伪代码版本

```markdown
for each batch b:
  for each kv head h_kv:
    for each column tile j:
      load K_j, V_j into shared memory
      initialize local dK_j = 0, dV_j = 0

      determine all query heads h_q that map to h_kv (for GQA)

      for each query head h_q in group(h_kv):
        for each row tile i:
          load Q_i, dO_i, L_i, D_i
          optionally load O_i if D is not precomputed

          recompute S_ij = Q_i @ K_j^T
          if scale is enabled:
            S_ij *= scale

          if causal mask is enabled:
            mask invalid positions in S_ij

          P_ij = exp(S_ij - L_i)

          if dropout is enabled:
            reconstruct dropout mask M_ij
            P_drop = M_ij * P_ij / (1 - p)
          else:
            P_drop = P_ij

          dV_j += P_drop^T @ dO_i

          dP_ij = dO_i @ V_j^T
          if dropout is enabled:
            dP_ij = M_ij * dP_ij / (1 - p)

          dS_ij = P_ij * (dP_ij - D_i)

          if scale is enabled:
            dS_ij *= scale

          dK_j += dS_ij^T @ Q_i
          dQ_i += dS_ij @ K_j

          write or atomically accumulate dQ_i to global memory

      write dK_j, dV_j to global memory
```

---

# 7. 实现时的关键注意点

## 7.1 `D = rowsum(dO ∘ O)` 最好预处理

论文就是这么做的。这样主 backward kernel 只需读取 `D_i`，不用再额外读取 `O_i` 并做一次按行归约，能减轻主 kernel 负担。

---

## 7.2 backward 的 matmul 数量更多

论文 benchmark 中指出 backward 由于重算，等价上有更多 matmul，因此 FLOPs 估计通常约为 forward 的 2.5 倍。

---

## 7.3 FlashAttention-2 backward 仍然强调避免 split-K

论文在 warp 工作划分部分提到，backward 也尽量避免 “split-K” 方案，从而减少 shared memory 读写和同步，尽管 backward 的依赖关系比 forward 更复杂。

---

## 7.4 block 大小要兼顾 shared memory 和寄存器压力

论文提到 block size 变大虽然能减少 shared memory 读写，但也会提高寄存器与 shared memory 占用，过大会导致 spilling 或 kernel 根本无法启动。常见 tile 大小会在 `{64,128}` 范围内手工调优。

---

# 8. 最终总结

FlashAttention-2 backward pass 的本质可以概括为：

1. **先预处理行级统计量**

   * 计算 (D = \mathrm{rowsum}(dO \circ O))

2. **按列块固定 `K_j, V_j`**

   * 让 `K_j,V_j` 常驻 shared memory

3. **遍历所有行块 `Q_i`**

   * 重算 (S_{ij})
   * 用保存的 `L_i` 恢复 (P_{ij})
   * 计算 (dP_{ij}, dS_{ij})

4. **逐 tile 累加三个输入梯度**

   * (dV_j += P_{ij}^T dO_i)
   * (dK_j += dS_{ij}^T Q_i)
   * (dQ_i += dS_{ij} K_j)

5. **支持附加功能**

   * causal：在 tile 内做位置裁剪
   * dropout：重建相同随机 mask
   * scale：在 `S` 与其梯度链中保持一致
   * GQA：让多个 q-head 共享同一组 kv-head，并对 `dK/dV` 做组内累加

---

你要是愿意，我下一步可以直接把这份内容继续整理成一份更正式的文档格式，比如：

* **“设计说明书风格” Markdown**
* **“伪代码 + 公式 + CUDA 映射”版**
* 或者直接给你写成 **项目里的 README 小节**。

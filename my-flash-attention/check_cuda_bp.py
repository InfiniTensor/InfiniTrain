import torch
import torch.nn.functional as F

# 参数设置
batch_size = 1
q_head = 1
kv_head = 1
q_seq = 64
kv_seq = 64
head_dim = 64
device = 'cuda:0'

# 随机输入
Q = torch.randn(batch_size, q_head, q_seq, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
K = torch.randn(batch_size, kv_head, kv_seq, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
V = torch.randn(batch_size, kv_head, kv_seq, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

dO = torch.randn(batch_size, q_head, q_seq, head_dim, device=device, dtype=torch.float32)

# 1. Forward (PyTorch SDPA)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    O = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True)

# 2. 计算 L (logsumexp) - 手动实现与 forward kernel 一致
with torch.no_grad():
    scale = head_dim ** 0.5
    S = torch.matmul(Q, K.transpose(-1, -2)) / scale  # [bs, q_head, q_seq, kv_seq]

    # causal mask
    mask = torch.triu(torch.ones(q_seq, kv_seq, device=device, dtype=torch.bool), diagonal=1)
    S.masked_fill_(mask, float('-inf'))

    # softmax 得到 P
    P = torch.softmax(S, dim=-1)

    # L = logsumexp = rowmax + log(sum(exp(S - rowmax)))
    rowmax = S.max(dim=-1, keepdim=True).values
    S_shifted = S - rowmax
    rowsumexp = torch.sum(torch.exp(S_shifted), dim=-1, keepdim=True)
    L = (rowmax.squeeze(-1) + torch.log(rowsumexp.squeeze(-1))).float()  # [bs, q_head, q_seq]

# 3. 计算 D = sum(dO * O, dim=-1)
D = torch.sum(dO * O.float(), dim=-1).float()  # [bs, q_head, q_seq]

# 4. 调用 CUDA backward
# 需要先编译扩展
from torch.utils.cpp_extension import load

print("Compiling CUDA extension...")
attention_bp = load(
    name="attention_bp",
    sources=["attention_v6_bp.cu"],
    extra_include_paths=["."],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    build_directory="/home/tangguochuan/my_tmp",
    verbose=False
)

print("Calling CUDA backward...")
dQ_cuda, dK_cuda, dV_cuda = attention_bp.flash_attention_backward(
    Q.contiguous(),
    K.contiguous(),
    V.contiguous(),
    O.contiguous(),
    L.contiguous(),
    D.contiguous(),
    dO.contiguous(),
    True  # is_causal
)

print("Done!")
print(f"dQ_cuda: {dQ_cuda.shape}, {dQ_cuda.dtype}")
print(f"dK_cuda: {dK_cuda.shape}, {dK_cuda.dtype}")
print(f"dV_cuda: {dV_cuda.shape}, {dV_cuda.dtype}")

# 5. 对比 autograd
O.backward(dO.to(torch.bfloat16))
dQ_autograd = Q.grad.clone().float()
dK_autograd = K.grad.clone().float()
dV_autograd = V.grad.clone().float()

def compare(name, cuda_val, autograd_val):
    diff = torch.abs(cuda_val - autograd_val)
    print(f"[{name}] Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")

print("\n=== Compare CUDA vs Autograd ===")
compare("dQ", dQ_cuda, dQ_autograd)
compare("dK", dK_cuda, dK_autograd)
compare("dV", dV_cuda, dV_autograd)
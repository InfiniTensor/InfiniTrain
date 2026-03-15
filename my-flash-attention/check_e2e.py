"""
End-to-end test: custom FlashAttention forward + backward vs PyTorch autograd.
Forward  : attention_v6.cu   → O (bf16), L (bf16)
Backward : attention_v6_bp.cu → dQ, dK, dV (float)
           D = sum(dO*O) is computed inside the backward kernel; L is passed as bf16.
"""
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

device = 'cuda:0'
head_dim = 64
BUILD_DIR = "/home/tangguochuan/my_tmp"

# ── Compile ───────────────────────────────────────────────────────────────────
print("Compiling forward extension...")
attn_fwd = load(
    name="attention_fwd",
    sources=["attention_v6.cu"],
    extra_include_paths=["."],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    build_directory=BUILD_DIR,
    verbose=False,
)

print("Compiling backward extension...")
attn_bwd = load(
    name="attention_bp",
    sources=["attention_v6_bp.cu"],
    extra_include_paths=["."],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    build_directory=BUILD_DIR,
    verbose=False,
)
print("Done compiling.\n")


def run_test(name, bs, q_head, kv_head, q_seq, kv_seq, is_causal):
    print(f"=== {name}: bs={bs}, q_head={q_head}, kv_head={kv_head}, "
          f"q_seq={q_seq}, kv_seq={kv_seq}, causal={is_causal} ===")

    q_kv_ratio = q_head // kv_head
    Q = torch.randn(bs, q_head,  q_seq,  head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    K = torch.randn(bs, kv_head, kv_seq, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    V = torch.randn(bs, kv_head, kv_seq, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    dO = torch.randn(bs, q_head, q_seq, head_dim, device=device, dtype=torch.float32)

    # ── Custom forward ────────────────────────────────────────────────────────
    O_cuda, L_cuda = attn_fwd.flash_attention_forward(
        Q.contiguous(), K.contiguous(), V.contiguous(), is_causal
    )
    # O_cuda: bf16  [bs, q_head, q_seq, head_dim]
    # L_cuda: bf16  [bs, q_head, q_seq]

    # ── Custom backward ───────────────────────────────────────────────────────
    # D and L conversion are handled inside flash_attention_backward
    dQ_cuda, dK_cuda, dV_cuda = attn_bwd.flash_attention_backward(
        Q.contiguous(), K.contiguous(), V.contiguous(),
        O_cuda.contiguous(), L_cuda.contiguous(), dO.contiguous(),
        is_causal
    )

    # ── Reference (PyTorch autograd) ──────────────────────────────────────────
    # For GQA: expand K/V to q_head dimension
    K_exp = K.repeat_interleave(q_kv_ratio, dim=1)
    V_exp = V.repeat_interleave(q_kv_ratio, dim=1)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        O_ref = F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=is_causal)
    O_ref.backward(dO.to(torch.bfloat16))
    dQ_ref = Q.grad.clone().float()
    dK_ref = K.grad.clone().float()   # autograd accumulates through repeat_interleave
    dV_ref = V.grad.clone().float()

    # ── Compare forward output ────────────────────────────────────────────────
    diff_O = torch.abs(O_cuda.float() - O_ref.float())
    status_O = "✓" if diff_O.max().item() < 0.1 else "✗"
    print(f"  [O]  max_diff={diff_O.max().item():.4e}  mean_diff={diff_O.mean().item():.4e}  {status_O}")

    def compare(tensor_name, cuda_val, ref_val):
        diff = torch.abs(cuda_val - ref_val)
        status = "✓" if diff.max().item() < 0.5 else "✗"
        print(f"  [{tensor_name}] max_diff={diff.max().item():.4e}  mean_diff={diff.mean().item():.4e}  {status}")

    compare("dQ", dQ_cuda, dQ_ref)
    compare("dK", dK_cuda, dK_ref)
    compare("dV", dV_cuda, dV_ref)
    print()


# ── Test cases ────────────────────────────────────────────────────────────────
run_test("non-GQA causal",      bs=4, q_head=8, kv_head=8, q_seq=128, kv_seq=128, is_causal=True)
run_test("non-GQA non-causal",  bs=4, q_head=8, kv_head=8, q_seq=128, kv_seq=128, is_causal=False)
run_test("GQA ratio=2 causal",  bs=4, q_head=8, kv_head=4, q_seq=128, kv_seq=128, is_causal=True)
run_test("GQA ratio=4 causal",  bs=2, q_head=8, kv_head=2, q_seq=128, kv_seq=128, is_causal=True)
run_test("GQA ratio=8 causal",  bs=2, q_head=8, kv_head=1, q_seq=128, kv_seq=128, is_causal=True)

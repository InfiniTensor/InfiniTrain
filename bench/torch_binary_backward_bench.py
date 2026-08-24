"""PyTorch baseline for binary-op backward kernels, BF16 vs FP32.

Covers the patterns that hit InfiniTrain's BinaryBackward:
  1. no-broadcast elementwise (mul / add) on [B*T, C]
  2. broadcast backward where B is a row vector [C] (bias-add style)
"""

import torch
import torch.utils.benchmark as tb

DEV = "cuda"


def bench(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = tb.Timer(stmt="fn()", globals={"fn": fn}).timeit(n)
    return t.median * 1e6  # us


def make_case(rows, cols, dtype, op, broadcast_b):
    a = torch.randn(rows, cols, device=DEV, dtype=dtype, requires_grad=True)
    if broadcast_b:
        b = torch.randn(cols, device=DEV, dtype=dtype, requires_grad=True)
    else:
        b = torch.randn(rows, cols, device=DEV, dtype=dtype, requires_grad=True)
    g = torch.randn(rows, cols, device=DEV, dtype=dtype)

    def fn():
        if a.grad is not None:
            a.grad = None
        if b.grad is not None:
            b.grad = None
        out = (a * b) if op == "mul" else (a + b)
        out.backward(g)

    return fn


print(f"device: {torch.cuda.get_device_name(0)}")
print(f"{'case':<28}{'dtype':<10}{'op':<6}{'bcastB':<8}{'us':>10}")
for rows, cols in [(65536, 768), (8192, 3072), (262144, 768)]:
    for dtype in (torch.bfloat16, torch.float32):
        for op in ("mul", "add"):
            for bcast in (False, True):
                fn = make_case(rows, cols, dtype, op, bcast)
                us = bench(fn)
                print(
                    f"[{rows:>6},{cols:>4}]  {str(dtype).split('.')[-1]:<10}{op:<6}{str(bcast):<8}{us:>10.1f}"
                )

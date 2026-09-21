#!/usr/bin/env python3
"""MnistCnn 对齐用 PyTorch 脚本（与 tests/example/mnist_parity.cc 配对）。

C++ 端（fixture）用固定的 mt19937 seed 生成输入/标签/初始参数，
跑 10 步 SGD 并把每步的 logits / loss / grad / 更新后参数 dump 成 .bin；
本脚本读取同一份 dump 目录，用完全相同的初始权重和输入在 PyTorch 里
重跑 10 步 SGD，逐项对比，阈值 1e-5。

文件布局（C++ 端生成，见 mnist_parity.cc 头注释）：
  meta.txt, input.bin(float32 [B,1,28,28]), labels.bin(uint8 [B]),
  init__<name>.bin,
  step<NNN>__logits.bin, step<NNN>__loss.bin,
  step<NNN>__grad__<name>.bin, step<NNN>__param__<name>.bin

参数名与形状（与 MnistCnn::NamedParameters 排序后一致）：
  conv1.weight [16,1,3,3], conv1.bias [16],
  conv2.weight [32,16,3,3], conv2.bias [32],
  fc.weight [10,18432], fc.bias [10]
注意：InfiniTrain Linear 权重形状为 [out_features, in_features]，
与 torch.nn.Linear.weight 一致，可直接按行主序拷贝。

用法：
  pip install "torch>=2.4 --index-url https://download.pytorch.org/whl/cpu" numpy
  MNIST_PARITY_OUT_DIR=/tmp/mnist_parity ./build/tests/example/test_mnist_parity
  python3 scripts/mnist_parity_torch.py --dump-dir /tmp/mnist_parity

退出码：全部通过返回 0，有任一项超差返回 1。
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("需要先安装 torch（CPU 版即可）：", file=sys.stderr)
    print('  pip install "torch>=2.4 --index-url https://download.pytorch.org/whl/cpu" numpy', file=sys.stderr)
    sys.exit(2)

TOL = 1e-5
STEPS = 10
LR = 0.01

# 与 mnist_parity.cc 中排序后的 NamedParameters 顺序保持一致
PARAM_SPECS = [
    ("conv1.bias", (16,)),
    ("conv1.weight", (16, 1, 3, 3)),
    ("conv2.bias", (32,)),
    ("conv2.weight", (32, 16, 3, 3)),
    ("fc.bias", (10,)),
    ("fc.weight", (10, 18432)),
]


class MnistCnn(torch.nn.Module):
    """Conv2d(1,16,3)->ReLU->Conv2d(16,32,3)->ReLU->Flatten(1)->Linear(18432,10)。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu2 = torch.nn.ReLU()
        self.fc = torch.nn.Linear(18432, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


def read_f32(path: Path, shape: tuple) -> np.ndarray:
    raw = path.read_bytes()
    expect = int(np.prod(shape)) * 4
    assert len(raw) == expect, f"{path}: 字节数 {len(raw)} != 期望 {expect}（形状 {shape}）"
    return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()


def read_u8(path: Path, n: int) -> np.ndarray:
    raw = path.read_bytes()
    assert len(raw) == n, f"{path}: 字节数 {len(raw)} != 期望 {n}"
    return np.frombuffer(raw, dtype=np.uint8).copy()


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser(description="MnistCnn PyTorch 对齐对比（阈值 1e-5）")
    ap.add_argument("--dump-dir", required=True, help="C++ fixture 的输出目录（含 meta.txt/input.bin/...）")
    ap.add_argument("--tol", type=float, default=TOL, help="最大允许绝对误差（默认 1e-5）")
    ap.add_argument("--steps", type=int, default=STEPS, help="对比步数（默认 10，需 <= dump 步数）")
    ap.add_argument("--lr", type=float, default=LR, help="SGD 学习率（默认 0.01，需与 fixture 一致）")
    args = ap.parse_args()

    dump = Path(args.dump_dir)
    assert (dump / "meta.txt").exists(), f"找不到 {dump / 'meta.txt'}，请先运行 test_mnist_parity 生成 dump"
    torch.manual_seed(0)
    torch.set_num_threads(1)

    batch = None
    # 从 init 文件推断 batch：读 input.bin 头部无法得知 B，先用 labels.bin 长度
    labels_np = read_u8(dump / "labels.bin", len((dump / "labels.bin").read_bytes()))
    batch = labels_np.shape[0]
    input_np = read_f32(dump / "input.bin", (batch, 1, 28, 28))

    model = MnistCnn()
    # 用 C++ dump 的初始权重覆盖 torch 默认初始化（逐元素精确对齐起点）
    with torch.no_grad():
        state = model.state_dict()
        name_map = {
            "conv1.bias": "conv1.bias",
            "conv1.weight": "conv1.weight",
            "conv2.bias": "conv2.bias",
            "conv2.weight": "conv2.weight",
            "fc.bias": "fc.bias",
            "fc.weight": "fc.weight",
        }
        for name, shape in PARAM_SPECS:
            arr = read_f32(dump / f"init__{name}.bin", shape)
            state[name_map[name]].copy_(torch.from_numpy(arr))

    x = torch.from_numpy(input_np)
    y = torch.from_numpy(labels_np.astype(np.int64))
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    n_fail = 0
    total_checks = 0
    worst = 0.0
    print(f"{'step':>8} {'check':>22} {'max_abs_diff':>14}  结论")
    for step in range(args.steps):
        tag = f"step{step:03d}"
        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss.backward()

        # 1) logits
        ref_logits = read_f32(dump / f"{tag}__logits.bin", (batch, 10))
        d = max_abs_diff(logits.detach().numpy(), ref_logits)
        ok = d <= args.tol
        print(f"{tag:>8} {'logits':>22} {d:14.3e}  {'通过' if ok else '超差'}")
        total_checks += 1
        worst = max(worst, d)
        n_fail += not ok

        # 2) loss
        ref_loss = read_f32(dump / f"{tag}__loss.bin", (1,))[0]
        d = abs(float(loss.item()) - float(ref_loss))
        ok = d <= args.tol
        print(f"{tag:>8} {'loss':>22} {d:14.3e}  {'通过' if ok else '超差'}")
        total_checks += 1
        worst = max(worst, d)
        n_fail += not ok

        # 3) 每个参数的 grad（SGD step 之前）
        grad_map = {
            "conv1.bias": model.conv1.bias.grad,
            "conv1.weight": model.conv1.weight.grad,
            "conv2.bias": model.conv2.bias.grad,
            "conv2.weight": model.conv2.weight.grad,
            "fc.bias": model.fc.bias.grad,
            "fc.weight": model.fc.weight.grad,
        }
        for name, shape in PARAM_SPECS:
            ref_g = read_f32(dump / f"{tag}__grad__{name}.bin", shape)
            d = max_abs_diff(grad_map[name].detach().numpy(), ref_g)
            ok = d <= args.tol
            print(f"{tag:>8} {('grad/' + name):>22} {d:14.3e}  {'通过' if ok else '超差'}")
            total_checks += 1
            worst = max(worst, d)
            n_fail += not ok

        opt.step()

        # 4) 每个参数 SGD 更新后的值
        param_map = {
            "conv1.bias": model.conv1.bias.detach(),
            "conv1.weight": model.conv1.weight.detach(),
            "conv2.bias": model.conv2.bias.detach(),
            "conv2.weight": model.conv2.weight.detach(),
            "fc.bias": model.fc.bias.detach(),
            "fc.weight": model.fc.weight.detach(),
        }
        for name, shape in PARAM_SPECS:
            ref_p = read_f32(dump / f"{tag}__param__{name}.bin", shape)
            d = max_abs_diff(param_map[name].numpy(), ref_p)
            ok = d <= args.tol
            print(f"{tag:>8} {('param/' + name):>22} {d:14.3e}  {'通过' if ok else '超差'}")
            total_checks += 1
            worst = max(worst, d)
            n_fail += not ok

    print(f"\n共 {total_checks} 项对比，超差 {n_fail} 项，最坏误差 {worst:.3e}（阈值 {args.tol:.0e}）")
    if n_fail == 0:
        print("结论：通过（140/140 口径：每步 14 项 × 10 步）")
        return 0
    print("结论：未通过，请检查 dump 目录是否与当前代码版本匹配", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

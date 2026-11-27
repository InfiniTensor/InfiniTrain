#!/usr/bin/env python3
"""Export deterministic PyTorch reference weights for InfiniTrain CIFAR-10 alignment.

Generates portable IFTB binary weight files under example/cifar10/weights/
without committing large binary blobs to git.
"""

import argparse
import os

import torch
from verify_pytorch_alignment import (
    build_ref_model,
    filter_state_dict,
    save_tensor_map,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
        help="Output directory for exported IFTB weight files",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for model_name in ("resnet18", "mobilenetv2"):
        torch.manual_seed(args.seed)
        model = build_ref_model(model_name)
        sd = filter_state_dict(model.state_dict())
        out_path = os.path.join(args.out_dir, f"{model_name}_init_seed{args.seed}.bin")
        save_tensor_map(out_path, sd)
        print(f"Exported {model_name} ({len(sd)} tensors) -> {out_path}")


if __name__ == "__main__":
    main()

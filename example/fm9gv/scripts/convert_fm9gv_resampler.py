#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import torch

MAGIC = b"FM9RS001"


def write_tensor(handle, tensor):
    handle.write(tensor.detach().cpu().float().contiguous().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Convert official FM9G4B-V Resampler weights for InfiniTrain"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    prefix = "resampler."
    get = lambda name: state[prefix + name]
    in_proj_weight = get("attn.in_proj_weight")
    in_proj_bias = get("attn.in_proj_bias")
    q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)
    q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
    tensors = [
        get("pos_embed"),
        get("feature_1x_embedding"),
        get("feature_4x_embedding"),
        get("query"),
        get("features_1x_projector.weight"),
        get("features_1x_projector.bias"),
        get("features_4x_projector.weight"),
        get("features_4x_projector.bias"),
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        get("attn.out_proj.weight"),
        get("attn.out_proj.bias"),
        get("ln_q.weight"),
        get("ln_q.bias"),
        get("ln_kv.weight"),
        get("ln_kv.bias"),
        get("ln_proj.weight"),
        get("ln_proj.bias"),
        get("ln_post.weight"),
        get("ln_post.bias"),
        get("out_mlp.0.weight"),
        get("out_mlp.0.bias"),
        get("out_mlp.2.weight"),
        get("out_mlp.2.bias"),
        get("proj").t(),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<qqqq", 64, 2560, 16, 1152))
        for tensor in tensors:
            write_tensor(handle, tensor)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

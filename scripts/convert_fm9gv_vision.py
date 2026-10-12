#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path


MAGIC = b"FM9VE001"


def require(state_dict, name):
    if name not in state_dict:
        raise KeyError(name)
    return state_dict[name]


def write_tensor_f32(handle, tensor):
    tensor = tensor.detach().cpu().contiguous().float()
    handle.write(tensor.numpy().tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser(description="Convert the FM9G4B-V SigLIP VPM to an InfiniTrain FP32 binary.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    vision = config["vision_config"]
    state = torch.load(args.model_dir / "pytorch_model.bin", map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    hidden_size = int(vision["hidden_size"])
    intermediate_size = int(vision["intermediate_size"])
    num_layers = int(vision["num_hidden_layers"])
    num_heads = int(vision["num_attention_heads"])
    num_channels = int(vision.get("num_channels", 3))
    image_size = int(vision["image_size"])
    patch_size = int(vision["patch_size"])
    layer_norm_eps = float(vision.get("layer_norm_eps", 1e-6))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack(
            "<8s7qf",
            MAGIC,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_channels,
            image_size,
            patch_size,
            layer_norm_eps,
        ))
        write_tensor_f32(handle, require(state, "vpm.embeddings.patch_embedding.weight"))
        write_tensor_f32(handle, require(state, "vpm.embeddings.patch_embedding.bias"))
        write_tensor_f32(handle, require(state, "vpm.embeddings.position_embedding.weight"))
        for layer in range(num_layers):
            prefix = f"vpm.encoder.layers.{layer}."
            for projection in ("k_proj", "v_proj", "q_proj", "out_proj"):
                write_tensor_f32(handle, require(state, prefix + f"self_attn.{projection}.weight"))
                write_tensor_f32(handle, require(state, prefix + f"self_attn.{projection}.bias"))
            write_tensor_f32(handle, require(state, prefix + "layer_norm1.weight"))
            write_tensor_f32(handle, require(state, prefix + "layer_norm1.bias"))
            write_tensor_f32(handle, require(state, prefix + "mlp.fc1.weight"))
            write_tensor_f32(handle, require(state, prefix + "mlp.fc1.bias"))
            write_tensor_f32(handle, require(state, prefix + "mlp.fc2.weight"))
            write_tensor_f32(handle, require(state, prefix + "mlp.fc2.bias"))
            write_tensor_f32(handle, require(state, prefix + "layer_norm2.weight"))
            write_tensor_f32(handle, require(state, prefix + "layer_norm2.bias"))
        write_tensor_f32(handle, require(state, "vpm.post_layernorm.weight"))
        write_tensor_f32(handle, require(state, "vpm.post_layernorm.bias"))

    print(f"converted: {args.output}")
    print(f"layers: {num_layers}, hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")


if __name__ == "__main__":
    main()

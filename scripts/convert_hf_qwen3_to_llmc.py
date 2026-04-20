#!/usr/bin/env python3
"""
Convert HuggingFace Qwen3-8B checkpoint to InfiniTrain LLMC format.

Usage:
    python convert_hf_qwen3_to_llmc.py \
        --hf-path ./Qwen3-8B \
        --output qwen3-8b-fp32.llmc \
        --tp-size 1
"""

import argparse
import struct
import os
import sys

import torch
from safetensors.torch import load_file


# ============================================================
# Qwen3 magic number (different from llama3's 20240803)
# ============================================================
K_QWEN3_MAGIC = 20240804
K_LLMC_FP32_VERSION = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HF Qwen3 to LLMC format")
    parser.add_argument("--hf-path", required=True, help="Path to HF Qwen3 checkpoint dir")
    parser.add_argument("--output", required=True, help="Output LLMC file path")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (default 1)")
    parser.add_argument("--tp-rank", type=int, default=0, help="Tensor parallel rank (default 0)")
    return parser.parse_args()


def load_hf_weights(hf_path):
    """Load all HF safetensors / pytorch_model.bin into a flat dict."""
    print(f"[1/4] Loading HF weights from {hf_path}...")
    state_dict = {}

    if os.path.exists(os.path.join(hf_path, "model.safetensors.index.json")):
        import json
        with open(os.path.join(hf_path, "model.safetensors.index.json")) as f:
            index = json.load(f)
        loaded_files = set()
        for key, filename in index["weight_map"].items():
            if filename not in loaded_files:
                shard = load_file(os.path.join(hf_path, filename), device="cpu")
                state_dict.update(shard)
                loaded_files.add(filename)
                print(f"  Loaded {filename} ({len(shard)} tensors)")
    elif os.path.exists(os.path.join(hf_path, "model.safetensors")):
        state_dict = load_file(os.path.join(hf_path, "model.safetensors"), device="cpu")
    else:
        # Fallback: pytorch_model.bin
        state_dict = torch.load(
            os.path.join(hf_path, "pytorch_model.bin"), map_location="cpu"
        )

    print(f"  Total tensors: {len(state_dict)}")
    return state_dict


def write_header(f, config, tp_rank, tp_size):
    """Write the 1024-byte LLMC header."""
    header = [0] * 256  # 256 int32 slots

    header[0] = K_QWEN3_MAGIC          # magic
    header[1] = K_LLMC_FP32_VERSION    # version
    header[2] = config["block_size"]
    header[3] = config["vocab_size"]
    header[4] = config["n_layer"]
    header[5] = config["n_head"]
    header[6] = config["n_kv_head"]
    header[7] = config["n_embd"]

    # Qwen3 has no ffn_dim_multiplier, write 0.0
    # Pack as float then unpack as int32 for the header slot
    ffn_mult_bytes = struct.pack("f", 0.0)
    ffn_mult_int = struct.unpack("i", ffn_mult_bytes)[0]
    header[8] = ffn_mult_int

    header[9] = config["multiple_of"]

    norm_eps_bytes = struct.pack("f", config["norm_eps"])
    header[10] = struct.unpack("i", norm_eps_bytes)[0]

    rope_theta_bytes = struct.pack("f", config["rope_theta"])
    header[11] = struct.unpack("i", rope_theta_bytes)[0]

    header[12] = int(config["use_scaled_rope"])
    header[13] = config.get("max_gen_bs", 4)
    header[14] = 1  # version_major
    header[15] = 0  # version_minor

    # Write 256 int32 values
    data = struct.pack(f"{len(header)}i", *header)
    assert len(data) == 1024
    f.write(data)
    print(f"[2/4] Header written (magic={K_QWEN3_MAGIC}, vocab={config['vocab_size']}, layers={config['n_layer']})")


def write_matrix(f, tensor):
    """Write a 2D tensor as fp32 row-major."""
    t = tensor.float().cpu()
    assert t.dim() == 2, f"Expected 2D tensor, got {t.dim()}D: {t.shape}"
    arr = t.contiguous().numpy().flatten().tolist()
    data = struct.pack(f"{len(arr)}f", *arr)
    f.write(data)


def write_vector(f, tensor):
    """Write a 1D tensor as fp32."""
    t = tensor.float().cpu()
    assert t.dim() == 1, f"Expected 1D tensor, got {t.dim()}D: {t.shape}"
    arr = t.contiguous().numpy().tolist()
    data = struct.pack(f"{len(arr)}f", *arr)
    f.write(data)


def shard_rows(tensor, tp_rank, tp_size):
    """Row-parallel shard: split along dim 0."""
    if tp_size == 1:
        return tensor
    chunks = torch.chunk(tensor, tp_size, dim=0)
    return chunks[tp_rank]


def shard_cols(tensor, tp_rank, tp_size):
    """Column-parallel shard: split along dim 1."""
    if tp_size == 1:
        return tensor
    chunks = torch.chunk(tensor, tp_size, dim=1)
    return chunks[tp_rank]


def convert(hf_path, output_path, tp_size=1, tp_rank=0):
    """Main conversion pipeline."""

    # ---- Load HF weights ----
    sd = load_hf_weights(hf_path)

    # ---- Build config from HF ----
    import json
    with open(os.path.join(hf_path, "config.json")) as f:
        hf_config = json.load(f)

    config = {
        "block_size": hf_config.get("max_position_embeddings", 40960),
        "vocab_size": hf_config["vocab_size"],
        "n_layer": hf_config["num_hidden_layers"],
        "n_head": hf_config["num_attention_heads"],
        "n_kv_head": hf_config["num_key_value_heads"],
        "n_embd": hf_config["hidden_size"],
        "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
        "rope_theta": hf_config.get("rope_theta", 1000000.0),
        "use_scaled_rope": False,
        "multiple_of": 1,
        "max_gen_bs": 4,
    }

    head_dim = config["n_embd"] // config["n_head"]
    q_out = config["n_embd"]            # 4096
    kv_out = config["n_kv_head"] * head_dim  # 8 * 128 = 1024
    ffn_hidden = hf_config["intermediate_size"]  # 12288

    print(f"[3/4] Config: {config}")
    print(f"  head_dim={head_dim}, q_out={q_out}, kv_out={kv_out}, ffn_hidden={ffn_hidden}")

    # ---- Write LLMC file ----
    with open(output_path, "wb") as f:
        # Header
        write_header(f, config, tp_rank, tp_size)

        # 1. wte.weight [vocab_size, n_embd] → row-shard for TP
        wte = sd["model.embed_tokens.weight"]
        wte_shard = shard_rows(wte, tp_rank, tp_size)
        write_matrix(f, wte_shard)
        print(f"  wte: {wte.shape} → shard {wte_shard.shape}")

        n_layer = config["n_layer"]

        for i in range(n_layer):
            prefix = f"model.layers.{i}"

            if (i + 1) % 6 == 0 or i == 0:
                print(f"  Layer {i}/{n_layer - 1}...")

            # 2. ln_1.weight (input_layernorm) [n_embd] — full copy
            ln1 = sd[f"{prefix}.input_layernorm.weight"]
            write_vector(f, ln1)

            # 3. c_attn.weight [q_out + 2*kv_out, n_embd] — row-shard
            # HF: q_proj, k_proj, v_proj are separate → concat
            q_proj = sd[f"{prefix}.self_attn.q_proj.weight"]    # [n_embd, n_embd]
            k_proj = sd[f"{prefix}.self_attn.k_proj.weight"]    # [kv_out, n_embd]
            v_proj = sd[f"{prefix}.self_attn.v_proj.weight"]    # [kv_out, n_embd]
            c_attn = torch.cat([q_proj, k_proj, v_proj], dim=0)  # [q+2kv, n_embd]
            c_attn_shard = shard_rows(c_attn, tp_rank, tp_size)
            write_matrix(f, c_attn_shard)

            # 4. c_proj (attn o_proj) [n_embd, n_embd] — col-shard
            o_proj = sd[f"{prefix}.self_attn.o_proj.weight"]     # [n_embd, n_embd]
            o_proj_shard = shard_cols(o_proj, tp_rank, tp_size)
            write_matrix(f, o_proj_shard)

            # 5. ln_2.weight (post_attention_layernorm) [n_embd] — full copy
            ln2 = sd[f"{prefix}.post_attention_layernorm.weight"]
            write_vector(f, ln2)

            # 6. c_fc (gate_proj) [ffn_hidden, n_embd] — row-shard
            gate_proj = sd[f"{prefix}.mlp.gate_proj.weight"]
            gate_shard = shard_rows(gate_proj, tp_rank, tp_size)
            write_matrix(f, gate_shard)

            # 7. c_fc2 (up_proj) [ffn_hidden, n_embd] — row-shard
            up_proj = sd[f"{prefix}.mlp.up_proj.weight"]
            up_shard = shard_rows(up_proj, tp_rank, tp_size)
            write_matrix(f, up_shard)

            # 8. c_proj (mlp down_proj) [n_embd, ffn_hidden] — col-shard
            down_proj = sd[f"{prefix}.mlp.down_proj.weight"]
            down_shard = shard_cols(down_proj, tp_rank, tp_size)
            write_matrix(f, down_shard)

        # 9. ln_f.weight (model.norm) [n_embd] — full copy
        ln_f = sd["model.norm.weight"]
        write_vector(f, ln_f)

        # 10. lm_head.weight [vocab_size, n_embd] — row-shard for TP
        lm_head = sd["lm_head.weight"]
        lm_head_shard = shard_rows(lm_head, tp_rank, tp_size)
        write_matrix(f, lm_head_shard)
        print(f"  lm_head: {lm_head.shape} → shard {lm_head_shard.shape}")

    file_size = os.path.getsize(output_path)
    print(f"[4/4] Done! Output: {output_path} ({file_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    args = parse_args()
    convert(args.hf_path, args.output, args.tp_size, args.tp_rank)

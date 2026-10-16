#!/usr/bin/env python3

import os
import shutil
import struct
from pathlib import Path

import torch
from huggingface_hub import get_token
from transformers import AutoConfig, AutoModelForCausalLM


LLMC_MAGIC = 20260513
LLMC_VERSION = 2
LLMC_HEADER_ENTRIES = 256


def validate_config(config):
    expected = {
        "model_type": "mixtral",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }
    for name, value in expected.items():
        actual = getattr(config, name, None)
        if actual != value:
            raise RuntimeError(f"unexpected {name}: expected {value}, got {actual}")

    if config.hidden_size % config.num_attention_heads != 0:
        raise RuntimeError("hidden_size must be divisible by num_attention_heads")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise RuntimeError("num_attention_heads must be divisible by num_key_value_heads")
    if getattr(config, "tie_word_embeddings", False):
        raise RuntimeError("Mixtral-8x7B-v0.1 must not tie embedding and lm_head weights")


def pack_header(config):
    header = bytearray(LLMC_HEADER_ENTRIES * 4)
    values = {
        0: LLMC_MAGIC,
        1: LLMC_VERSION,
        2: int(config.max_position_embeddings),
        3: int(config.vocab_size),
        4: int(config.num_hidden_layers),
        5: int(config.num_attention_heads),
        6: int(config.num_key_value_heads),
        7: int(config.hidden_size),
        8: int(config.num_local_experts),
        14: 0,  # use_scaled_rope
        15: int(config.num_experts_per_tok),
        16: int(config.intermediate_size),
    }
    for slot, value in values.items():
        struct.pack_into("<i", header, slot * 4, value)
    struct.pack_into("<f", header, 12 * 4, float(config.rms_norm_eps))
    struct.pack_into("<f", header, 13 * 4, float(config.rope_theta))
    return header


def expected_payload_numel(config):
    hidden = int(config.hidden_size)
    head_dim = hidden // int(config.num_attention_heads)
    kv_rows = int(config.num_key_value_heads) * head_dim
    intermediate = int(config.intermediate_size)
    experts = int(config.num_local_experts)

    embedding = int(config.vocab_size) * hidden
    attention_per_layer = hidden * hidden + 2 * kv_rows * hidden + hidden * hidden
    moe_per_layer = experts * (2 * intermediate * hidden + hidden * intermediate)
    layer_total = 2 * hidden + attention_per_layer + experts * hidden + moe_per_layer
    final = hidden + int(config.vocab_size) * hidden
    return embedding + int(config.num_hidden_layers) * layer_total + final


def write_tensor(out, tensor, name, chunk_rows=256):
    tensor = tensor.detach()
    if tensor.layout != torch.strided:
        tensor = tensor.to_dense()

    if tensor.ndim <= 1:
        chunks = (tensor,)
    else:
        chunks = (tensor[start : start + chunk_rows] for start in range(0, tensor.shape[0], chunk_rows))

    written = 0
    for chunk in chunks:
        chunk = chunk.to(device="cpu", dtype=torch.float32).contiguous()
        array = chunk.numpy().astype("<f4", copy=False)
        out.write(array.tobytes(order="C"))
        written += chunk.numel()

    if written != tensor.numel():
        raise RuntimeError(f"short tensor write for {name}: expected {tensor.numel()}, wrote {written}")
    return written


def require_tensor(state, key):
    try:
        return state[key]
    except KeyError as error:
        raise KeyError(f"tensor not found in Hugging Face checkpoint: {key}") from error


def find_tensor(state, *keys):
    for key in keys:
        if key in state:
            return state[key], key
    raise KeyError(f"none of the expected tensors were found: {', '.join(keys)}")


def write_payload(out, state, config):
    written = 0

    def emit(key):
        nonlocal written
        written += write_tensor(out, require_tensor(state, key), key)

    def emit_tensor(tensor, name):
        nonlocal written
        written += write_tensor(out, tensor, name)

    emit("model.embed_tokens.weight")
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        print(f"[mixtral] writing layer {layer + 1}/{config.num_hidden_layers}")
        emit(f"{prefix}.input_layernorm.weight")
        emit(f"{prefix}.self_attn.q_proj.weight")
        emit(f"{prefix}.self_attn.k_proj.weight")
        emit(f"{prefix}.self_attn.v_proj.weight")
        emit(f"{prefix}.self_attn.o_proj.weight")
        emit(f"{prefix}.post_attention_layernorm.weight")
        router, router_key = find_tensor(
            state,
            f"{prefix}.mlp.gate.weight",
            f"{prefix}.block_sparse_moe.gate.weight",
        )
        emit_tensor(router, router_key)

        # Transformers <= 4.56 exposes one module per expert. Newer versions
        # pack every expert into gate_up_proj / down_proj parameters. Accept
        # both layouts and always serialize the LLMC order [gate, up, down].
        packed_gate_up_key = f"{prefix}.mlp.experts.gate_up_proj"
        packed_down_key = f"{prefix}.mlp.experts.down_proj"
        if packed_gate_up_key in state:
            gate_up = require_tensor(state, packed_gate_up_key)
            down = require_tensor(state, packed_down_key)
            expected_gate_up = (
                config.num_local_experts,
                2 * config.intermediate_size,
                config.hidden_size,
            )
            expected_down = (
                config.num_local_experts,
                config.hidden_size,
                config.intermediate_size,
            )
            if tuple(gate_up.shape) != expected_gate_up or tuple(down.shape) != expected_down:
                raise RuntimeError(
                    "unexpected packed expert shapes: "
                    f"gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}"
                )
            for expert in range(config.num_local_experts):
                emit_tensor(
                    gate_up[expert, : config.intermediate_size],
                    f"{packed_gate_up_key}[{expert}, gate]",
                )
                emit_tensor(
                    gate_up[expert, config.intermediate_size :],
                    f"{packed_gate_up_key}[{expert}, up]",
                )
                emit_tensor(down[expert], f"{packed_down_key}[{expert}]")
        else:
            for expert in range(config.num_local_experts):
                expert_prefix = f"{prefix}.block_sparse_moe.experts.{expert}"
                emit(f"{expert_prefix}.w1.weight")
                emit(f"{expert_prefix}.w3.weight")
                emit(f"{expert_prefix}.w2.weight")

    emit("model.norm.weight")
    emit("lm_head.weight")
    return written


def main():
    model_id = os.environ.get("MIXTRAL_MODEL_ID", "mistralai/Mixtral-8x7B-v0.1")
    output_dir = Path(os.environ["MIXTRAL_OUTPUT_DIR"])
    cache_dir = Path(os.environ["MIXTRAL_CACHE_DIR"])
    force = os.environ.get("FORCE", "0") == "1"
    local_files_only = os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1"
    revision = os.environ.get("MIXTRAL_REVISION", "main")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mixtral_8x7b_v0.1_fp32.bin"
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(f"[mixtral] skip existing: {output_path}")
        return

    token = os.environ.get("HF_TOKEN") or get_token()
    common_args = {
        "pretrained_model_name_or_path": model_id,
        "revision": revision,
        "cache_dir": str(cache_dir),
        "token": token,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    print(f"[mixtral] reading Hugging Face config for {model_id}@{revision}")
    config = AutoConfig.from_pretrained(**common_args)
    validate_config(config)

    expected_numel = expected_payload_numel(config)
    expected_size = LLMC_HEADER_ENTRIES * 4 + expected_numel * 4
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < expected_size:
        raise RuntimeError(
            f"insufficient free space in {output_dir}: need at least {expected_size / 2**30:.1f} GiB, "
            f"have {free_bytes / 2**30:.1f} GiB"
        )

    print(
        "[mixtral] loading weights with PyTorch/Transformers\n"
        f"           output={output_path}\n"
        f"           expected FP32 size={expected_size / 2**30:.1f} GiB"
    )
    model = AutoModelForCausalLM.from_pretrained(
        **common_args,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.eval()
    state = model.state_dict()

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with tmp_path.open("wb") as out:
        out.write(pack_header(config))
        written_numel = write_payload(out, state, config)

    if written_numel != expected_numel:
        raise RuntimeError(f"payload size mismatch: expected {expected_numel} values, wrote {written_numel}")
    if tmp_path.stat().st_size != expected_size:
        raise RuntimeError(f"file size mismatch: expected {expected_size}, got {tmp_path.stat().st_size}")

    tmp_path.replace(output_path)
    print(f"[mixtral] wrote {written_numel:,} FP32 values -> {output_path}")


if __name__ == "__main__":
    main()

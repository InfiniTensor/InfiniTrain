import argparse
import json
import struct
from pathlib import Path

MAGIC = 20260826
VERSION = 1


def write_tensor_f32(handle, tensor):
    tensor = tensor.detach().cpu().contiguous().float()
    handle.write(tensor.numpy().tobytes(order="C"))


def write_header(handle, config):
    header = bytearray(256 * 4)

    def put_i32(index, value):
        struct.pack_into("<i", header, index * 4, int(value))

    def put_f32(index, value):
        struct.pack_into("<f", header, index * 4, float(value))

    put_i32(0, MAGIC)
    put_i32(1, VERSION)
    put_i32(2, config["max_position_embeddings"])
    put_i32(3, config["vocab_size"])
    put_i32(4, config["num_hidden_layers"])
    put_i32(5, config["num_attention_heads"])
    put_i32(6, config["hidden_size"])
    put_i32(7, config["intermediate_size"])
    put_i32(8, config["q_lora_rank"])
    put_i32(9, config["kv_lora_rank"])
    put_i32(10, config["qk_nope_head_dim"])
    put_i32(11, config["qk_rope_head_dim"])
    put_i32(12, config.get("v_head_dim") or config["qk_nope_head_dim"])
    put_f32(13, config["rope_theta"])
    put_f32(14, config["scale_emb"])
    put_f32(15, config["scale_depth"])
    put_f32(16, config["rms_norm_eps"])
    handle.write(header)


def require(sd, key):
    if key not in sd:
        raise KeyError(key)
    return sd[key]


def main():
    parser = argparse.ArgumentParser(description="Convert FM9G4B-V HF checkpoint to InfiniTrain text-only FP32 bin.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to convert the FM9G checkpoint") from exc

    model_dir = args.model_dir.expanduser()
    checkpoint = model_dir / "pytorch_model.bin"
    config_path = model_dir / "config.json"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    sd = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        write_header(f, config)
        write_tensor_f32(f, require(sd, "llm.model.embed_tokens.weight"))

        for layer in range(config["num_hidden_layers"]):
            prefix = f"llm.model.layers.{layer}"
            ordered_keys = [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.q_a_layernorm.weight",
                f"{prefix}.self_attn.q_b_proj.weight",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                f"{prefix}.self_attn.kv_a_layernorm.weight",
                f"{prefix}.self_attn.kv_b_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
            for key in ordered_keys:
                write_tensor_f32(f, require(sd, key))

        write_tensor_f32(f, require(sd, "llm.model.norm.weight"))
        write_tensor_f32(f, require(sd, "llm.lm_head.weight"))

    print(f"converted: {args.output}")
    print("format: fm9gv_text_fp32")
    print(f"layers: {config['num_hidden_layers']}")
    print(f"vocab_size: {config['vocab_size']}")
    print(f"hidden_size: {config['hidden_size']}")


if __name__ == "__main__":
    main()

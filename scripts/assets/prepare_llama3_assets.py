#!/usr/bin/env python3

import json
import mmap
import os
import struct
from pathlib import Path

import numpy as np
from huggingface_hub import get_token, snapshot_download
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"

out_dir = Path(os.environ["LLAMA3_OUTPUT_DIR"])
cache_dir = Path(os.environ["LLAMA3_CACHE_DIR"])
tiny_path = Path(os.environ["TINY_SHAKESPEARE_TXT"])
force = os.environ.get("FORCE", "0") == "1"
skip_weights = os.environ.get("SKIP_LLAMA3_WEIGHTS", "0") == "1"

out_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise SystemExit(
        f"\nLLaMA3 preparation needs access to {MODEL_ID}.\n"
        "1) Accept the model license on Hugging Face.\n"
        "2) Run `hf auth login` or export HF_TOKEN=hf_xxx.\n"
    )

allow_patterns = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "*.model",
]
if not skip_weights:
    allow_patterns.extend([
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
    ])

print(f"[llama3] downloading/reusing Hugging Face files for {MODEL_ID}")
model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    token=token,
    cache_dir=str(cache_dir),
    allow_patterns=allow_patterns,
))

# ---------------------------------------------------------------------------
# TinyShakespeare -> InfiniTrain / llm.c LLaMA-3 data format
# header: 256 int32 = 1024 bytes
#   [0] magic   = 20240801
#   [1] version = 7
#   [2] ntokens
# payload: uint32 token ids
# ---------------------------------------------------------------------------

def write_datafile(path: Path, toks):
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"[llama3] skip existing: {path}")
        return

    header = np.zeros(256, dtype="<i4")
    header[0] = 20240801
    header[1] = 7
    header[2] = len(toks)
    toks_np = np.asarray(toks, dtype="<u4")

    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
    tmp.replace(path)
    print(f"[llama3] wrote {len(toks):,} tokens -> {path}")

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
    token=token,
    use_fast=True,
)

text = tiny_path.read_text(encoding="utf-8")
sections = text.split("\n\n")

bos = tokenizer.bos_token_id
if bos is None:
    probe = tokenizer.encode("")
    if not probe:
        raise RuntimeError("could not determine LLaMA3 BOS token")
    bos = probe[0]

def encode_no_special(s: str):
    # Match llm.c's tinyshakespeare.py behavior as closely as current
    # transformers versions permit.
    try:
        return tokenizer.encode(
            s,
            add_special_tokens=False,
            verbose=False,
            split_special_tokens=True,
        )
    except TypeError:
        return tokenizer.encode(s, add_special_tokens=False)

tokens = []
for i, section in enumerate(sections):
    tokens.append(int(bos))
    padded = section + "\n\n" if i != len(sections) - 1 else section
    tokens.extend(int(x) for x in encode_no_special(padded))

val_tokens = tokens[:32768]
train_tokens = tokens[32768:]

write_datafile(out_dir / "tiny_shakespeare_val.bin", val_tokens)
write_datafile(out_dir / "tiny_shakespeare_train.bin", train_tokens)

if skip_weights:
    print("[llama3] SKIP_LLAMA3_WEIGHTS=1: dataset prepared; weight conversion skipped")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Hugging Face safetensors -> InfiniTrain LLaMA3 LLMC FP32 format
#
# InfiniTrain's current loader expects:
#   magic   = 20240803
#   version = 3 (FP32)
#
# Followed by weights in the same order as llm.c train_llama3.py:
#   wte
#   all ln_1
#   all packed QKV
#   all attention output projections
#   all ln_2
#   all up projections
#   all gate projections
#   all down projections
#   final norm
#   lm_head
#
# This implementation parses safetensors directly and streams large matrices,
# so it does not need to instantiate the model or load all tensors into memory.
# ---------------------------------------------------------------------------

weight_out = out_dir / "llama3.2_1B_fp32.bin"
if weight_out.exists() and weight_out.stat().st_size > 0 and not force:
    print(f"[llama3] skip existing: {weight_out}")
    raise SystemExit(0)

config = json.loads((model_dir / "config.json").read_text())

hidden = int(config["hidden_size"])
n_layer = int(config["num_hidden_layers"])
n_head = int(config["num_attention_heads"])
n_kv_head = int(config["num_key_value_heads"])
vocab = int(config["vocab_size"])
intermediate = int(config["intermediate_size"])
norm_eps = float(config.get("rms_norm_eps", 1e-5))
rope_theta = float(config.get("rope_theta", 500000.0))

# InfiniTrain's current TinyShakespeare reader caps LLaMA-3 sequence length at
# 8192, matching the llm.c reference config used by this loader.
block_size = 8192

if (hidden, intermediate) != (2048, 8192):
    raise RuntimeError(
        "unexpected LLaMA 3.2 1B dimensions: "
        f"hidden_size={hidden}, intermediate_size={intermediate}"
    )

ffn_dim_multiplier = 1.5
multiple_of = 256

# Match the current InfiniTrain-Test 3.2 1B baseline, which trains with an
# 8192-token context and writes use_scaled_rope=0. InfiniTrain does not yet
# implement the extended-context LLaMA RoPE scaling path.
use_scaled_rope = 0
max_gen_bs = 4

# Sanity checks for the requested model.
if n_head % n_kv_head != 0:
    raise RuntimeError("num_attention_heads must be divisible by num_key_value_heads")
if hidden % n_head != 0:
    raise RuntimeError("hidden_size must be divisible by num_attention_heads")

# Build key -> safetensors shard mapping.
index_path = model_dir / "model.safetensors.index.json"
if index_path.exists():
    index = json.loads(index_path.read_text())
    weight_map = dict(index["weight_map"])
else:
    weight_map = {}

class SafeTensorShard:
    _dtype_map = {
        "F32": np.dtype("<f4"),
        "F16": np.dtype("<f2"),
        "BF16": np.dtype("<u2"),  # converted manually
    }

    def __init__(self, path: Path):
        self.path = path
        self.fp = path.open("rb")
        raw = self.fp.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"invalid safetensors file: {path}")
        self.header_len = struct.unpack("<Q", raw)[0]
        self.header = json.loads(self.fp.read(self.header_len))
        self.data_base = 8 + self.header_len
        self.mm = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

    def has(self, key):
        return key in self.header and key != "__metadata__"

    def raw_array(self, key):
        meta = self.header[key]
        dtype_name = meta["dtype"]
        if dtype_name not in self._dtype_map:
            raise RuntimeError(f"unsupported safetensors dtype {dtype_name} for {key}")
        dtype = self._dtype_map[dtype_name]
        begin, end = meta["data_offsets"]
        shape = tuple(meta["shape"])
        return np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=self.mm,
            offset=self.data_base + begin,
            order="C",
        ), dtype_name

    def close(self):
        self.mm.close()
        self.fp.close()

readers = {}

def reader_for_file(filename: str):
    if filename not in readers:
        readers[filename] = SafeTensorShard(model_dir / filename)
    return readers[filename]

if not weight_map:
    for sf in sorted(model_dir.glob("*.safetensors")):
        r = reader_for_file(sf.name)
        for key in r.header:
            if key != "__metadata__":
                weight_map[key] = sf.name

def locate(key):
    if key not in weight_map:
        raise KeyError(f"tensor not found in HF checkpoint: {key}")
    return reader_for_file(weight_map[key])

def bf16_to_f32(x):
    u32 = np.asarray(x, dtype=np.uint16).astype(np.uint32)
    u32 <<= 16
    return u32.view(np.float32)

def as_f32(x, dtype_name):
    if dtype_name == "BF16":
        return bf16_to_f32(x)
    return np.asarray(x).astype(np.float32, copy=False)

def get_full_f32(key):
    r = locate(key)
    arr, dtype_name = r.raw_array(key)
    return np.ascontiguousarray(as_f32(arr, dtype_name))

def write_tensor_stream(out, key, chunk_rows=256):
    r = locate(key)
    arr, dtype_name = r.raw_array(key)

    if arr.ndim == 0:
        out.write(np.asarray(as_f32(arr, dtype_name), dtype="<f4").tobytes())
        return

    if arr.ndim == 1:
        chunk = np.ascontiguousarray(as_f32(arr, dtype_name), dtype="<f4")
        out.write(chunk.tobytes(order="C"))
        return

    rows = arr.shape[0]
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = np.ascontiguousarray(as_f32(arr[start:end], dtype_name), dtype="<f4")
        out.write(chunk.tobytes(order="C"))

def unpermute(w, n_heads):
    dim1, dim2 = w.shape
    if dim1 % (n_heads * 2) != 0:
        raise RuntimeError(f"cannot unpermute shape={w.shape}, n_heads={n_heads}")
    return (
        w.reshape(n_heads, 2, dim1 // n_heads // 2, dim2)
         .transpose(0, 2, 1, 3)
         .reshape(dim1, dim2)
    )

def pack_header():
    h = bytearray(256 * 4)
    struct.pack_into("<i", h, 0, 20240803)
    struct.pack_into("<i", h, 4, 3)  # FP32
    struct.pack_into("<I", h, 8, block_size)
    struct.pack_into("<I", h, 12, vocab)
    struct.pack_into("<I", h, 16, n_layer)
    struct.pack_into("<I", h, 20, n_head)
    struct.pack_into("<I", h, 24, n_kv_head)
    struct.pack_into("<I", h, 28, hidden)
    struct.pack_into("<f", h, 32, ffn_dim_multiplier)
    struct.pack_into("<I", h, 36, multiple_of)
    struct.pack_into("<f", h, 40, norm_eps)
    struct.pack_into("<f", h, 44, rope_theta)
    struct.pack_into("<i", h, 48, use_scaled_rope)
    struct.pack_into("<i", h, 52, max_gen_bs)
    struct.pack_into("<i", h, 56, 3)
    struct.pack_into("<i", h, 60, 2)
    return h

tmp = weight_out.with_suffix(weight_out.suffix + ".part")
print(
    "[llama3] converting to InfiniTrain FP32 LLMC format\n"
    f"         hidden={hidden}, layers={n_layer}, heads={n_head}, kv_heads={n_kv_head}, "
    f"intermediate={intermediate}, vocab={vocab}\n"
    f"         output={weight_out}\n"
    "         note: the final FP32 file is roughly 6 GB for LLaMA-3.2 1B"
)

with tmp.open("wb") as out:
    out.write(pack_header())

    # token embedding
    write_tensor_stream(out, "model.embed_tokens.weight")

    # attention RMSNorm
    for i in range(n_layer):
        write_tensor_stream(out, f"model.layers.{i}.input_layernorm.weight")

    # packed Q | K | V
    for i in range(n_layer):
        q = get_full_f32(f"model.layers.{i}.self_attn.q_proj.weight")
        k = get_full_f32(f"model.layers.{i}.self_attn.k_proj.weight")

        q = np.ascontiguousarray(unpermute(q, n_head), dtype="<f4")
        k = np.ascontiguousarray(unpermute(k, n_kv_head), dtype="<f4")

        out.write(q.tobytes(order="C"))
        out.write(k.tobytes(order="C"))
        write_tensor_stream(out, f"model.layers.{i}.self_attn.v_proj.weight")

        del q, k

    # attention output projection
    for i in range(n_layer):
        write_tensor_stream(out, f"model.layers.{i}.self_attn.o_proj.weight")

    # FFN RMSNorm
    for i in range(n_layer):
        write_tensor_stream(out, f"model.layers.{i}.post_attention_layernorm.weight")

    # llm.c c_fc  <- HF up_proj
    for i in range(n_layer):
        write_tensor_stream(out, f"model.layers.{i}.mlp.up_proj.weight")

    # llm.c c_fc2 <- HF gate_proj
    for i in range(n_layer):
        write_tensor_stream(out, f"model.layers.{i}.mlp.gate_proj.weight")

    # llm.c c_proj <- HF down_proj
    for i in range(n_layer):
        write_tensor_stream(out, f"model.layers.{i}.mlp.down_proj.weight")

    # final norm
    write_tensor_stream(out, "model.norm.weight")

    # output head
    if "lm_head.weight" in weight_map:
        write_tensor_stream(out, "lm_head.weight")
    elif bool(config.get("tie_word_embeddings", False)):
        write_tensor_stream(out, "model.embed_tokens.weight")
    else:
        raise KeyError("lm_head.weight missing and tie_word_embeddings is false")

tmp.replace(weight_out)

for r in readers.values():
    r.close()

print(f"[llama3] wrote FP32 weights -> {weight_out}")

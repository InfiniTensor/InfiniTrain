#!/usr/bin/env python3
"""Generate a small synthetic GPT-2 LLMC checkpoint and token file.

The pipeline layout regression compares one reference run against several parallel
layouts, so it only needs weights and tokens that are byte-identical across runs --
not pretrained ones. Use this when the real llm.c starter pack cannot be downloaded
(air-gapped machine, blocked mirror) or when a smaller/faster model is enough.

    python3 scripts/assets/make_synthetic_gpt2_assets.py --out-dir data/gpt2-synthetic

Both files use the formats the GPT-2 example already reads: the LLMC fp32 (version 3)
checkpoint layout of example/gpt2/checkpoint_loader.cc and the uint16 token layout of
example/common/tiny_shakespeare_dataset.cc.
"""

import argparse
from pathlib import Path

import numpy as np

CHECKPOINT_MAGIC = 20240326
CHECKPOINT_FP32_VERSION = 3
TOKENS_MAGIC_UINT16 = 20240520
TOKENS_VERSION = 1
HEADER_INTS = 256


def write_checkpoint(path: Path, args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    layers, embd, vocab = args.n_layer, args.n_embd, args.padded_vocab_size

    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = CHECKPOINT_MAGIC
    header[1] = CHECKPOINT_FP32_VERSION
    header[2] = args.block_size
    header[3] = args.vocab_size
    header[4] = layers
    header[5] = args.n_head
    header[6] = embd
    header[7] = args.padded_vocab_size

    def normal(*shape: int, std: float = 0.02) -> np.ndarray:
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    def zeros(*shape: int) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    def ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    # Same tensor order as llm.c: token/position embeddings, then one full pass over
    # the layers per parameter, then the final norm.
    tensors = [
        normal(vocab, embd),
        normal(args.block_size, embd),
        ones(layers, embd),
        zeros(layers, embd),
        normal(layers, 3 * embd, embd),
        zeros(layers, 3 * embd),
        normal(layers, embd, embd),
        zeros(layers, embd),
        ones(layers, embd),
        zeros(layers, embd),
        normal(layers, 4 * embd, embd),
        zeros(layers, 4 * embd),
        normal(layers, embd, 4 * embd),
        zeros(layers, embd),
        ones(embd),
        zeros(embd),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(header.tobytes())
        for tensor in tensors:
            out.write(np.ascontiguousarray(tensor).tobytes())
    print(f"checkpoint: {path} ({path.stat().st_size / (1 << 20):.1f} MiB)")


def write_tokens(path: Path, args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed + 1)

    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = TOKENS_MAGIC_UINT16
    header[1] = TOKENS_VERSION
    header[2] = args.num_tokens

    tokens = rng.integers(0, args.vocab_size, size=args.num_tokens, dtype=np.uint16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(header.tobytes())
        out.write(tokens.tobytes())
    print(f"tokens: {path} ({args.num_tokens} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/gpt2-synthetic")
    parser.add_argument("--n-layer", type=int, default=8)
    # The GPT-2 loader keeps n_kv_head at the GPT2Config() default, and
    # SanitizeGPT2Config() requires n_kv_head == n_head, so n_head must stay 12.
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--padded-vocab-size", type=int, default=0,
                        help="defaults to --vocab-size so that TP and non-TP runs read identical weights")
    parser.add_argument("--num-tokens", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    if args.padded_vocab_size == 0:
        args.padded_vocab_size = args.vocab_size
    if args.padded_vocab_size < args.vocab_size:
        raise SystemExit("--padded-vocab-size must be >= --vocab-size")
    if args.n_embd % args.n_head:
        raise SystemExit("--n-embd must be divisible by --n-head")

    out_dir = Path(args.out_dir)
    write_checkpoint(out_dir / "gpt2_synthetic.bin", args)
    write_tokens(out_dir / "tokens_train.bin", args)


if __name__ == "__main__":
    main()

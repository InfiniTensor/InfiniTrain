#!/usr/bin/env python3
"""Reproducible tiny LLMC fixtures using only the Python standard library.

The integer PRNG and power-of-two float scale are part of fixture version 1.
Files load through the production checkpoint/data readers, with non-repeating
batches to exercise data-cursor resume. No network or model download is needed.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

MODELS = ("gpt2", "llama3", "qwen3", "mixtral")
CONFIG = dict(
    block_size=64,
    vocab_size=128,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=32,
    intermediate_size=64,
)


class Weights:
    def __init__(self, stream):
        self.stream, self.state = stream, 20261009

    def write(self, count, kind="weight"):
        data = []
        for _ in range(count):
            self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
            data.append(
                1.0
                if kind == "norm"
                else 0.0 if kind == "bias" else ((self.state >> 16) - 32768) / 1048576.0
            )
        self.stream.write(struct.pack("<%df" % count, *data))


def weights(model, path):
    c = CONFIG
    b, v, l, h, k, d, f = [
        c[x]
        for x in (
            "block_size",
            "vocab_size",
            "n_layer",
            "n_head",
            "n_kv_head",
            "n_embd",
            "intermediate_size",
        )
    ]
    header = bytearray(1024)

    def fields(fmt, *values):
        struct.pack_into("<" + fmt, header, 0, *values)

    if model == "gpt2":
        fields("8i", 20240326, 3, b, v, l, h, d, v)
    elif model == "llama3":
        fields(
            "8ififf4i",
            20240803,
            3,
            b,
            v,
            l,
            h,
            k,
            d,
            0.75,
            32,
            1e-5,
            10000.0,
            0,
            4,
            3,
            2,
        )
    elif model == "qwen3":
        fields(
            "10iff4i", 20240804, 4, b, v, l, h, k, d, f, 1, 1e-6, 10000.0, 0, 4, 3, 0
        )
    else:
        fields(
            "9iffiff3i",
            20260513,
            2,
            b,
            v,
            l,
            h,
            k,
            d,
            4,
            3.0,
            1.0,
            1,
            1e-5,
            10000.0,
            0,
            2,
            f,
        )
    with path.open("wb") as stream:
        stream.write(header)
        w = Weights(stream)
        w.write(v * d)
        if model == "gpt2":
            w.write(b * d)
            for count, kind in [
                (d, "norm"),
                (d, "bias"),
                (3 * d * d, "weight"),
                (3 * d, "bias"),
                (d * d, "weight"),
                (d, "bias"),
                (d, "norm"),
                (d, "bias"),
                (4 * d * d, "weight"),
                (4 * d, "bias"),
                (4 * d * d, "weight"),
                (d, "bias"),
            ]:
                for _ in range(l):
                    w.write(count, kind)
            w.write(d, "norm")
            w.write(d, "bias")
        elif model in ("llama3", "qwen3"):
            for _ in range(l):
                w.write(d, "norm")
            if model == "qwen3":
                for _ in range(l):
                    w.write(d // h, "norm")
                    w.write(d // h, "norm")
            for count, kind in [
                (3 * d * d, "weight"),
                (d * d, "weight"),
                (d, "norm"),
                (f * d, "weight"),
                (f * d, "weight"),
                (d * f, "weight"),
            ]:
                for _ in range(l):
                    w.write(count, kind)
            w.write(d, "norm")
            w.write(v * d)
        else:
            for _ in range(l):
                w.write(d, "norm")
                w.write(3 * d * d)
                w.write(d * d)
                w.write(d, "norm")
                w.write(4 * d)
                for _ in range(4):
                    w.write(f * d)
                    w.write(f * d)
                    w.write(d * f)
            w.write(d, "norm")
            w.write(v * d)


def tokens(model, path):
    header = bytearray(1024)
    count, state = 16384, 17
    struct.pack_into(
        "<3i", header, 0, 20240520 if model == "gpt2" else 20240801, 1, count
    )
    values = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values.append((state >> 16) % CONFIG["vocab_size"])
    path.write_bytes(
        header
        + struct.pack("<%d%s" % (count, "H" if model == "gpt2" else "I"), *values)
    )


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for model in MODELS:
        for kind, writer in (("weights", weights), ("tokens", tokens)):
            path = output / (model + "_" + kind + ".bin")
            writer(model, path)
            files["tiny." + model + "." + kind] = dict(
                path=path.name,
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    manifest = dict(schema_version=1, fixture_version=1, config=CONFIG, files=files)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)

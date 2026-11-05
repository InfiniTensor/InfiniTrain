#!/usr/bin/env python3
"""Independent PyTorch FP32 oracle for the versioned tiny fixtures.

Uses production-exported initial tensors, independently computes attention,
normalization, routing, gradients and Adam updates. Requires optional torch and
numpy; these are not dependencies of the regression runner or fixture generator.
The oracle intentionally supports only the fixed fixture architecture, unmodified
FP32 parameters, constant LR, and a single logical rank.
"""
import argparse
import io
import json
import math
import struct
import time
from pathlib import Path


def batch_token_offsets(
    num_tokens, sequence_length, micro_batch_size, global_batch_size, step
):
    """Complete microbatch offsets, with data wrapping only between batches.

    Samples start every sequence_length tokens and labels need a following
    token. The dataset reserves its final sequence; incomplete batch tails
    are dropped before the next epoch starts.
    """
    values = (num_tokens, sequence_length, micro_batch_size, global_batch_size, step)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("batch dimensions and step must be positive integers")
    if global_batch_size % micro_batch_size:
        raise ValueError("global batch must contain complete microbatches")
    complete_batches = (num_tokens // sequence_length - 1) // micro_batch_size
    if complete_batches < 1:
        raise ValueError("token fixture has no complete microbatch")
    accumulation = global_batch_size // micro_batch_size
    first_batch = (step - 1) * accumulation
    return [
        ((first_batch + micro) % complete_batches) * micro_batch_size * sequence_length
        for micro in range(accumulation)
    ]


def read_snapshot(manifest_path):
    """Read legacy NPY files or one cached bundle per rank without repeated I/O."""
    import numpy as np

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cache, arrays = {}, {}
    for item in manifest["tensors"]:
        path = manifest_path.parent / item["path"]
        if path not in cache:
            cache[path] = path.read_bytes()
        payload = cache[path]
        if "offset" in item or "length" in item:
            offset, length = item["offset"], item["length"]
            if (
                type(offset) is not int
                or type(length) is not int
                or offset < 0
                or length <= 0
                or offset + length > len(payload)
            ):
                raise ValueError("invalid snapshot bundle extent")
            payload = payload[offset : offset + length]
        arrays[item["kind"], item["name"]] = np.load(
            io.BytesIO(payload), allow_pickle=False
        )
    return manifest, arrays


def run(
    initial_snapshot,
    tokens,
    output,
    steps=4,
    batch_size=2,
    sequence_length=16,
    learning_rate=1e-4,
    candidate_snapshots=None,
    micro_batch_size=None,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    micro_batch_size = batch_size if micro_batch_size is None else micro_batch_size
    initial_snapshot, output = Path(initial_snapshot), Path(output)
    manifest, initial_arrays = read_snapshot(initial_snapshot)
    model = manifest["model"]
    if (
        manifest["step"] != 0
        or manifest["tp"] != 1
        or manifest["pp"] != 1
        or manifest["dtype"] != "float32"
    ):
        raise ValueError("oracle requires step-0, single-rank FP32 snapshot")
    initial = {}
    for item in manifest["tensors"]:
        if item["kind"] == "parameter":
            if "lora_" in item["name"]:
                raise ValueError("oracle does not support LoRA")
            initial[item["name"]] = initial_arrays["parameter", item["name"]]
    # Decode the versioned LLMC fixture independently, so a loader regression
    # cannot pass by feeding the same incorrectly loaded weights to both paths.
    weights_path = Path(tokens).with_name(model + "_weights.bin")
    data = weights_path.read_bytes()
    offset = 1024
    expected = {}

    def read(name, shape):
        nonlocal offset
        count = math.prod(shape)
        expected[name] = (
            np.frombuffer(data, dtype="<f4", count=count, offset=offset)
            .copy()
            .reshape(shape)
        )
        offset += count * 4

    def layers(suffix, shape):
        for layer in range(4):
            read("transformer.h." + str(layer) + suffix, shape)

    read("transformer.wte.weight", (128, 32))
    if model == "gpt2":
        read("transformer.wpe.weight", (64, 32))
        for suffix, shape in [
            (".ln_1.weight", (32,)),
            (".ln_1.bias", (32,)),
            (".attn.c_attn.weight", (96, 32)),
            (".attn.c_attn.bias", (96,)),
            (".attn.c_proj.weight", (32, 32)),
            (".attn.c_proj.bias", (32,)),
            (".ln_2.weight", (32,)),
            (".ln_2.bias", (32,)),
            (".mlp.c_fc.weight", (128, 32)),
            (".mlp.c_fc.bias", (128,)),
            (".mlp.c_proj.weight", (32, 128)),
            (".mlp.c_proj.bias", (32,)),
        ]:
            layers(suffix, shape)
        read("transformer.ln_f.weight", (32,))
        read("transformer.ln_f.bias", (32,))
        expected["lm_head.weight"] = expected["transformer.wte.weight"]
    elif model in ("llama3", "qwen3"):
        layers(".ln_1.weight", (32,))
        if model == "qwen3":
            for layer in range(4):
                read("transformer.h." + str(layer) + ".attn.q_norm.weight", (8,))
                read("transformer.h." + str(layer) + ".attn.k_norm.weight", (8,))
        layers(".attn.c_attn.weight", (96, 32))
        layers(".attn.c_proj.weight", (32, 32))
        layers(".ln_2.weight", (32,))
        first, second = ("up", "gate") if model == "llama3" else ("gate", "up")
        layers(".mlp." + first, (64, 32))
        layers(".mlp." + second, (64, 32))
        for layer in range(4):
            prefix = "transformer.h." + str(layer) + ".mlp."
            expected[prefix + "c_fc.weight"] = np.concatenate(
                (expected.pop(prefix + "gate"), expected.pop(prefix + "up"))
            )
        layers(".mlp.c_proj.weight", (32, 64))
        read("transformer.ln_f.weight", (32,))
        read("lm_head.weight", (128, 32))
    elif model == "mixtral":
        for layer in range(4):
            prefix = "transformer.h." + str(layer)
            for suffix, shape in [
                (".ln_1.weight", (32,)),
                (".attn.c_attn.weight", (96, 32)),
                (".attn.c_proj.weight", (32, 32)),
                (".ln_2.weight", (32,)),
                (".mlp.router.weight", (4, 32)),
            ]:
                read(prefix + suffix, shape)
            for expert in range(4):
                ep = prefix + ".mlp.experts.expert_" + str(expert)
                read(ep + ".c_fc.weight", (128, 32))
                read(ep + ".c_proj.weight", (32, 64))
        read("transformer.ln_f.weight", (32,))
        read("lm_head.weight", (128, 32))
    else:
        raise ValueError("unsupported oracle architecture: " + model)
    if offset != len(data) or initial.keys() != expected.keys():
        raise ValueError(
            "fixture size or initial parameter inventory does not match independent oracle"
        )
    for name in expected:
        if not np.array_equal(initial[name], expected[name]):
            raise ValueError("initial parameter differs from fixture: " + name)
    parameters = {
        name: torch.tensor(value, requires_grad=True)
        for name, value in expected.items()
    }
    # Preserve declared storage aliases: the current framework retains GPT-2
    # weight tying on CPU, while Module::To currently materializes two CUDA
    # parameters. This oracle validates each backend's explicit runtime contract.
    for item in manifest["tensors"]:
        if item["kind"] == "parameter" and item.get("alias_of"):
            parameters[item["name"]] = parameters[item["alias_of"]]
    unique_parameters = list({id(p): p for p in parameters.values()}.values())
    optimizer = (
        torch.optim.SGD(unique_parameters, lr=learning_rate, foreach=False)
        if model == "gpt2"
        else torch.optim.Adam(unique_parameters, lr=learning_rate, foreach=False)
    )
    eps = 1e-6 if model == "qwen3" else 1e-5

    def linear(x, prefix):
        return F.linear(
            x, parameters[prefix + ".weight"], parameters.get(prefix + ".bias")
        )

    def norm(x, prefix):
        w = parameters[prefix + ".weight"]
        if model == "gpt2":
            return F.layer_norm(
                x, (x.shape[-1],), w, parameters[prefix + ".bias"], 1e-5
            )
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w

    def rope(x):
        angles = (
            torch.arange(x.shape[1], dtype=torch.float32)[:, None]
            * (10000.0 ** (-torch.arange(0, 8, 2, dtype=torch.float32) / 8))[None, :]
        )
        cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
        if model == "qwen3":
            a, b = x.chunk(2, dim=-1)
            return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)

    def mlp(x, prefix):
        hidden = linear(x, prefix + ".c_fc")
        if model == "gpt2":
            hidden = F.gelu(hidden, approximate="tanh")
        else:
            gate, up = hidden.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
        return linear(hidden, prefix + ".c_proj")

    def forward(ids):
        x = F.embedding(ids, parameters["transformer.wte.weight"])
        if model == "gpt2":
            x = x + parameters["transformer.wpe.weight"][: ids.shape[1]]
        for layer in range(4):
            prefix = "transformer.h." + str(layer)
            a = norm(x, prefix + ".ln_1")
            q, k, v = linear(a, prefix + ".attn.c_attn").chunk(3, dim=-1)
            q, k, v = [z.reshape(*ids.shape, 4, 8) for z in (q, k, v)]
            if model == "qwen3":
                q, k = norm(q, prefix + ".attn.q_norm"), norm(
                    k, prefix + ".attn.k_norm"
                )
            if model != "gpt2":
                q, k = rope(q), rope(k)
            q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
            scores = q @ k.transpose(-1, -2) / math.sqrt(8)
            mask = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool).triu(1)
            attention = scores.masked_fill(mask, -float("inf")).softmax(-1) @ v
            attention = attention.transpose(1, 2).contiguous().reshape(*ids.shape, 32)
            x = x + linear(attention, prefix + ".attn.c_proj")
            a = norm(x, prefix + ".ln_2")
            if model == "mixtral":
                routing, experts = linear(a, prefix + ".mlp.router").topk(2, dim=-1)
                probabilities = routing.softmax(-1)
                weights = torch.zeros(*a.shape[:-1], 4).scatter(
                    -1, experts, probabilities
                )
                # Dense independent expert evaluation avoids sharing the framework's
                # dispatch, permutation and scatter implementation.
                expert_output = sum(
                    mlp(a, prefix + ".mlp.experts.expert_" + str(e))
                    * weights[..., e : e + 1]
                    for e in range(4)
                )
                x = x + expert_output
            else:
                x = x + mlp(a, prefix + ".mlp")
        return linear(norm(x, "transformer.ln_f"), "lm_head")

    data = Path(tokens).read_bytes()
    magic, _, count = struct.unpack_from("<3i", data)
    tokens = torch.tensor(
        struct.unpack_from(
            "<%d%s" % (count, "H" if magic == 20240520 else "I"), data, 1024
        )
    )
    records = []
    output.mkdir(parents=True, exist_ok=True)

    def candidate_tensors(step, kind):
        directory = Path(candidate_snapshots) / ("step_" + str(step)) / "rank_0"
        _, arrays = read_snapshot(directory / "manifest.json")
        result = {
            name: value
            for (array_kind, name), value in arrays.items()
            if array_kind == kind
        }
        if result.keys() != parameters.keys():
            raise ValueError("candidate " + kind + " inventory differs from oracle")
        for name, value in result.items():
            if (
                tuple(value.shape) != tuple(parameters[name].shape)
                or not np.isfinite(value).all()
            ):
                raise ValueError("invalid candidate " + kind + ": " + name)
        return result

    for step in range(1, steps + 1):
        started = time.perf_counter()
        if candidate_snapshots is not None:
            previous = candidate_tensors(step - 1, "parameter")
            # Independently verify the preceding update BEFORE using its output
            # as this step's exact input. Never copy a candidate final output
            # into the expected snapshot or reset independent optimizer moments.
            with torch.no_grad():
                for name, parameter in parameters.items():
                    if not np.allclose(
                        parameter.detach().numpy(), previous[name], atol=1e-5, rtol=1e-5
                    ):
                        raise ValueError(
                            "preceding independent optimizer update differs: " + name
                        )
                    parameter.copy_(torch.from_numpy(previous[name]))
        optimizer.zero_grad()
        offsets = batch_token_offsets(
            count, sequence_length, micro_batch_size, batch_size, step
        )
        loss_value = torch.zeros(())
        for offset in offsets:
            extent = micro_batch_size * sequence_length
            ids = tokens[offset : offset + extent].reshape(
                micro_batch_size, sequence_length
            )
            labels = tokens[offset + 1 : offset + extent + 1].reshape(
                micro_batch_size, sequence_length
            )
            loss = F.cross_entropy(forward(ids).flatten(0, 1), labels.flatten()) / len(
                offsets
            )
            loss.backward()
            loss_value += loss.detach()
        independent_gradients = {
            name: p.grad.detach().clone() for name, p in parameters.items()
        }
        if candidate_snapshots is not None:
            # Check forward/backward independently, then replay the optimizer
            # with its measured input. Adam's 1e-8 epsilon can magnify harmless
            # cancellation in a 1e-9 gradient into a 1e-5 parameter delta; mixing
            # the two errors would require a much weaker blanket tolerance.
            gradients = candidate_tensors(step, "gradient")
            for name, parameter in parameters.items():
                parameter.grad.copy_(torch.from_numpy(gradients[name]))
        optimizer.step()
        duration_ms = (time.perf_counter() - started) * 1000
        directory = output / ("step_" + str(step)) / "rank_0"
        directory.mkdir(parents=True, exist_ok=True)
        tensors = []
        with (directory / "tensors.bin").open("wb") as bundle:
            for name, p in sorted(parameters.items()):
                for kind, value in (
                    ("parameter", p),
                    ("gradient", independent_gradients[name]),
                ):
                    if value is None:
                        raise ValueError("missing gradient: " + name)
                    offset = bundle.tell()
                    np.save(bundle, value.detach().numpy())
                    tensors.append(
                        dict(
                            name=name,
                            kind=kind,
                            dtype="float32",
                            shape=list(value.shape),
                            path="tensors.bin",
                            offset=offset,
                            length=bundle.tell() - offset,
                        )
                    )
        (directory / "manifest.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    model=model,
                    step=step,
                    rank=0,
                    dp_rank=0,
                    tp_rank=0,
                    pp_rank=0,
                    tensors=tensors,
                )
            )
        )
        records.append(
            dict(
                schema_version=1,
                type="step",
                model=model,
                dtype="float32",
                rank=0,
                dp=1,
                tp=1,
                pp=1,
                sp=1,
                dp_rank=0,
                tp_rank=0,
                pp_rank=0,
                step=step,
                loss=loss_value.item(),
                lr=learning_rate,
                global_tokens=batch_size * sequence_length,
                consumed_samples=step * batch_size,
                step_ms=duration_ms,
                peak_memory_bytes=0,
                oracle=(
                    "pytorch_operations"
                    if candidate_snapshots is not None
                    else "pytorch_trajectory"
                ),
            )
        )
    records.append(
        dict(
            schema_version=1,
            type="complete",
            rank=0,
            step=steps,
            consumed_samples=steps * batch_size,
        )
    )
    (output / "metrics.jsonl.rank_0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-snapshot", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="Microbatch samples and data wrapping boundaries",
    )
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--candidate-snapshots",
        help="Validate independent operations and replay measured optimizer inputs",
    )
    args = vars(parser.parse_args())
    run(**args)

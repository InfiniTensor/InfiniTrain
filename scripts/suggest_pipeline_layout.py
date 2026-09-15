#!/usr/bin/env python3
"""Generate a contiguous Pipeline partition from layer costs or profiler records."""

import argparse
import json
import math
import re
from pathlib import Path


LAYER_RECORD = re.compile(
    r"TransformerLayer\.(\d+)\s+(?:Device\([^)]*\)|\S+)\s*(\d+)\s+(\d+)\s+\d+\s*$"
)


def parse_numbers(value: str) -> list[float]:
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.is_file() else value
    try:
        parsed = json.loads(text)
        values = parsed if isinstance(parsed, list) else parsed["layer_costs"]
    except (json.JSONDecodeError, KeyError, TypeError):
        values = [item.strip() for item in text.strip().split(",")]
    costs = [float(item) for item in values]
    if not costs or any(not math.isfinite(cost) or cost <= 0 for cost in costs):
        raise ValueError("layer costs must be finite positive numbers")
    return costs


def parse_profiler_records(paths: list[str], warmup_samples: int = 0) -> list[float]:
    samples: dict[int, list[float]] = {}
    for value in paths:
        candidates = sorted(Path().glob(value)) if any(ch in value for ch in "*?[") else [Path(value)]
        for path in candidates:
            for line in path.read_text(encoding="utf-8").splitlines():
                match = LAYER_RECORD.search(line)
                if match:
                    layer, host_us, device_us = map(int, match.groups())
                    samples.setdefault(layer, []).append(float(device_us or host_us))
    if not samples or sorted(samples) != list(range(max(samples) + 1)):
        raise ValueError("profiler records must contain contiguous TransformerLayer.0..N samples")
    if warmup_samples < 0 or any(len(values) <= warmup_samples for values in samples.values()):
        raise ValueError("profiler warmup samples must leave at least one sample per layer")
    return [
        sum(samples[layer][warmup_samples:]) / len(samples[layer][warmup_samples:])
        for layer in range(len(samples))
    ]


def balanced_partition(costs: list[float], stages: int) -> tuple[list[int], list[float]]:
    layers = len(costs)
    if stages <= 0 or stages > layers:
        raise ValueError("stages must satisfy 0 < stages <= number of layers")
    prefix = [0.0]
    for cost in costs:
        prefix.append(prefix[-1] + cost)
    best = [[math.inf] * (layers + 1) for _ in range(stages + 1)]
    split = [[-1] * (layers + 1) for _ in range(stages + 1)]
    best[0][0] = 0.0
    for stage_count in range(1, stages + 1):
        for end in range(stage_count, layers + 1):
            for start in range(stage_count - 1, end):
                candidate = max(best[stage_count - 1][start], prefix[end] - prefix[start])
                if candidate < best[stage_count][end]:
                    best[stage_count][end] = candidate
                    split[stage_count][end] = start
    counts = [0] * stages
    end = layers
    for stage in range(stages - 1, -1, -1):
        start = split[stage + 1][end]
        counts[stage] = end - start
        end = start
    stage_costs = []
    start = 0
    for count in counts:
        stage_costs.append(sum(costs[start : start + count]))
        start += count
    return counts, stage_costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--costs", help="CSV/JSON layer costs or a file containing them")
    sources.add_argument("--parameter-counts", help="CSV/JSON per-layer parameter counts or a file")
    sources.add_argument("--profiler-records", nargs="+", help="Profiler record files or glob patterns")
    parser.add_argument("--profiler-warmup-samples", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, required=True)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    costs = (
        parse_profiler_records(args.profiler_records, args.profiler_warmup_samples)
        if args.profiler_records
        else parse_numbers(args.costs or args.parameter_counts)
    )
    counts, stage_costs = balanced_partition(costs, args.pipeline_parallel)
    uniform_counts = [len(costs) // args.pipeline_parallel] * args.pipeline_parallel
    for stage in range(len(costs) % args.pipeline_parallel):
        uniform_counts[stage] += 1
    uniform_costs = []
    offset = 0
    for count in uniform_counts:
        uniform_costs.append(sum(costs[offset : offset + count]))
        offset += count
    bubble = (args.pipeline_parallel - 1) / (args.microbatches + args.pipeline_parallel - 1)
    result = {
        "partition": counts,
        "stage_costs": stage_costs,
        "maximum_stage_cost": max(stage_costs),
        "uniform_partition": uniform_counts,
        "uniform_stage_costs": uniform_costs,
        "uniform_maximum_stage_cost": max(uniform_costs),
        "modeled_maximum_improvement_percent": 100 * (1 - max(stage_costs) / max(uniform_costs)),
        "theoretical_pipeline_bubble_percent": 100 * bubble,
    }
    print("--pipeline_layer_partition=" + ",".join(map(str, counts)))
    print(json.dumps(result, indent=2))
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

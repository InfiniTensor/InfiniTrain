#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def load_array(path):
    return np.fromfile(path, dtype=np.float32)


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def relative_error(actual, reference):
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(actual - reference) / denominator) if denominator else 0.0


def paired_paths(parser, args, left_name, right_name):
    left = getattr(args, left_name)
    right = getattr(args, right_name)
    if bool(left) != bool(right):
        parser.error(f"--{left_name.replace('_', '-')} and --{right_name.replace('_', '-')} must be used together")
    return left, right


def main():
    parser = argparse.ArgumentParser(description="Compare matched FM9G4B-V multi-step training runs.")
    parser.add_argument("--infinitrain-json", type=Path, required=True)
    parser.add_argument("--pytorch-json", type=Path, required=True)
    parser.add_argument("--infinitrain-initial", type=Path)
    parser.add_argument("--pytorch-initial", type=Path)
    parser.add_argument("--infinitrain-final", type=Path)
    parser.add_argument("--pytorch-final", type=Path)
    parser.add_argument("--infinitrain-grad", type=Path)
    parser.add_argument("--pytorch-grad", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    initial_paths = paired_paths(parser, args, "infinitrain_initial", "pytorch_initial")
    final_paths = paired_paths(parser, args, "infinitrain_final", "pytorch_final")
    grad_paths = paired_paths(parser, args, "infinitrain_grad", "pytorch_grad")
    if bool(initial_paths[0]) != bool(final_paths[0]):
        parser.error("initial and final parameter snapshots must be supplied together")

    infini = json.loads(args.infinitrain_json.read_text(encoding="utf-8"))
    torch = json.loads(args.pytorch_json.read_text(encoding="utf-8"))
    infini_losses = np.asarray(infini["losses"], dtype=np.float64)
    torch_losses = np.asarray(torch["losses"], dtype=np.float64)
    if infini_losses.shape != torch_losses.shape:
        raise RuntimeError(f"loss shape mismatch: {infini_losses.shape} vs {torch_losses.shape}")
    loss_error = np.abs(infini_losses - torch_losses)
    result = {
        "steps": int(infini_losses.size),
        "loss": {
            "infinitrain": infini_losses.tolist(),
            "pytorch": torch_losses.tolist(),
            "absolute_error": loss_error.tolist(),
            "max_absolute_error": float(loss_error.max()),
            "mean_absolute_error": float(loss_error.mean()),
        },
        "performance": {
            "infinitrain_mean_ms": float(infini["mean_elapsed_ms"]),
            "pytorch_mean_ms": float(torch["mean_elapsed_ms"]),
            "infinitrain_over_pytorch": float(infini["mean_elapsed_ms"] / torch["mean_elapsed_ms"]),
        },
        "memory": {
            "infinitrain_peak_used_mb": float(infini["peak_used_mb"]),
            "pytorch_peak_allocated_mb": float(torch["peak_allocated_mb"]),
            "used_over_allocated": float(infini["peak_used_mb"] / torch["peak_allocated_mb"]),
            "infinitrain_peak_reserved_mb": float(infini["peak_reserved_mb"]),
            "pytorch_peak_reserved_mb": float(torch["peak_reserved_mb"]),
            "reserved_ratio": float(infini["peak_reserved_mb"] / torch["peak_reserved_mb"]),
        },
    }

    if initial_paths[0]:
        infini_initial, torch_initial = map(load_array, initial_paths)
        infini_final, torch_final = map(load_array, final_paths)
        if not (infini_initial.shape == torch_initial.shape == infini_final.shape == torch_final.shape):
            raise RuntimeError("parameter snapshot shape mismatch")
        infini_delta = infini_final - infini_initial
        torch_delta = torch_final - torch_initial
        result["parameter_update"] = {
            "elements": int(infini_delta.size),
            "initial_max_absolute_error": float(np.max(np.abs(infini_initial - torch_initial))),
            "infinitrain_l2": float(np.linalg.norm(infini_delta)),
            "pytorch_l2": float(np.linalg.norm(torch_delta)),
            "cosine_similarity": cosine(infini_delta, torch_delta),
            "relative_error": relative_error(infini_delta, torch_delta),
        }

    if grad_paths[0]:
        infini_grad, torch_grad = map(load_array, grad_paths)
        if infini_grad.shape != torch_grad.shape:
            raise RuntimeError("gradient snapshot shape mismatch")
        result["gradient"] = {
            "elements": int(infini_grad.size),
            "infinitrain_l2": float(np.linalg.norm(infini_grad)),
            "pytorch_l2": float(np.linalg.norm(torch_grad)),
            "cosine_similarity": cosine(infini_grad, torch_grad),
            "relative_error": relative_error(infini_grad, torch_grad),
            "sign_agreement": float(np.mean(np.signbit(infini_grad) == np.signbit(torch_grad))),
        }

    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

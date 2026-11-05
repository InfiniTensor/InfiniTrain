#!/usr/bin/env python3
"""Compare all legacy training logs, failing on incomplete or invalid evidence."""
import math
import re
import sys
from pathlib import Path
from argparse import ArgumentParser
from compare_utils import (
    matching_logs,
    load_legacy_metrics,
    compare_metrics,
    MetricError,
)


def get_dtype_from_filename(filename):
    # Explicit --dtype is preferred. Keep old names, including Qwen's bf16 alias.
    return (
        "bfloat16"
        if re.search(r"(?:^|[_\-.])(?:bfloat16|bf16)(?:[_\-.]|$)", filename)
        else "float32"
    )


def parse_log(file_path):
    return {row["step"]: row["loss"] for row in load_legacy_metrics(file_path).records}


def compare_files(file1, file2, threshold, dtype=None):
    try:
        dtype = dtype or get_dtype_from_filename(Path(file1).name)
        reference = load_legacy_metrics(file1, dtype=dtype)
        candidate = load_legacy_metrics(file2, dtype=dtype)
        checked = compare_metrics(
            reference, candidate, dtype=dtype, atol=threshold, rtol=0
        )
        return len(reference.records), len(checked["reasons"]), checked["reasons"]
    except (MetricError, ValueError, OSError) as error:
        return 0, 1, [str(error)]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dir1", type=Path)
    parser.add_argument("dir2", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-fp32", type=float, default=1e-5)
    parser.add_argument("--threshold-bf16", type=float, default=1e-2)
    parser.add_argument("--dtype", choices=("float32", "fp32", "bfloat16", "bf16"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.threshold is not None:
        args.threshold_fp32 = args.threshold_bf16 = args.threshold
    if any(
        not math.isfinite(t) or t < 0
        for t in (args.threshold_fp32, args.threshold_bf16)
    ):
        parser.error("thresholds must be finite and nonnegative")
    try:
        baseline, candidate = matching_logs(args.dir1, args.dir2)
    except ValueError as error:
        print(f"[FAIL] {error}")
        return 1
    passed = 0
    for name in sorted(baseline):
        dtype = args.dtype or get_dtype_from_filename(name)
        threshold = (
            args.threshold_bf16
            if dtype in ("bf16", "bfloat16")
            else args.threshold_fp32
        )
        steps, errors, reasons = compare_files(
            baseline[name], candidate[name], threshold, dtype
        )
        passed += not errors
        if errors or args.verbose:
            print(f"[{'FAIL' if errors else 'PASS'}] {name} ({dtype}, {steps} steps)")
            for reason in reasons:
                print(f"  {reason}")
    print(f"Summary: {passed}/{len(baseline)} test cases passed")
    return int(passed != len(baseline))


if __name__ == "__main__":
    sys.exit(main())

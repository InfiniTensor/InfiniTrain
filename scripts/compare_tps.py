#!/usr/bin/env python3
"""Legacy throughput comparison using the runner's arithmetic-mean log policy."""
import math
import statistics
import sys
from pathlib import Path
from argparse import ArgumentParser
from compare_utils import matching_logs, load_legacy_metrics, MetricError


def parse_log(file_path):
    return {
        row["step"]: row["tokens_per_second"]
        for row in load_legacy_metrics(
            file_path, required=("loss", "tokens_per_second")
        ).records
    }


def compare_files(file1, file2, threshold):
    try:
        reference, candidate = parse_log(file1), parse_log(file2)
        if reference.keys() != candidate.keys():
            raise MetricError("Throughput step sets differ")
        steps = list(reference)[1:]
        if not steps:
            raise MetricError("No measurement steps after warmup")
        # Match history.py: exclude the first step, then average logged tok/s.
        avg1 = statistics.mean(reference[step] for step in steps)
        avg2 = statistics.mean(candidate[step] for step in steps)
        change = avg2 / avg1 - 1
        failed = change < -threshold
        messages = (
            [f"Throughput {avg1:.4g} -> {avg2:.4g} tok/s ({change:+.2%})"]
            if failed or change > threshold
            else []
        )
        return 1, failed, messages, avg1, avg2, change, len(steps), len(steps)
    except (MetricError, OSError, ValueError) as error:
        return 0, True, [str(error)], 0, 0, 0, 0, 0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dir1", type=Path)
    parser.add_argument("dir2", type=Path)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold < 1:
        parser.error("threshold must be finite, nonnegative and below one")
    try:
        reference, candidate = matching_logs(args.dir1, args.dir2)
    except ValueError as error:
        print(f"[FAIL] {error}")
        return 1
    passed = 0
    for name in sorted(reference):
        _, failed, messages, *_ = compare_files(
            reference[name], candidate[name], args.threshold
        )
        passed += not failed
        if failed or messages or args.verbose:
            print(f"[{'FAIL' if failed else 'PASS'}] {name}")
            for message in messages:
                print(f"  {message}")
    print(f"Summary: {passed}/{len(reference)} test cases passed")
    return int(passed != len(reference))


if __name__ == "__main__":
    sys.exit(main())

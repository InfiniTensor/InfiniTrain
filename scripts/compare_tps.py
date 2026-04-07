#!/usr/bin/env python3
# Usage:
# python tools/compare_tps.py \
#   /path/to/baseline/logs \
#   /path/to/test/logs \
#   --threshold 0.20

import re
import sys
from pathlib import Path
from argparse import ArgumentParser
from compare_utils import collect_log_files, exit_if_duplicate_logs

def parse_log(file_path):
    """Extract step -> tok/s mapping from log file."""
    pattern = re.compile(r'step\s+(\d+)/\d+.*?\|\s+(\d+)\s+tok/s')
    tps_values = {}
    with open(file_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                tps_values[int(match.group(1))] = float(match.group(2))
    return tps_values

def compare_files(file1, file2, threshold):
    """Compare tok/s values from two log files, excluding first step."""
    tps1 = parse_log(file1)
    tps2 = parse_log(file2)

    # Remove step 1
    tps1 = {k: v for k, v in tps1.items() if k > 1}
    tps2 = {k: v for k, v in tps2.items() if k > 1}

    if not tps1 or not tps2:
        return 0, True, ["  No valid steps found (after excluding step 1)"], 0, 0, 0, 0, 0

    # Calculate averages
    avg1 = sum(tps1.values()) / len(tps1)
    avg2 = sum(tps2.values()) / len(tps2)

    # Calculate signed relative change of test vs baseline: positive means test faster, negative means test slower
    signed_change = (avg2 - avg1) / avg1 if avg1 > 0 else 0

    messages = []
    failed = False
    if abs(signed_change) > threshold:
        sign = "+" if signed_change >= 0 else ""
        if signed_change < 0:
            # test slower than baseline -> failure
            label = "✗ SLOWER"
            failed = True
        else:
            # test faster than baseline -> pass but notify
            label = "↑ FASTER"
        messages.append(f"  Average tok/s: {avg1:.2f} (baseline) vs {avg2:.2f} (test) {label} ({sign}{signed_change*100:.1f}%, threshold: ±{threshold*100:.0f}%)")
        messages.append(f"  Steps compared: {len(tps1)} vs {len(tps2)} (excluding step 1)")

    return 1, failed, messages, avg1, avg2, signed_change, len(tps1), len(tps2)

def main():
    parser = ArgumentParser(description='Compare tok/s between two log directories')
    parser.add_argument('dir1', type=Path, help='Baseline log directory')
    parser.add_argument('dir2', type=Path, help='Test log directory')
    parser.add_argument('--threshold', type=float, default=0.20, help='Relative error threshold (default: 0.20 = 20%%)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output for all files, including passed ones')
    args = parser.parse_args()

    print(f"Baseline: {args.dir1.resolve()}")
    print(f"Test:     {args.dir2.resolve()}")
    print()

    files1, duplicates1 = collect_log_files(args.dir1)
    files2, duplicates2 = collect_log_files(args.dir2)
    exit_if_duplicate_logs(args.dir1, duplicates1)
    exit_if_duplicate_logs(args.dir2, duplicates2)

    only_in_1 = set(files1.keys()) - set(files2.keys())
    only_in_2 = set(files2.keys()) - set(files1.keys())
    common = set(files1.keys()) & set(files2.keys())

    if only_in_1:
        print(f"Files only in baseline: {', '.join(sorted(only_in_1))}")
    if only_in_2:
        print(f"Files only in test: {', '.join(sorted(only_in_2))}")
    if only_in_1 or only_in_2:
        print()

    improvements = []  # (name, avg1, avg2, signed_change)
    regressions = []
    normal = []

    for name in sorted(common):
        _, failed, messages, avg1, avg2, signed_change, steps1, steps2 = compare_files(files1[name], files2[name], args.threshold)
        if failed:
            regressions.append((name, avg1, avg2, signed_change, steps1, steps2))
        elif messages:
            improvements.append((name, avg1, avg2, signed_change, steps1, steps2))
        else:
            normal.append((name, avg1, avg2, signed_change, steps1, steps2))

    pct = f"{args.threshold*100:.0f}%"

    if improvements:
        print(f"{'=' * 50}")
        print(f"Large improvements (>{pct})")
        print(f"{'=' * 50}")
        for name, avg1, avg2, signed_change, steps1, steps2 in improvements:
            sign = "+" if signed_change >= 0 else ""
            print(f"[PASS] {name}")
            print(f"  average tok/s: {avg1:.2f} vs {avg2:.2f}  ({sign}{signed_change*100:.1f}%)")
            print(f"  effective steps: {steps1} vs {steps2} (step > 1)")
            print()

    if regressions:
        print(f"{'=' * 50}")
        print(f"FAILED regressions (>{pct})")
        print(f"{'=' * 50}")
        for name, avg1, avg2, signed_change, steps1, steps2 in regressions:
            sign = "+" if signed_change >= 0 else ""
            print(f"[FAIL] {name}")
            print(f"  average tok/s: {avg1:.2f} vs {avg2:.2f}  ({sign}{signed_change*100:.1f}%)")
            print(f"  effective steps: {steps1} vs {steps2} (step > 1)")
            print()

    if args.verbose and normal:
        print(f"{'=' * 50}")
        print(f"Within threshold (<={pct})")
        print(f"{'=' * 50}")
        for name, avg1, avg2, signed_change, steps1, steps2 in normal:
            sign = "+" if signed_change >= 0 else ""
            print(f"[PASS] {name}")
            print(f"  tok/s: {avg1:.2f} vs {avg2:.2f}  ({sign}{signed_change*100:.1f}%)")
            print()

    total = len(improvements) + len(regressions) + len(normal)
    passed = len(improvements) + len(normal)
    print(f"{'=' * 50}")
    print(f"Summary: {passed}/{total} test cases passed")
    print(f"failed regressions : {len(regressions)}")
    print(f"large improvements : {len(improvements)}")
    print(f"within threshold   : {len(normal)}")
    print(f"total cases        : {total}")
    print(f"{'=' * 50}")

    sys.exit(1 if regressions else 0)

if __name__ == '__main__':
    main()

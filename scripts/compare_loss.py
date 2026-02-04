#!/usr/bin/env python3
# Usage:
# python tools/compare_loss.py \
#   /data/shared/InfiniTrain-dev/logs/202511_a800/20260105/feature/add_1F1B_f2a383a/logs \
#   /data/shared/InfiniTrain-dev/logs/202511_a800/20251223/feature/tp-pp-split-stream/logs \
#   --threshold-fp32 1e-5 --threshold-bf16 1e-2

import re
import sys
from pathlib import Path
from argparse import ArgumentParser

def get_dtype_from_filename(filename):
    """Determine dtype from filename. Returns 'bfloat16' or 'fp32'."""
    return 'bfloat16' if '_bfloat16' in filename else 'fp32'

def parse_log(file_path):
    """Extract step -> loss mapping from log file."""
    pattern = re.compile(r'step\s+(\d+)/\d+\s+\|\s+train loss\s+([\d.]+)')
    losses = {}
    with open(file_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                losses[int(match.group(1))] = float(match.group(2))
    return losses

def compare_files(file1, file2, threshold):
    """Compare loss values from two log files."""
    losses1 = parse_log(file1)
    losses2 = parse_log(file2)

    all_steps = sorted(set(losses1.keys()) | set(losses2.keys()))
    mismatches = []

    for step in all_steps:
        if step not in losses1:
            mismatches.append(f"  Step {step}: missing in {file1.name}")
        elif step not in losses2:
            mismatches.append(f"  Step {step}: missing in {file2.name}")
        else:
            loss1, loss2 = losses1[step], losses2[step]
            diff = abs(loss1 - loss2)
            if diff > threshold:
                rel = diff / max(abs(loss1), abs(loss2)) * 100 if max(abs(loss1), abs(loss2)) > 0 else 0
                mismatches.append(f"  Step {step}: {loss1:.6f} vs {loss2:.6f} ✗ (diff: {diff:.2e}, {rel:.4f}%)")

    return len(all_steps), len(mismatches), mismatches

def main():
    parser = ArgumentParser(description='Compare training loss between two log directories')
    parser.add_argument('dir1', type=Path, help='First log directory')
    parser.add_argument('dir2', type=Path, help='Second log directory')
    parser.add_argument('--threshold', type=float, help='Loss difference threshold (deprecated, use --threshold-fp32 and --threshold-bf16)')
    parser.add_argument('--threshold-fp32', type=float, default=1e-5, help='Loss difference threshold for fp32 (default: 1e-5)')
    parser.add_argument('--threshold-bf16', type=float, default=1e-2, help='Loss difference threshold for bfloat16 (default: 1e-2)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output for all files, including passed ones')
    args = parser.parse_args()

    # Support legacy --threshold argument
    if args.threshold is not None:
        args.threshold_fp32 = args.threshold
        args.threshold_bf16 = args.threshold

    files1 = {f.name: f for f in args.dir1.glob('*.log') if not f.name.startswith('build')}
    files2 = {f.name: f for f in args.dir2.glob('*.log') if not f.name.startswith('build')}

    only_in_1 = set(files1.keys()) - set(files2.keys())
    only_in_2 = set(files2.keys()) - set(files1.keys())
    common = set(files1.keys()) & set(files2.keys())

    if only_in_1:
        print(f"Files only in {args.dir1.resolve()}: {', '.join(sorted(only_in_1))}")
    if only_in_2:
        print(f"Files only in {args.dir2.resolve()}: {', '.join(sorted(only_in_2))}")
    if only_in_1 or only_in_2:
        print()

    total_mismatches = 0
    fp32_total = 0
    fp32_passed = 0
    bf16_total = 0
    bf16_passed = 0

    for name in sorted(common):
        dtype = get_dtype_from_filename(name)
        threshold = args.threshold_bf16 if dtype == 'bfloat16' else args.threshold_fp32

        if dtype == 'bfloat16':
            bf16_total += 1
        else:
            fp32_total += 1

        total_steps, num_mismatches, mismatches = compare_files(files1[name], files2[name], threshold)

        if mismatches:
            print(f"Comparing {name} ({dtype}, threshold: {threshold:.0e}):")
            for msg in mismatches:
                print(msg)
            total_mismatches += num_mismatches
        else:
            if dtype == 'bfloat16':
                bf16_passed += 1
            else:
                fp32_passed += 1

        # Only print details when there are mismatches or verbose mode
        if mismatches or args.verbose:
            if mismatches:
                matched = total_steps - num_mismatches
                print(f"  Summary: {matched}/{total_steps} steps matched")
            print()

    print("=" * 50)
    print(f"Overall Summary:")
    print(f"  fp32:    {fp32_passed}/{fp32_total} test cases passed (threshold: {args.threshold_fp32:.0e})")
    print(f"  bfloat16: {bf16_passed}/{bf16_total} test cases passed (threshold: {args.threshold_bf16:.0e})")
    print(f"  Total:   {fp32_passed + bf16_passed}/{fp32_total + bf16_total} test cases passed")
    print("=" * 50)

    sys.exit(1 if total_mismatches > 0 else 0)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Usage:
# python tools/compare_loss.py \
#   /data/shared/InfiniTrain-dev/logs/202511_a800/20260105/feature/add_1F1B_f2a383a/logs \
#   /data/shared/InfiniTrain-dev/logs/202511_a800/20251223/feature/tp-pp-split-stream/logs \
#   --threshold-fp32 1e-5 --threshold-bf16 1e-2
#
# With plotting:
# python scripts/compare_loss.py dir1 dir2 --plot

import re
import sys
from pathlib import Path
from argparse import ArgumentParser

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

    return len(all_steps), len(mismatches), mismatches, losses1, losses2


def plot_loss_comparison(dir1_name, dir2_name, losses1, losses2, output_path):
    """Plot loss curves comparison between two log files."""
    if not HAS_MATPLOTLIB:
        print("  WARNING: matplotlib not installed, skipping plot")
        return

    steps1 = sorted(losses1.keys())
    steps2 = sorted(losses2.keys())

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(steps1, [losses1[s] for s in steps1], 'b-o', label=f'{dir1_name}', markersize=4)
    ax1.plot(steps2, [losses2[s] for s in steps2], 'r--s', label=f'{dir2_name}', markersize=4)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Absolute difference
    ax2 = axes[1]
    common_steps = sorted(set(steps1) & set(steps2))
    diffs = [abs(losses1[s] - losses2[s]) for s in common_steps]
    ax2.plot(common_steps, diffs, 'g-^', markersize=4)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Absolute Difference')
    ax2.set_title('Loss Difference (|Loss1 - Loss2|)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved to: {output_path}")

def main():
    parser = ArgumentParser(description='Compare training loss between two log directories')
    parser.add_argument('dir1', type=Path, help='First log directory')
    parser.add_argument('dir2', type=Path, help='Second log directory')
    parser.add_argument('--threshold', type=float, help='Loss difference threshold (deprecated, use --threshold-fp32 and --threshold-bf16)')
    parser.add_argument('--threshold-fp32', type=float, default=1e-5, help='Loss difference threshold for fp32 (default: 1e-5)')
    parser.add_argument('--threshold-bf16', type=float, default=1e-2, help='Loss difference threshold for bfloat16 (default: 1e-2)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output for all files, including passed ones')
    parser.add_argument('--plot', action='store_true', help='Generate loss curve comparison plots')
    parser.add_argument('--plot-dir', type=Path, default=None, help='Output directory for plots (default: dir2)')
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

        total_steps, num_mismatches, mismatches, losses1, losses2 = compare_files(files1[name], files2[name], threshold)

        # Generate plot if requested
        if args.plot:
            plot_dir = args.plot_dir or args.dir2
            plot_path = plot_dir / f"{name.replace('.log', '')}_loss_comparison.png"
            plot_loss_comparison(args.dir1.name, args.dir2.name, losses1, losses2, plot_path)

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

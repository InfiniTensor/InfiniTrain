#!/usr/bin/env python3
"""
Extract training statistics from log files and display them in a table.

This script parses log files from training runs and extracts statistics such as
loss, time, tokens per second, and memory usage. It can compare
results from different configurations (e.g., flash vs noflash attention)
and calculate speedup ratios.

Usage:
    python extract_training_stats.py <log_dir1> <log_dir2> [options]

Arguments:
    log_dir1: First log directory (typically contains noflash results)
    log_dir2: Second log directory (typically contains flash results)

Options:
    --threshold-fp32: Loss difference threshold for fp32 (default: 1e-5)
    --threshold-bf16: Loss difference threshold for bfloat16 (default: 1e-2)
    --markdown: Output tables in markdown format
    --output: Specify output file path (for markdown tables)
    --speedup: Output speedup ratio table (noflash_time / flash_time)

Examples:
    # Display training statistics in plain text
    python extract_training_stats.py logs compare_logs

    # Display training statistics in markdown format
    python extract_training_stats.py logs compare_logs --markdown

    # Display training statistics in markdown and save to file
    python extract_training_stats.py logs compare_logs --markdown --output results.md

    # Display speedup ratio table
    python extract_training_stats.py logs compare_logs --speedup

    # Display speedup ratio table in markdown and save to file
    python extract_training_stats.py logs compare_logs --speedup --markdown --output results.md

Output:
    The script outputs three types of information:
    1. Training Statistics Table: Detailed statistics for each configuration
    2. Comparison between flash and noflash: Comparison of loss, time, and memory
    3. Speedup Ratio Table (if --speedup is specified): Speedup ratios for each configuration
"""

import re
import sys
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict


def get_dtype_from_filename(filename):
    """
    Determine dtype from filename.
    
    Args:
        filename: Name of the log file
        
    Returns:
        'bfloat16' or 'fp32'
    """
    return 'bfloat16' if '_bfloat16' in filename else 'fp32'


def get_flash_from_filename(filename, dir_path):
    """
    Determine flash mode from filename and directory.
    
    Args:
        filename: Name of the log file
        dir_path: Path to the directory containing the log file
        
    Returns:
        'flash' or 'noflash'
    """
    # First check if filename contains flash/noflash suffix
    if '_flash' in filename:
        return 'flash'
    elif '_noflash' in filename:
        return 'noflash'
    # Otherwise, determine from directory name
    dir_name = str(dir_path).lower()
    if 'compare' in dir_name:
        return 'flash'
    else:
        return 'noflash'


def get_disopt_from_filename(filename):
    """Determine distributed optimizer from filename. Returns 'disopt' or 'nodisopt'."""
    return 'disopt' if '_distopt' in filename else 'nodisopt'


def get_model_from_filename(filename):
    """Determine model name from filename. Returns 'gpt2' or 'llama3'."""
    if 'gpt2' in filename:
        return 'gpt2'
    elif 'llama3' in filename:
        return 'llama3'
    else:
        return 'unknown'


def parse_command_line(line):
    """Extract configuration from command line."""
    dp_match = re.search(r'--nthread_per_process=(\d+)', line)
    dp = int(dp_match.group(1)) if dp_match else 1
    
    tp_match = re.search(r'--tensor_parallel=(\d+)', line)
    tp = int(tp_match.group(1)) if tp_match else 1
    
    sp = 1
    if '--sequence_parallel' in line:
        sp = 1  # SP is enabled, but value is not explicitly set
    
    pp_match = re.search(r'--pipeline_parallel=(\d+)', line)
    pp = int(pp_match.group(1)) if pp_match else 1
    
    disopt = '--use_distributed_optimizer' in line
    
    return {
        'dp': dp,
        'tp': tp,
        'sp': sp,
        'pp': pp,
        'disopt': disopt
    }


def parse_log_file(file_path):
    """
    Extract statistics from log file.
    
    Args:
        file_path: Path to the log file
        
    Returns:
        Dictionary containing lists of losses, times, tokens per second,
        peak memory usage, and configuration
    """
    stats = {
        'losses': [],
        'times': [],
        'toks_per_sec': [],
        'peak_used': [],
        'peak_reserved': [],
        'config': None
    }
    
    with open(file_path, 'r') as f:
        for line in f:
            # Parse command line
            if line.startswith('[COMMAND]'):
                stats['config'] = parse_command_line(line)
                continue
            
            # Parse step information
            match = re.search(r'step\s+(\d+)/\d+\s+\|\s+train loss\s+([\d.]+)\s+\|\s+lr\s+[\d.e+-]+\s+\|\s+\(([\d.]+)\s+ms\s+\|\s+(\d+)\s+tok/s.*peak used:\s+(\d+)\s+MB.*peak reserved:\s+(\d+)\s+MB.*DP=(\d+),\s+TP=(\d+),\s+SP=(\d+),\s+PP=(\d+)', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                time_ms = float(match.group(3))
                tok_per_sec = int(match.group(4))
                peak_used = int(match.group(5))
                peak_reserved = int(match.group(6))
                dp = int(match.group(7))
                tp = int(match.group(8))
                sp = int(match.group(9))
                pp = int(match.group(10))
                
                stats['losses'].append((step, loss))
                stats['times'].append((step, time_ms))
                stats['toks_per_sec'].append((step, tok_per_sec))
                stats['peak_used'].append((step, peak_used))
                stats['peak_reserved'].append((step, peak_reserved))
    
    return stats


def calculate_averages(stats, exclude_first_step=True):
    """
    Calculate average values, optionally excluding first step.
    
    Args:
        stats: Dictionary containing lists of losses, times, tokens per second,
                and peak memory usage
        exclude_first_step: If True, exclude the first step from calculations
                             (default: True, as first step may have overhead)
                             
    Returns:
        Dictionary containing average values for loss, time, tokens per second,
        and peak memory usage
    """
    losses = stats['losses']
    times = stats['times']
    toks_per_sec = stats['toks_per_sec']
    peak_used = stats['peak_used']
    peak_reserved = stats['peak_reserved']
    
    start_idx = 1 if exclude_first_step else 0
    
    avg_loss = sum(loss for _, loss in losses[start_idx:]) / len(losses[start_idx:]) if len(losses[start_idx:]) > 0 else 0
    avg_time = sum(time for _, time in times[start_idx:]) / len(times[start_idx:]) if len(times[start_idx:]) > 0 else 0
    avg_tok_per_sec = sum(tok for _, tok in toks_per_sec[start_idx:]) / len(toks_per_sec[start_idx:]) if len(toks_per_sec[start_idx:]) > 0 else 0
    avg_peak_used = sum(mem for _, mem in peak_used[start_idx:]) / len(peak_used[start_idx:]) if len(peak_used[start_idx:]) > 0 else 0
    avg_peak_reserved = sum(mem for _, mem in peak_reserved[start_idx:]) / len(peak_reserved[start_idx:]) if len(peak_reserved[start_idx:]) > 0 else 0
    
    return {
        'avg_loss': avg_loss,
        'avg_time_ms': avg_time,
        'avg_tok_per_sec': avg_tok_per_sec,
        'avg_peak_used': avg_peak_used,
        'avg_peak_reserved': avg_peak_reserved
    }


def main():
    """
    Main function to extract and display training statistics.
    
    This function:
    1. Parses command line arguments
    2. Reads log files from two directories
    3. Extracts statistics from each log file
    4. Groups results by configuration
    5. Displays tables and comparisons
    6. Optionally displays speedup ratios
    """
    parser = ArgumentParser(description='Extract training statistics from log files')
    parser.add_argument('dir1', type=Path, help='First log directory')
    parser.add_argument('dir2', type=Path, help='Second log directory')
    parser.add_argument('--threshold-fp32', type=float, default=1e-5, help='Loss difference threshold for fp32 (default: 1e-5)')
    parser.add_argument('--threshold-bf16', type=float, default=1e-2, help='Loss difference threshold for bfloat16 (default: 1e-2)')
    parser.add_argument('--markdown', action='store_true', help='Output as markdown table')
    parser.add_argument('--output', type=str, default='', help='Output file path (optional)')
    parser.add_argument('--speedup', action='store_true', help='Output speedup ratio table')
    args = parser.parse_args()
    
    # Get all log files from both directories
    files1 = [(f, args.dir1) for f in args.dir1.glob('*.log') if not f.name.startswith('build') and 'comparison' not in f.name.lower()]
    files2 = [(f, args.dir2) for f in args.dir2.glob('*.log') if not f.name.startswith('build') and 'comparison' not in f.name.lower()]
    
    # Combine files from both directories
    all_files = files1 + files2
    
    # Parse all files and collect statistics
    results = []
    for file_path, dir_path in sorted(all_files, key=lambda x: (x[1], x[0].name)):
        filename = file_path.name
        model = get_model_from_filename(filename)
        dtype = get_dtype_from_filename(filename)
        flash = get_flash_from_filename(filename, dir_path)
        disopt = get_disopt_from_filename(filename)
        
        # Extract test ID from filename (e.g., gpt2_3_bfloat16_distopt_profile.log -> 3)
        test_id_match = re.search(r'_(\d+)_', filename)
        test_id = test_id_match.group(1) if test_id_match else ''
        
        # Parse log file
        stats = parse_log_file(file_path)
        averages = calculate_averages(stats, exclude_first_step=True)
        
        # Get configuration
        config = stats['config'] if stats['config'] else {}
        dp = config.get('dp', 1)
        tp = config.get('tp', 1)
        sp = config.get('sp', 1)
        pp = config.get('pp', 1)
        
        # Create row identifier (include test ID to distinguish different test configurations)
        row_id = f"{model}_{dtype}_{flash}_{disopt}_{test_id}"
        
        results.append({
            'row_id': row_id,
            'model': model,
            'dtype': dtype,
            'flash': flash,
            'disopt': disopt,
            'test_id': test_id,
            'avg_loss': averages['avg_loss'],
            'avg_time_ms': averages['avg_time_ms'],
            'avg_tok_per_sec': averages['avg_tok_per_sec'],
            'avg_peak_used': averages['avg_peak_used'],
            'avg_peak_reserved': averages['avg_peak_reserved'],
            'dp': dp,
            'tp': tp,
            'sp': sp,
            'pp': pp
        })
    
    # Sort results by row_id
    results.sort(key=lambda x: x['row_id'])
    
    # Group by model, dtype, and disopt (excluding flash)
    grouped_by_config = defaultdict(list)
    for result in results:
        key = f"{result['model']}_{result['dtype']}_{result['disopt']}"
        grouped_by_config[key].append(result)
    
    # Prepare output
    if args.markdown:
        # Output as markdown table
        markdown_lines = []
        
        # Add header
        markdown_lines.append("| Row ID | Avg Loss | Avg Time (ms) | Avg Tok/s | Peak Used (MB) | Peak Reserved (MB) | DP | TP | SP | PP |")
        markdown_lines.append("|---------|----------|---------------|-----------|-----------------|-------------------|----|----|----|----|")
        
        # Print rows (flash and noflash interleaved)
        for key in sorted(grouped_by_config.keys()):
            group = grouped_by_config[key]
            if len(group) >= 2:
                # Separate flash and noflash results
                flash_results = [r for r in group if r['flash'] == 'flash']
                noflash_results = [r for r in group if r['flash'] == 'noflash']
                
                # Interleave flash and noflash results
                for i in range(max(len(flash_results), len(noflash_results))):
                    if i < len(flash_results):
                        result = flash_results[i]
                        row = f"| {result['row_id']:<40} | {result['avg_loss']:<15.6f} | {result['avg_time_ms']:<15.2f} | {result['avg_tok_per_sec']:<15.2f} | {result['avg_peak_used']:<15.2f} | {result['avg_peak_reserved']:<15.2f} | {result['dp']:<5} | {result['tp']:<5} | {result['sp']:<5} | {result['pp']:<5} |"
                        markdown_lines.append(row)
                    if i < len(noflash_results):
                        result = noflash_results[i]
                        row = f"| {result['row_id']:<40} | {result['avg_loss']:<15.6f} | {result['avg_time_ms']:<15.2f} | {result['avg_tok_per_sec']:<15.2f} | {result['avg_peak_used']:<15.2f} | {result['avg_peak_reserved']:<15.2f} | {result['dp']:<5} | {result['tp']:<5} | {result['sp']:<5} | {result['pp']:<5} |"
                        markdown_lines.append(row)
            else:
                # Only one result, print it
                for result in group:
                    row = f"| {result['row_id']:<40} | {result['avg_loss']:<15.6f} {result['avg_time_ms']:<15.2f} {result['avg_tok_per_sec']:<15.2f} {result['avg_peak_used']:<15.2f} | {result['avg_peak_reserved']:<15.2f} | {result['dp']:<5} | {result['tp']:<5} | {result['sp']:<5} | {result['pp']:<5} |"
                    markdown_lines.append(row)
        
        # Print markdown table
        markdown_table = "\n".join(markdown_lines)
        print(markdown_table)
        
        # Write to file if output path is specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(markdown_table)
            print(f"Markdown table saved to: {args.output}")
    else:
        # Print plain text table
        print("\n" + "=" * 150)
        print("Training Statistics Table (excluding step 1)")
        print("=" * 150)
        
        # Print header
        header = f"{'Row ID':<40} {'Avg Loss':<15} {'Avg Time (ms)':<15} {'Avg Tok/s':<15} {'Peak Used (MB)':<15} {'Peak Reserved (MB)':<15} {'DP':<5} {'TP':<5} {'SP':<5} {'PP':<5}"
        print(header)
        print("-" * 150)
        
        # Print rows (flash and noflash interleaved)
        for key in sorted(grouped_by_config.keys()):
            group = grouped_by_config[key]
            if len(group) >= 2:
                # Separate flash and noflash results
                flash_results = [r for r in group if r['flash'] == 'flash']
                noflash_results = [r for r in group if r['flash'] == 'noflash']
                
                # Interleave flash and noflash results
                for i in range(max(len(flash_results), len(noflash_results))):
                    if i < len(flash_results):
                        result = flash_results[i]
                        row = f"{result['row_id']:<40} {result['avg_loss']:<15.6f} {result['avg_time_ms']:<15.2f} {result['avg_tok_per_sec']:<15.2f} {result['avg_peak_used']:<15.2f} {result['avg_peak_reserved']:<15.2f} {result['dp']:<5} {result['tp']:<5} {result['sp']:<5} {result['pp']:<5}"
                        print(row)
                    if i < len(noflash_results):
                        result = noflash_results[i]
                        row = f"{result['row_id']:<40} {result['avg_loss']:<15.6f} {result['avg_time_ms']:<15.2f} {result['avg_tok_per_sec']:<15.2f} {result['avg_peak_used']:<15.2f} {result['avg_peak_reserved']:<15.2f} {result['dp']:<5} {result['tp']:<5} {result['sp']:<5} {result['pp']:<5}"
                        print(row)
        else:
            # Only one result, print it
            for result in group:
                row = f"{result['row_id']:<40} {result['avg_loss']:<15.6f} {result['avg_time_ms']:<15.2f} {result['avg_tok_per_sec']:<15.2f} {result['avg_peak_used']:<15.2f} {result['avg_peak_reserved']:<15.2f} {result['dp']:<5} {result['tp']:<5} {result['sp']:<5} {result['pp']:<5}"
                print(row)
    
    print("=" * 150)
    
    # Print comparison between flash and noflash
    print("\nComparison between flash and noflash:")
    print("=" * 150)
    
    for key, group in sorted(grouped_by_config.items()):
        # Separate flash and noflash results
        flash_results = [r for r in group if r['flash'] == 'flash']
        noflash_results = [r for r in group if r['flash'] == 'noflash']
        
        if flash_results and noflash_results:
            # Calculate averages for each group
            avg_flash_loss = sum(r['avg_loss'] for r in flash_results) / len(flash_results)
            avg_flash_time = sum(r['avg_time_ms'] for r in flash_results) / len(flash_results)
            avg_flash_tok = sum(r['avg_tok_per_sec'] for r in flash_results) / len(flash_results)
            avg_flash_peak_used = sum(r['avg_peak_used'] for r in flash_results) / len(flash_results)
            avg_flash_peak_reserved = sum(r['avg_peak_reserved'] for r in flash_results) / len(flash_results)
            
            avg_noflash_loss = sum(r['avg_loss'] for r in noflash_results) / len(noflash_results)
            avg_noflash_time = sum(r['avg_time_ms'] for r in noflash_results) / len(noflash_results)
            avg_noflash_tok = sum(r['avg_tok_per_sec'] for r in noflash_results) / len(noflash_results)
            avg_noflash_peak_used = sum(r['avg_peak_used'] for r in noflash_results) / len(noflash_results)
            avg_noflash_peak_reserved = sum(r['avg_peak_reserved'] for r in noflash_results) / len(noflash_results)
            
            # Calculate differences
            loss_diff = avg_flash_loss - avg_noflash_loss
            time_diff = avg_flash_time - avg_noflash_time
            tok_diff = avg_flash_tok - avg_noflash_tok
            
            loss_pct = (loss_diff / avg_noflash_loss * 100) if avg_noflash_loss > 0 else 0
            time_pct = (time_diff / avg_noflash_time * 100) if avg_noflash_time > 0 else 0
            tok_pct = (tok_diff / avg_noflash_tok * 100) if avg_noflash_tok > 0 else 0
            
            print(f"\n{key}:")
            print(f"  Loss: {avg_flash_loss:.6f} vs {avg_noflash_loss:.6f} (diff: {loss_diff:+.6f}, {loss_pct:+.2f}%)")
            print(f"  Time: {avg_flash_time:.2f} vs {avg_noflash_time:.2f} ms (diff: {time_diff:+.2f}, {time_pct:+.2f}%)")
            print(f"  Tok/s: {avg_flash_tok:.2f} vs {avg_noflash_tok:.2f} (diff: {tok_diff:+.2f}, {tok_pct:+.2f}%)")
            
            # Calculate memory differences
            mem_used_diff = avg_flash_peak_used - avg_noflash_peak_used
            mem_reserved_diff = avg_flash_peak_reserved - avg_noflash_peak_reserved
            
            mem_used_pct = (mem_used_diff / avg_noflash_peak_used * 100) if avg_noflash_peak_used > 0 else 0
            mem_reserved_pct = (mem_reserved_diff / avg_noflash_peak_reserved * 100) if avg_noflash_peak_reserved > 0 else 0
            
            print(f"  Peak Used: {avg_flash_peak_used:.2f} vs {avg_noflash_peak_used:.2f} MB (diff: {mem_used_diff:+.2f}, {mem_used_pct:+.2f}%)")
            print(f"  Peak Reserved: {avg_flash_peak_reserved:.2f} vs {avg_noflash_peak_reserved:.2f} MB (diff: {mem_reserved_diff:+.2f}, {mem_reserved_pct:+.2f}%)")
            
            # Calculate memory differences
            mem_used_diff = avg_flash_peak_used - avg_noflash_peak_used
            mem_reserved_diff = avg_flash_peak_reserved - avg_noflash_peak_reserved
            mem_used_pct = (mem_used_diff / avg_noflash_peak_used * 100) if avg_noflash_peak_used > 0 else 0
            mem_reserved_pct = (mem_reserved_diff / avg_noflash_peak_reserved * 100) if avg_noflash_peak_reserved > 0 else 0
    
    # Output speedup ratio table if requested
    if args.speedup:
        print("\n" + "=" * 120)
        print("Speedup Ratio Table (noflash_time / flash_time)")
        print("=" * 120)
        
        # Group by model, dtype, disopt, and test_id
        # This allows us to calculate speedup for each individual test configuration
        grouped_by_test = defaultdict(list)
        for result in results:
            key = f"{result['model']}_{result['dtype']}_{result['disopt']}_{result.get('test_id', '')}"
            grouped_by_test[key].append(result)
        
        # Calculate speedup ratios for each test configuration
        speedup_results = []
        for key, group in sorted(grouped_by_test.items()):
            # Separate flash and noflash results
            flash_results = [r for r in group if r['flash'] == 'flash']
            noflash_results = [r for r in group if r['flash'] == 'noflash']
            
            if flash_results and noflash_results:
                # Calculate average times for this test configuration
                avg_flash_time = sum(r['avg_time_ms'] for r in flash_results) / len(flash_results)
                avg_noflash_time = sum(r['avg_time_ms'] for r in noflash_results) / len(noflash_results)
                
                # Calculate speedup ratio
                # Speedup > 1.0 means flash is faster than noflash
                speedup = avg_noflash_time / avg_flash_time if avg_flash_time > 0 else 0
                
                # Extract test_id from key
                test_id = key.split('_')[-1]
                
                speedup_results.append({
                    'config': key,
                    'speedup': speedup,
                    'flash_time': avg_flash_time,
                    'noflash_time': avg_noflash_time
                })
        
        # Sort by configuration name (alphabetical order)
        speedup_results.sort(key=lambda x: x['config'])
        
        # Print speedup table
        if args.markdown:
            # Output as markdown table
            markdown_lines = []
            markdown_lines.append("| Configuration | Speedup Ratio | Flash Time (ms) | Noflash Time (ms) |")
            markdown_lines.append("|---------------|---------------|-----------------|-------------------|")
            
            for result in speedup_results:
                row = f"| {result['config']:<37} | {result['speedup']:<13.2f} | {result['flash_time']:<15.2f} | {result['noflash_time']:<17.2f} |"
                markdown_lines.append(row)
            
            # Print markdown table
            markdown_table = "\n".join(markdown_lines)
            print(markdown_table)
            
            # Write to file if output path is specified
            if args.output:
                output_path = args.output.replace('.md', '_speedup.md')
                with open(output_path, 'w') as f:
                    f.write(markdown_table)
                print(f"Speedup table saved to: {output_path}")
        else:
            # Print plain text table
            header = f"{'Configuration':<37} {'Speedup Ratio':<15} {'Flash Time (ms)':<17} {'Noflash Time (ms)':<17}"
            print(header)
            print("-" * 120)
            
            for result in speedup_results:
                row = f"{result['config']:<37} {result['speedup']:<15.2f} {result['flash_time']:<17.2f} {result['noflash_time']:<17.2f}"
                print(row)
        
        print("=" * 120)


if __name__ == '__main__':
    main()

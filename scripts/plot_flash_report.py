#!/usr/bin/env python3
"""
Generate Flash Attention test & profile report charts from log files.

Usage:
    python3 plot_flash_report.py [log_dir]
"""

import re
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "./logs"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./report_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Log parsing
# ──────────────────────────────────────────────
STEP_RE = re.compile(
    r"step\s+(\d+)/\d+\s+\|"
    r"\s+train loss\s+([\d.]+)\s+\|"
    r".*?\(\s*([\d.]+)\s*ms"
    r"\s*\|\s*([\d.]+)\s*tok/s"
    r".*?peak used:\s*([\d.]+)\s*MB"
)

def parse_log(path):
    steps, losses, latencies, toks, mems = [], [], [], [], []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                step, loss, ms, tok, mem = m.groups()
                steps.append(int(step))
                losses.append(float(loss))
                latencies.append(float(ms))
                toks.append(float(tok))
                mems.append(float(mem))
    if not steps:
        return None
    return dict(steps=steps, losses=losses, latencies=latencies, toks=toks, mems=mems)

# ──────────────────────────────────────────────
# Helper: avg over steady-state steps (skip step 1 warmup)
# ──────────────────────────────────────────────
def steady(values, skip=1):
    v = values[skip:]
    return np.mean(v) if v else float('nan')

# ──────────────────────────────────────────────
# Collect all experiments
# ──────────────────────────────────────────────
MODELS  = ["gpt2", "llama3"]
SEQ_LENS = [128, 512]
MODES   = ["no_flash", "flash"]
LABELS  = {"no_flash": "No Flash", "flash": "Flash Attn"}
COLORS  = {"no_flash": "#4C72B0", "flash": "#DD8452"}

data = {}
for model in MODELS:
    for seq in SEQ_LENS:
        for mode in MODES:
            key = f"{model}_seq{seq}_{mode}"
            path = os.path.join(LOG_DIR, f"{key}.log")
            parsed = parse_log(path)
            if parsed:
                data[key] = parsed
                print(f"  Loaded {key}: {len(parsed['steps'])} steps")
            else:
                print(f"  MISSING: {path}")

# ──────────────────────────────────────────────
# Figure 1 & 2: Loss curves  (GPT2 / LLaMA3)
# ──────────────────────────────────────────────
def plot_loss_curves(model, seq_lens, out_path):
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(6 * len(seq_lens), 5))
    if len(seq_lens) == 1:
        axes = [axes]
    fig.suptitle(f"{model.upper()} – Training Loss: Flash vs No-Flash", fontsize=14, fontweight='bold')

    for ax, seq in zip(axes, seq_lens):
        for mode in MODES:
            key = f"{model}_seq{seq}_{mode}"
            if key not in data:
                continue
            d = data[key]
            ax.plot(d['steps'], d['losses'],
                    color=COLORS[mode], label=LABELS[mode],
                    linewidth=2, marker='o', markersize=4)
        ax.set_title(f"seq_len = {seq}", fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

plot_loss_curves("gpt2",   SEQ_LENS, os.path.join(OUT_DIR, "fig1_gpt2_loss.png"))
plot_loss_curves("llama3", SEQ_LENS, os.path.join(OUT_DIR, "fig2_llama3_loss.png"))

# ──────────────────────────────────────────────
# Figure 3: Avg Latency (ms) bar chart – all combos
# ──────────────────────────────────────────────
def bar_chart(metric_key, ylabel, title, out_path, fmt="{:.1f}"):
    keys_ordered = [f"{m}_seq{s}_{mode}"
                    for m in MODELS for s in SEQ_LENS for mode in MODES]
    group_labels = [f"{m.upper()}\nseq{s}" for m in MODELS for s in SEQ_LENS]
    n_groups = len(MODELS) * len(SEQ_LENS)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2.5), 6))
    x = np.arange(n_groups)
    w = 0.35

    vals_no = []
    vals_fl = []
    for m in MODELS:
        for s in SEQ_LENS:
            k_no = f"{m}_seq{s}_no_flash"
            k_fl = f"{m}_seq{s}_flash"
            vals_no.append(steady(data[k_no][metric_key]) if k_no in data else 0)
            vals_fl.append(steady(data[k_fl][metric_key]) if k_fl in data else 0)

    bars1 = ax.bar(x - w/2, vals_no, w, label='No Flash', color=COLORS['no_flash'])
    bars2 = ax.bar(x + w/2, vals_fl, w, label='Flash Attn', color=COLORS['flash'])

    def label_bars(bars, vals):
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        fmt.format(v), ha='center', va='bottom', fontsize=9)

    label_bars(bars1, vals_no)
    label_bars(bars2, vals_fl)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

bar_chart('latencies', 'Avg Latency (ms)',
          'Avg Step Latency: Flash vs No-Flash',
          os.path.join(OUT_DIR, "fig3_avg_latency.png"))

bar_chart('toks', 'Throughput (tok/s)',
          'Throughput: Flash vs No-Flash',
          os.path.join(OUT_DIR, "fig4_throughput.png"))

bar_chart('mems', 'Peak GPU Memory Used (MB)',
          'Peak GPU Memory: Flash vs No-Flash',
          os.path.join(OUT_DIR, "fig5_memory.png"))

# ──────────────────────────────────────────────
# Figure 6: Speedup & Memory-saving ratio
# ──────────────────────────────────────────────
def ratio_chart(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Flash Attention Gain Ratio vs No-Flash (>1 = better)', fontsize=13, fontweight='bold')

    combos = [(m, s) for m in MODELS for s in SEQ_LENS]
    labels = [f"{m.upper()}\nseq{s}" for m, s in combos]
    x = np.arange(len(combos))

    speedups, mem_savings = [], []
    for m, s in combos:
        k_no = f"{m}_seq{s}_no_flash"
        k_fl = f"{m}_seq{s}_flash"
        lat_no = steady(data[k_no]['latencies']) if k_no in data else 1
        lat_fl = steady(data[k_fl]['latencies']) if k_fl in data else 1
        mem_no = steady(data[k_no]['mems']) if k_no in data else 1
        mem_fl = steady(data[k_fl]['mems']) if k_fl in data else 1
        speedups.append(lat_no / lat_fl if lat_fl > 0 else 1)
        mem_savings.append(mem_no / mem_fl if mem_fl > 0 else 1)

    for ax, vals, title, clr in [
        (axes[0], speedups, 'Speedup (No-Flash latency / Flash latency)', '#2ca02c'),
        (axes[1], mem_savings, 'Memory Reduction Ratio (No-Flash mem / Flash mem)', '#9467bd'),
    ]:
        bars = ax.bar(x, vals, color=clr, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Baseline (1.0×)')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}×", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Ratio')
        ax.set_title(title, fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.25 + 0.1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

ratio_chart(os.path.join(OUT_DIR, "fig6_speedup_memory_ratio.png"))

# ──────────────────────────────────────────────
# Print summary table
# ──────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'Config':<26} {'Loss(avg)':>10} {'Latency(ms)':>12} {'Tok/s':>10} {'Mem(MB)':>10}")
print("-"*90)
for m in MODELS:
    for s in SEQ_LENS:
        for mode in MODES:
            key = f"{m}_seq{s}_{mode}"
            if key not in data:
                print(f"{key:<26}  -- missing --")
                continue
            d = data[key]
            avg_loss = np.mean(d['losses'][1:])
            avg_lat  = steady(d['latencies'])
            avg_tok  = steady(d['toks'])
            avg_mem  = steady(d['mems'])
            print(f"{key:<26} {avg_loss:>10.4f} {avg_lat:>12.2f} {avg_tok:>10.0f} {avg_mem:>10.0f}")
    print()
print("="*90)
print(f"\nAll figures saved to: {OUT_DIR}/")

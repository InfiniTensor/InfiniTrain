#!/usr/bin/env python3
"""Generate Flash Attention large-seq report charts."""

import re, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "./logs/flash_large"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./report_figures/large_seq"
os.makedirs(OUT_DIR, exist_ok=True)

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
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                latencies.append(float(m.group(3)))
                toks.append(float(m.group(4)))
                mems.append(float(m.group(5)))
    return dict(steps=steps, losses=losses, latencies=latencies, toks=toks, mems=mems) if steps else None

def steady(values, skip=1):
    v = values[skip:]
    return np.mean(v) if len(v) > 0 else float('nan')

MODES  = ["no_flash", "flash"]
LABELS = {"no_flash": "No Flash", "flash": "Flash Attn"}
COLORS = {"no_flash": "#4C72B0", "flash": "#DD8452"}

# GPT2: only seq1024; LLaMA3: seq1024/2048/4096
EXPERIMENTS = {
    "gpt2":   [1024],
    "llama3": [1024, 2048, 4096],
}

data = {}
for model, seqs in EXPERIMENTS.items():
    for seq in seqs:
        for mode in MODES:
            key = f"{model}_seq{seq}_{mode}"
            path = os.path.join(LOG_DIR, f"{key}.log")
            parsed = parse_log(path)
            if parsed:
                data[key] = parsed
                print(f"  Loaded {key}: {len(parsed['steps'])} steps")
            else:
                print(f"  MISSING: {path}")

# ── Figure 1: Loss curves GPT2 seq1024 ──
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("GPT2 – Loss: Flash vs No-Flash (seq=1024)", fontsize=13, fontweight='bold')
for mode in MODES:
    key = f"gpt2_seq1024_{mode}"
    if key in data:
        d = data[key]
        ax.plot(d['steps'], d['losses'], color=COLORS[mode], label=LABELS[mode],
                linewidth=2, marker='o', markersize=3)
ax.set_xlabel("Step"); ax.set_ylabel("Train Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, "fig1_gpt2_seq1024_loss.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {out}")

# ── Figure 2: Loss curves LLaMA3 all seqs ──
seqs_llama3 = [1024, 2048, 4096]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("LLaMA3 – Loss: Flash vs No-Flash", fontsize=13, fontweight='bold')
for ax, seq in zip(axes, seqs_llama3):
    for mode in MODES:
        key = f"llama3_seq{seq}_{mode}"
        if key in data:
            d = data[key]
            ax.plot(d['steps'], d['losses'], color=COLORS[mode], label=LABELS[mode],
                    linewidth=2, marker='o', markersize=3)
    ax.set_title(f"seq_len = {seq}"); ax.set_xlabel("Step"); ax.set_ylabel("Train Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, "fig2_llama3_loss.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {out}")

# ── Figure 3: Latency & Speedup bar chart (all combos) ──
combos = [("gpt2", 1024), ("llama3", 1024), ("llama3", 2048), ("llama3", 4096)]
labels = ["GPT2\nseq1024", "LLaMA3\nseq1024", "LLaMA3\nseq2048", "LLaMA3\nseq4096"]
x = np.arange(len(combos))
w = 0.35

for metric, ylabel, title, fname in [
    ('latencies', 'Avg Latency (ms)', 'Avg Step Latency: Flash vs No-Flash (Large Seq)', 'fig3_latency.png'),
    ('toks',      'Throughput (tok/s)', 'Throughput: Flash vs No-Flash (Large Seq)', 'fig4_throughput.png'),
    ('mems',      'Peak GPU Memory (MB)', 'Peak GPU Memory: Flash vs No-Flash (Large Seq)', 'fig5_memory.png'),
]:
    vals_no, vals_fl = [], []
    for model, seq in combos:
        kno = f"{model}_seq{seq}_no_flash"
        kfl = f"{model}_seq{seq}_flash"
        vals_no.append(steady(data[kno][metric]) if kno in data else 0)
        vals_fl.append(steady(data[kfl][metric]) if kfl in data else 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, vals_no, w, label='No Flash', color=COLORS['no_flash'])
    b2 = ax.bar(x + w/2, vals_fl, w, label='Flash Attn', color=COLORS['flash'])
    mx = max(max(vals_no), max(vals_fl))
    for bar, v in list(zip(b1, vals_no)) + list(zip(b2, vals_fl)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + mx*0.01,
                    f"{v:.0f}", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out}")

# ── Figure 6: Speedup & Memory ratio ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Flash Attention Gain Ratio vs No-Flash – Large Seq (>1 = better)',
             fontsize=12, fontweight='bold')
speedups, mem_savings = [], []
for model, seq in combos:
    kno = f"{model}_seq{seq}_no_flash"
    kfl = f"{model}_seq{seq}_flash"
    lat_no = steady(data[kno]['latencies']) if kno in data else 1
    lat_fl = steady(data[kfl]['latencies']) if kfl in data else 1
    mem_no = steady(data[kno]['mems']) if kno in data else 1
    mem_fl = steady(data[kfl]['mems']) if kfl in data else 1
    speedups.append(lat_no / lat_fl if lat_fl > 0 else 1)
    mem_savings.append(mem_no / mem_fl if mem_fl > 0 else 1)

for ax, vals, title, clr in [
    (axes[0], speedups, 'Speedup (No-Flash / Flash latency)', '#2ca02c'),
    (axes[1], mem_savings, 'Memory Ratio (No-Flash / Flash mem)', '#9467bd'),
]:
    bars = ax.bar(x, vals, color=clr, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Baseline (1.0×)')
    ymax = max(vals) * 1.3 + 0.1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax*0.01,
                f"{v:.2f}x", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Ratio'); ax.set_title(title, fontsize=11)
    ax.legend(); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, ymax)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig6_speedup_memory_ratio.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {out}")

# ── Summary table ──
print("\n" + "="*87)
print(f"{'Config':<32} {'Loss':>8} {'Lat(ms)':>10} {'Tok/s':>8} {'Mem(MB)':>9} {'Speedup':>8} {'MemSave':>8}")
print("-"*87)
for model, seqs in [("gpt2", [1024]), ("llama3", [1024, 2048, 4096])]:
    for seq in seqs:
        kno = f"{model}_seq{seq}_no_flash"
        kfl = f"{model}_seq{seq}_flash"
        dno = data.get(kno); dfl = data.get(kfl)
        for tag, d in [("no_flash", dno), ("flash", dfl)]:
            if not d: continue
            loss = np.mean(d['losses'][1:])
            lat  = steady(d['latencies']); tok = steady(d['toks']); mem = steady(d['mems'])
            key  = f"{model}_seq{seq}_{tag}"
            if tag == "flash" and dno and dfl:
                sp = steady(dno['latencies']) / lat
                ms = (steady(dno['mems']) - mem) / steady(dno['mems']) * 100
                print(f"{key:<32} {loss:>8.4f} {lat:>10.1f} {tok:>8.0f} {mem:>9.0f} {sp:>7.2f}x {ms:>7.1f}%")
            else:
                print(f"{key:<32} {loss:>8.4f} {lat:>10.1f} {tok:>8.0f} {mem:>9.0f} {'—':>8} {'—':>8}")
    print()
print("="*87)
print(f"\nAll figures saved to: {OUT_DIR}/")

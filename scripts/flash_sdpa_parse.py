#!/usr/bin/env python3

# ------modify-start------------------------------------------
# Parse InfiniTrain training logs produced by flash_sdpa_benchmark.bash.
# Extracts:
# - avg latency (ms) excluding step1 warmup
# - avg tokens/s excluding step1 warmup
# - peak used/reserved MB (max over steps)
# - loss (step1)
# Generates a markdown report and a CSV summary.
# ---------modify-end-----------------------------------------

from __future__ import annotations

import argparse
import csv
import pathlib
import re
from dataclasses import dataclass
from typing import List, Optional


STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+train loss\s+(?P<loss>[-+\w\.]+)\s+\|\s+lr\s+(?P<lr>[-+\w\.eE]+)\s+\|\s+\((?P<ms>[0-9\.]+)\s+ms\s+\|\s+(?P<toks>[0-9\.]+)\s+tok/s\s+\|\s+peak used:\s+(?P<used>[0-9\.]+)\s+MB\s+\|\s+peak reserved:\s+(?P<reserved>[0-9\.]+)\s+MB"
)


@dataclass
class RunMetrics:
    name: str
    steps: List[int]
    losses: List[float]
    ms: List[float]
    toks: List[float]
    peak_used_mb: float
    peak_reserved_mb: float

    @property
    def loss_step1(self) -> Optional[float]:
        if not self.steps:
            return None
        try:
            idx = self.steps.index(1)
        except ValueError:
            return None
        return self.losses[idx]

    def avg_ms_excl_warmup(self) -> Optional[float]:
        pairs = [(s, v) for s, v in zip(self.steps, self.ms) if s != 1]
        if not pairs:
            return None
        return sum(v for _, v in pairs) / len(pairs)

    def avg_toks_excl_warmup(self) -> Optional[float]:
        pairs = [(s, v) for s, v in zip(self.steps, self.toks) if s != 1]
        if not pairs:
            return None
        return sum(v for _, v in pairs) / len(pairs)


def parse_log(path: pathlib.Path, name: str) -> RunMetrics:
    steps: List[int] = []
    losses: List[float] = []
    ms: List[float] = []
    toks: List[float] = []
    peak_used = 0.0
    peak_reserved = 0.0

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        s = int(m.group("step"))
        loss_str = m.group("loss")
        if loss_str.lower() == "nan":
            loss = float("nan")
        else:
            loss = float(loss_str)
        steps.append(s)
        losses.append(loss)
        ms.append(float(m.group("ms")))
        toks.append(float(m.group("toks")))
        peak_used = max(peak_used, float(m.group("used")))
        peak_reserved = max(peak_reserved, float(m.group("reserved")))

    return RunMetrics(
        name=name,
        steps=steps,
        losses=losses,
        ms=ms,
        toks=toks,
        peak_used_mb=peak_used,
        peak_reserved_mb=peak_reserved,
    )


def fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--timestamp", required=True)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--env_file", required=True)
    args = ap.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    logs_dir = out_dir / "logs"

    runs = {}
    for key in ["gpt2_baseline", "gpt2_flash", "llama3_baseline", "llama3_flash"]:
        log_path = logs_dir / f"{key}_{args.timestamp}.log"
        if not log_path.exists():
            raise SystemExit(f"Missing log: {log_path}")
        runs[key] = parse_log(log_path, key)

    # CSV summary
    csv_path = out_dir / f"summary_{args.timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "avg_ms_excl_step1",
                "avg_tok_s_excl_step1",
                "peak_used_mb",
                "peak_reserved_mb",
                "loss_step1",
            ]
        )
        for k, r in runs.items():
            w.writerow(
                [
                    k,
                    r.avg_ms_excl_warmup(),
                    r.avg_toks_excl_warmup(),
                    r.peak_used_mb,
                    r.peak_reserved_mb,
                    r.loss_step1,
                ]
            )

    def speedup(b: RunMetrics, f: RunMetrics) -> Optional[float]:
        bm = b.avg_ms_excl_warmup()
        fm = f.avg_ms_excl_warmup()
        if bm is None or fm is None or fm == 0:
            return None
        return bm / fm

    def mem_saving_ratio(b: RunMetrics, f: RunMetrics) -> Optional[float]:
        if b.peak_used_mb <= 0:
            return None
        return (b.peak_used_mb - f.peak_used_mb) / b.peak_used_mb

    # Markdown report
    report_path = out_dir / f"report_{args.timestamp}.md"
    env_text = pathlib.Path(args.env_file).read_text(encoding="utf-8", errors="ignore")

    gpt2_su = speedup(runs["gpt2_baseline"], runs["gpt2_flash"])
    llama3_su = speedup(runs["llama3_baseline"], runs["llama3_flash"])
    gpt2_mem = mem_saving_ratio(runs["gpt2_baseline"], runs["gpt2_flash"])
    llama3_mem = mem_saving_ratio(runs["llama3_baseline"], runs["llama3_flash"])

    def loss_diff(a: RunMetrics, b: RunMetrics) -> Optional[float]:
        if a.loss_step1 is None or b.loss_step1 is None:
            return None
        return abs(a.loss_step1 - b.loss_step1)

    gpt2_ld = loss_diff(runs["gpt2_baseline"], runs["gpt2_flash"])
    llama3_ld = loss_diff(runs["llama3_baseline"], runs["llama3_flash"])

    lines = []
    lines.append(f"# Flash SDPA 性能与正确性报告 ({args.timestamp})")
    lines.append("")
    lines.append("## 实验配置")
    lines.append(f"- seq_len: {args.seq_len}")
    lines.append(f"- iters: {args.iters} (统计时排除 step1 warmup)")
    lines.append("")
    lines.append("## 环境信息")
    lines.append("```text")
    lines.append(env_text.strip())
    lines.append("```")
    lines.append("")

    lines.append("## 指标汇总")
    lines.append("")
    lines.append(
        "| Model | Variant | Avg latency (ms/step) | Avg tok/s | Peak used (MB) | Peak reserved (MB) | step1 loss |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    def row(model: str, variant: str, r: RunMetrics) -> str:
        return (
            f"| {model} | {variant} | {fmt(r.avg_ms_excl_warmup())} | {fmt(r.avg_toks_excl_warmup())} | "
            f"{fmt(r.peak_used_mb, 0)} | {fmt(r.peak_reserved_mb, 0)} | {fmt(r.loss_step1, 6)} |"
        )

    lines.append(row("GPT-2", "baseline", runs["gpt2_baseline"]))
    lines.append(row("GPT-2", "flash", runs["gpt2_flash"]))
    lines.append(row("LLaMA-3", "baseline", runs["llama3_baseline"]))
    lines.append(row("LLaMA-3", "flash", runs["llama3_flash"]))

    lines.append("")
    lines.append("## 对比结论")
    lines.append("")
    lines.append(f"- GPT-2 speedup = {fmt(gpt2_su, 3)}")
    lines.append(f"- GPT-2 memory saving ratio = {fmt((gpt2_mem or 0.0) * 100.0, 2)}%")
    lines.append(f"- GPT-2 |step1 loss diff| = {fmt(gpt2_ld, 6)}")
    lines.append("")
    lines.append(f"- LLaMA-3 speedup = {fmt(llama3_su, 3)}")
    lines.append(
        f"- LLaMA-3 memory saving ratio = {fmt((llama3_mem or 0.0) * 100.0, 2)}%"
    )
    lines.append(f"- LLaMA-3 |step1 loss diff| = {fmt(llama3_ld, 6)}")
    lines.append("")
    lines.append("## 日志文件")
    lines.append("")
    for k in ["gpt2_baseline", "gpt2_flash", "llama3_baseline", "llama3_flash"]:
        lines.append(f"- {args.out_dir}/logs/{k}_{args.timestamp}.log")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()

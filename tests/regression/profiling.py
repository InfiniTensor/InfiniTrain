"""Validate per-device profiler files and summarize recorded dispatcher calls."""
import math
from pathlib import Path
import re

from execution import atomic_json, file_sha256


def check_build_mode(environment, provenance, enabled):
    cache_mode = environment.get("build_options", {}).get("PROFILE_MODE")
    on = str(cache_mode).upper() in {"1", "ON", "TRUE", "YES"}
    if enabled:
        recorded_mode = provenance.get("manifest", {}).get("environment", {}).get("build_options", {}).get("PROFILE_MODE")
        if (not on or not provenance.get("verified")
                or str(recorded_mode).upper() not in {"1", "ON", "TRUE", "YES"}):
            raise ValueError("Profiling requires verified PROFILE_MODE=ON binaries; rebuild without --no-build")
    elif on:
        raise ValueError("Normal suites require PROFILE_MODE=OFF; choose a separate build directory")


def collect(directory, binary, world_size, start_step, final_step):
    """Collect only this execution's outputs; profiler rank is a device index.

    Preserve raw files and aggregate per rank/operator, excluding the first
    observed training step. Initialization and all step records remain available.
    """
    directory = Path(directory)
    reports = {p.name.split(".report.rank", 1)[1]: p
               for p in directory.glob(binary + ".report.rank*")}
    logs = {p.name.split(".records.log.rank", 1)[1]: p
            for p in directory.glob(binary + ".records.log.rank*")}
    if len(reports) != world_size or reports.keys() != logs.keys():
        raise ValueError("Missing profiler report/records pairs: expected %d devices, got %d reports and %d records"
                         % (world_size, len(reports), len(logs)))
    expected_tags = {"Step_" + str(step - 1) for step in range(start_step, final_step + 1)}
    measured_tags = expected_tags - {"Step_" + str(start_step - 1)}
    rows, files = [], []
    for rank in sorted(reports):
        if not re.fullmatch(r"-?\d+", rank):
            raise ValueError("Invalid profiler device index: " + rank)
        report = reports[rank].read_text()
        if "Profiler Report for Rank " not in report or "Device Total(us)" not in report:
            raise ValueError("Invalid profiler report: " + reports[rank].name)
        seen, stats, tag, calls = set(), {}, None, 0
        for line in logs[rank].read_text().splitlines():
            line = line.strip()
            if line.startswith("Tag: "):
                tag = line[5:]
                continue
            if not line or line.startswith(("===", "Idx")):
                continue
            match = re.fullmatch(r"\d+\s+\S+\s+\S+\s+(.+?)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
            if match is None or tag is None:
                raise ValueError("Malformed profiler record: " + logs[rank].name)
            name, device, host, elapsed, reserved = match.groups()
            host, elapsed = int(host), int(elapsed)
            if not all(math.isfinite(v) for v in (host, elapsed, int(reserved))):
                raise ValueError("Nonfinite profiler record")
            calls += 1
            seen.add(tag)
            if tag not in measured_tags:
                continue
            key = (device, name)
            value = stats.setdefault(key, {"count": 0, "host_total_us": 0, "device_total_us": 0})
            value["count"] += 1
            value["host_total_us"] += host
            value["device_total_us"] += elapsed
        if not calls or not expected_tags <= seen:
            raise ValueError("Profiler records missing training steps: " + logs[rank].name)
        for (device, name), value in sorted(stats.items()):
            rows.append(dict(rank=int(rank), device=device, operator=name, **value,
                             host_mean_us=value["host_total_us"] / value["count"],
                             device_mean_us=value["device_total_us"] / value["count"]))
        for path in (reports[rank], logs[rank]):
            files.append({"path": path.name, "sha256": file_sha256(path)})
    result = dict(schema_version=1, profile_mode=True, device_count=world_size,
                  rank_semantics="profiler device index", warmup_steps=1,
                  measured_step_count=len(measured_tags), operators=rows, files=files,
                  timing_scope="dispatcher calls with stream synchronization; not individual CUDA kernels")
    atomic_json(directory / "profiles.json", result)
    return result

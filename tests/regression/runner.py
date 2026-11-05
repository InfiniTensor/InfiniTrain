#!/usr/bin/env python3
"""One entry point for InfiniTrain training, correctness and performance tests."""
import argparse
import datetime
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(ROOT))

import checks
from history import History
import planning
import profiling
from assets import Assets
from execution import (
    ResourcePool,
    atomic_json,
    build_project,
    command_text,
    discover_gpus,
    environment_fingerprint,
    file_sha256,
    probe,
    run_process,
    safe_case_path,
    source_metadata,
    unique_run_id,
    terminate_all,
    verify_build_manifest,
)
from reporting import write_reports
from reproduction import command_for_case


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--suite", choices=("smoke", "pr", "nightly", "perf", "release"), default="pr"
    )
    cli.add_argument(
        "--alignment",
        choices=("history", "deep"),
        default=None,
        help="Compare previous-run logs; deep additionally runs tensor and independent-reference checks",
    )
    cli.add_argument(
        "--reference-run", "--compare-to",
        type=Path,
        help="Use this successful run directory instead of the latest matching run",
    )
    cli.add_argument(
        "--history-root",
        type=Path,
        default=None,
        help="Directory searched for previous run logs",
    )
    cli.add_argument(
        "--record-history",
        action="store_true",
        help="Record a successful run as new history without comparing to an older run",
    )
    cli.add_argument("--loss-atol-fp32", type=float, default=1e-5)
    cli.add_argument("--loss-atol-bf16", type=float, default=1e-2)
    cli.add_argument(
        "--performance-threshold",
        type=float,
        default=0.20,
        help="Maximum throughput slowdown before failure (default: 0.20 = 20%%)",
    )
    cli.add_argument(
        "--model", action="append", help="Model filter; repeat for alternatives"
    )
    cli.add_argument(
        "--tag", action="append", help="Required tag; repeat for intersection"
    )
    cli.add_argument("--case", action="append", help="Exact case ID or quoted wildcard")
    cli.add_argument(
        "--changed",
        metavar="GIT_REF",
        help="Select PR impact plus fixed anchors from a merge-base diff",
    )
    cli.add_argument("--backend", default=None, help="cpu, cuda or configured provider")
    cli.add_argument("--config-root", type=Path, default=ROOT)
    cli.add_argument("--local-config", type=Path, default=REPO / "tests.local.json")
    cli.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="ARG=JSON",
        help="Explicit training-argument override",
    )
    cli.add_argument(
        "--list",
        action="store_true",
        help="List cases without building or accessing GPU/data",
    )
    cli.add_argument(
        "--dry-run", action="store_true", help="Print resolved plan without running"
    )
    cli.add_argument(
        "--explain", action="store_true", help="Include resolved arguments and coverage"
    )
    cli.add_argument(
        "--output",
        type=Path,
        help="New run directory (must not already contain a plan)",
    )
    cli.add_argument(
        "--data-root", type=Path, help="Classic LLMC asset root; see fixture manifest"
    )
    cli.add_argument("--fixture", action="append", default=[], metavar="ID=PATH")
    cli.add_argument("--build-dir", type=Path)
    cli.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Large scratch filesystem; each run gets a unique owned subdirectory",
    )
    cli.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Retain successful run checkpoints after verification",
    )
    cli.add_argument(
        "--no-build", action="store_true", help="Use an existing configured build"
    )
    cli.add_argument("--build-jobs", type=int, default=8)
    cli.add_argument("--build-timeout", type=int, default=3600)
    cli.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Concurrent independent cases; GPUs are reserved without oversubscription",
    )
    cli.add_argument(
        "--gpus", help="Comma separated physical GPU IDs (defaults to visible devices)"
    )
    cli.add_argument("--resource-timeout", type=int, default=60)
    cli.add_argument(
        "--skip-unit", action="store_true", help="Explicit partial run: omit CTest"
    )
    cli.add_argument(
        "--strict",
        action="store_true",
        help="Require full suite selection, units and all checks",
    )
    cli.add_argument("--rerun-failed", type=Path, metavar="SUMMARY_JSON")
    cli.add_argument(
        "--perf-environment",
        help="Optional machine profile recorded with performance logs",
    )
    cli.add_argument(
        "--docker-container",
        help="Execute in an existing container with this repository mounted",
    )
    return cli


def _pairs(values):
    output = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError("Expected NAME=VALUE: " + value)
        if key in output:
            raise ValueError("Duplicate override: " + key)
        output[key] = item
    return output


def _configuration(args):
    local = {}
    if args.local_config.exists():
        local = json.loads(args.local_config.read_text())
        allowed = {
            "backend",
            "data_root",
            "build_dir",
            "docker_container",
            "gpus",
            "fixtures",
            "perf_environment",
            "checkpoint_root",
            "history_root",
            "reference_run",
        }
        if not isinstance(local, dict) or set(local) - allowed:
            raise ValueError("Unknown local configuration fields")
        for field in ("data_root", "build_dir", "checkpoint_root", "history_root", "reference_run"):
            if local.get(field) and not Path(local[field]).expanduser().is_absolute():
                local[field] = str(
                    (args.local_config.resolve().parent / local[field]).resolve()
                )
        if "fixtures" in local:
            local["fixtures"] = {
                key: str(
                    (
                        args.local_config.resolve().parent / Path(value).expanduser()
                    ).resolve()
                )
                for key, value in local["fixtures"].items()
            }
    args.backend = args.backend or local.get("backend", "cuda")
    args.data_root = (
        args.data_root
        or os.environ.get("INFINITRAIN_TEST_DATA_ROOT")
        or local.get("data_root")
    )
    args.docker_container = args.docker_container or local.get("docker_container")
    args.checkpoint_root = args.checkpoint_root or local.get("checkpoint_root")
    args.alignment = args.alignment or "history"
    args.history_root = Path(args.history_root or local.get("history_root") or REPO / "artifacts" / "test-runs").resolve()
    args.reference_run = args.reference_run or local.get("reference_run")
    args.perf_environment = args.perf_environment or local.get("perf_environment")
    args.gpus = args.gpus if args.gpus is not None else local.get("gpus")
    args.build_dir_explicit = args.build_dir is not None
    args.build_dir = Path(
        args.build_dir
        or local.get("build_dir")
        or REPO / ".cache" / "regression" / ("build-" + args.backend)
    ).resolve()
    overrides = dict(local.get("fixtures", {}))
    overrides.update(_pairs(args.fixture))
    return local, overrides


def _select_build_mode(args, plan):
    args.profile_mode = bool(plan["suite_metadata"].get("profile_mode", False))
    if args.profile_mode and not args.build_dir_explicit:
        args.build_dir = args.build_dir.with_name(args.build_dir.name + "-profile")


def _reexec_container(args):
    if (
        not args.docker_container
        or os.environ.get("INFINITRAIN_REGRESSION_CONTAINER") == "1"
        or Path("/.dockerenv").exists()
    ):
        return None
    if args.list or args.dry_run:
        return None
    argv = ["docker", "exec", "-e", "INFINITRAIN_REGRESSION_CONTAINER=1"]
    for name in (
        "INFINITRAIN_TEST_DATA_ROOT",
        "INFINITRAIN_TEST_IMAGE",
        "CUDA_VISIBLE_DEVICES",
    ):
        if name in os.environ:
            argv += ["-e", name + "=" + os.environ[name]]
    forwarded = list(sys.argv[1:])
    path_flags = {
        "--config-root",
        "--local-config",
        "--output",
        "--data-root",
        "--build-dir",
        "--reference-run",
        "--compare-to",
        "--history-root",
        "--rerun-failed",
        "--checkpoint-root",
    }
    for index, value in enumerate(forwarded):
        key, separator, inline = value.partition("=")
        if key in path_flags:
            if separator:
                forwarded[index] = key + "=" + str(Path(inline).expanduser().resolve())
            elif index + 1 < len(forwarded):
                forwarded[index + 1] = str(
                    Path(forwarded[index + 1]).expanduser().resolve()
                )
        if key == "--fixture":
            target = inline if separator else forwarded[index + 1]
            fixture_id, equals, filename = target.partition("=")
            if equals:
                normalized = (
                    fixture_id + "=" + str(Path(filename).expanduser().resolve())
                )
                if separator:
                    forwarded[index] = key + "=" + normalized
                else:
                    forwarded[index + 1] = normalized
    argv += [
        "-w",
        str(REPO),
        args.docker_container,
        "python3",
        str(Path(__file__).resolve()),
    ] + forwarded
    return subprocess.call(argv)


def _argv(case, build_dir, args):
    argv = [str(Path(build_dir) / case["binary"])]
    for key, value in sorted(args.items()):
        text = str(value).lower() if isinstance(value, bool) else str(value)
        argv.append("--{}={}".format(key, text))
    if case["nproc_per_node"] > 1:
        argv = [
            str(Path(build_dir) / "infini_run"),
            "--nproc_per_node=" + str(case["nproc_per_node"]),
        ] + argv
    return argv


def _status(check_results):
    statuses = {item["status"] for item in check_results}
    for status in (
        "ERROR",
        "FAIL",
        "NOT_COMPARABLE",
        "BLOCKED",
        "INCONCLUSIVE",
        "SKIP",
        "RECORDED",
    ):
        if status in statuses:
            return status
    return "PASS" if check_results else "ERROR"


def checkpoint_storage_budget(plan, assets, reference=False):
    """Conservative retained-checkpoint budget, including Adam master/moments."""
    total = 0
    for case in plan["cases"]:
        args = case["resolved_args"]
        if not args.get("save") or not args.get("save_interval"):
            continue
        weights = [key for key in case["fixtures"] if key.endswith(".weights")]
        if not weights:
            raise ValueError("Checkpoint storage estimate requires a weights fixture")
        size = max(Path(assets.paths[key]).stat().st_size for key in weights)
        first, last, interval = (
            case.get("start_step", 1),
            case["expected_steps"],
            args["save_interval"],
        )
        saves = last // interval - (first - 1) // interval + int(last % interval != 0)
        keep = args.get("max_checkpoint_keep", 3)
        # Pruning happens after writing the next checkpoint: reserve that
        # transient extra copy, but resumed stages only create their own steps.
        retained = min(saves, keep + 1) if keep > 0 else saves
        total += (
            size
            * 4
            * retained
            * case.get("repetitions", 1)
            * case.get("topology", {}).get("dp", 1)
        )
    return int(total * (2 if reference else 1) * 1.1)


def storage_preflight(path, required_bytes):
    directory = Path(path).expanduser().resolve()
    while not directory.exists():
        if directory == directory.parent:
            raise ValueError("Cannot identify checkpoint filesystem")
        directory = directory.parent
    free = shutil.disk_usage(directory).free
    reserve = 1024**3
    if required_bytes and free < required_bytes + reserve:
        return "Checkpoint storage needs an estimated {:.1f} GiB plus reserve, {:.1f} GiB free; set --checkpoint-root on a larger scratch filesystem".format(
            required_bytes / 1024**3, free / 1024**3
        )
    return None


class Run:
    def __init__(self, args, plan, assets, output, gpu_ids):
        self.args, self.plan, self.assets, self.output = args, plan, assets, output
        self.cases_dir = output / "cases"
        self.checkpoint_dir = (
            Path(args.checkpoint_root).expanduser().resolve() / unique_run_id()
            if args.checkpoint_root
            else output / "checkpoints"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(
            self.checkpoint_dir / "owner.json", {"run_output": str(output.resolve())}
        )
        lock_dir = os.environ.get(
            "INFINITRAIN_TEST_LOCK_DIR",
            str(Path.home() / ".cache" / "infinitrain-regression-locks"),
        )
        self.pool = ResourcePool(gpu_ids, lock_dir, args.resource_timeout)
        self.environment = environment_fingerprint(REPO, args.build_dir, args.backend)
        self.environment["selected_gpu_ids"] = gpu_ids
        self.environment["performance_profile"] = args.perf_environment
        self.source = source_metadata(REPO)
        binary_names = {case["binary"] for case in plan["cases"]}
        if any(case["nproc_per_node"] > 1 for case in plan["cases"]):
            binary_names.add("infini_run")
        self.build_provenance = verify_build_manifest(args.build_dir, binary_names)
        self.profile_mode = bool(plan.get("suite_metadata", {}).get("profile_mode", False))
        profiling.check_build_mode(self.environment, self.build_provenance, self.profile_mode)
        self.done = {}
        self.case_map = {c["id"]: c for c in plan["cases"]}
        self.selection_scope = {
            "models": sorted(getattr(args, "model", None) or []),
            "tags": sorted(getattr(args, "tag", None) or []),
            "cases": sorted(getattr(args, "case", None) or []),
            "changed": getattr(args, "changed", None),
        }
        self.history = History(
            args.history_root,
            output,
            plan["suite"],
            args.backend,
            args.reference_run,
            loss_atol_fp32=args.loss_atol_fp32,
            loss_atol_bf16=args.loss_atol_bf16,
            performance_threshold=args.performance_threshold,
            expected_case_ids=[case["id"] for case in plan["cases"]],
            selection_scope=self.selection_scope,
        )
        self.history_selection = (
            checks.result("RECORDED", ["Explicit --record-history"])
            if getattr(args, "record_history", False) else self.history.select()
        )
        if self.history_selection["status"] == "ERROR":
            raise ValueError("; ".join(self.history_selection["reasons"]))

    def needs_snapshots(self, case):
        if getattr(self.args, "alignment", "deep") == "history":
            return False
        kinds = {item["kind"] for item in case["checks"]}
        referenced = any(
            any(ch.get("reference_case") == case["id"] for ch in c["checks"])
            for c in self.plan["cases"]
        )
        return case["profile"] == "tiny" and (
            referenced
            or bool(kinds & {"equivalent", "resume", "lora_frozen", "oracle"})
        )

    def train(self, case, stage_dir, gpu_ids, *, reference=False, repeat=0):
        build_dir = self.args.build_dir
        artifact_root = self.cases_dir
        checkpoint_root = self.checkpoint_dir / (
            "reference" if reference else "candidate"
        )
        arguments = {
            k: self.assets.resolve(v, artifact_root, checkpoint_root)
            for k, v in case["resolved_args"].items()
        }
        metrics_file = stage_dir / "metrics.jsonl"
        snapshots = stage_dir / "snapshots"
        structured = self.args.alignment == "deep"
        if self.args.alignment == "history":
            arguments.pop("metrics_file", None)
            arguments.pop("regression_snapshot_dir", None)
        if structured:
            arguments["metrics_file"] = str(metrics_file)
            if self.needs_snapshots(case) and not reference:
                arguments["regression_snapshot_dir"] = str(snapshots)
        argv = _argv(case, build_dir, arguments)
        env = os.environ.copy()
        if self.args.backend == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[: case["gpus"]])
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        process = run_process(
            argv,
            cwd=stage_dir,
            log_path=stage_dir / "train.log",
            timeout=case["timeout_seconds"],
            env=env,
            headers={
                "RUN_ID": self.output.name,
                "UTC_TIME": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "GIT_COMMIT": self.source.get("commit") or "unknown",
                "GIT_DIRTY": str(self.source.get("dirty", False)).lower(),
                "SOURCE_DIFF_SHA256": self.source.get("diff_sha256", "unknown"),
                "CASE_ID": case["id"],
                "REFERENCE_RUN_STAGE": str(reference),
            },
        )
        stage = {
            "process": process,
            "reference": reference,
            "repeat": repeat,
            "snapshot_dir": str(snapshots),
        }
        if process["status"] != "PASS":
            stage.update(
                status=process["status"],
                reasons=[process.get("reason", "training failed")],
            )
            return stage
        expected = list(range(case.get("start_step", 1), case["expected_steps"] + 1))
        try:
            if self.profile_mode:
                stage["profiling"] = profiling.collect(
                    stage_dir, case["binary"],
                    case["nproc_per_node"] * case["nthread_per_process"],
                    case.get("start_step", 1), case["expected_steps"],
                )
            if structured:
                metrics = checks.load_distributed_metrics(
                    metrics_file,
                    expected_steps=expected,
                    dtype=case["dtype"],
                    world_size=case["nproc_per_node"] * case["nthread_per_process"],
                )
            else:
                metrics = checks.load_metrics(
                    stage_dir / "train.log",
                    expected_steps=expected,
                    dtype=case["dtype"],
                    required=("loss", "tokens_per_second"),
                )
            stage.update(status="PASS", reasons=[], metrics=metrics.to_dict())
            atomic_json(stage_dir / "metrics.normalized.json", metrics.to_dict())
        except (checks.MetricError, ValueError, OSError) as error:
            stage.update(status="FAIL", reasons=[str(error)])
        return stage

    def tensor_checks(self, case, kind, reference, candidate, steps, required,
                      expected_world_sizes=(None, None), check_initial=False):
        policies = case.get("resolved_tolerances", {}).get(kind, {})
        groups = {}
        for metric in required:
            options = policies.get(metric, {})
            group = groups.setdefault(tuple(sorted(options.items())), {"options": options, "metrics": []})
            group["metrics"].append(metric)
        results = []
        for index, group in enumerate(groups.values()):
            results.append(checks.compare_snapshot_trajectories(
                reference, candidate, steps=steps, dtype=case["dtype"],
                required_kinds=tuple(group["metrics"]), expected_world_sizes=expected_world_sizes,
                check_initial=check_initial and index == 0, **group["options"]))
        return checks.result(_status(results), [reason for r in results for reason in r.get("reasons", [])], groups=results)

    def evaluate(self, case, stages):
        outcomes = []
        candidate_stages = [s for s in stages if not s["reference"]]
        metrics = [checks.Metrics.from_dict(s["metrics"]) for s in candidate_stages]
        if self.args.record_history:
            outcomes.append(
                dict(
                    checks.result(
                        "RECORDED",
                        ["Comparison skipped by explicit --record-history"],
                    ),
                    kind="history",
                )
            )
        else:
            history = self.history.compare(
                case,
                stages,
                self.assets.hashes,
                self.environment,
                loss_atol_fp32=self.args.loss_atol_fp32,
                loss_atol_bf16=self.args.loss_atol_bf16,
                performance_threshold=self.args.performance_threshold,
            )
            outcomes.append(dict(history, kind="history"))
        if self.args.alignment == "history":
            # Log mode only checks complete logs against the selected history.
            return [dict(checks.check_complete(
                metrics[0], range(case.get("start_step", 1), case["expected_steps"] + 1)
            ), kind="finite_and_complete"), *outcomes]
        for spec in case["checks"]:
            kind = spec["kind"]
            if kind == "finite_and_complete":
                outcome = checks.check_complete(
                    metrics[0],
                    range(case.get("start_step", 1), case["expected_steps"] + 1),
                )
            elif kind in ("equivalent", "resume"):
                other = self.done[spec["reference_case"]]
                reference = checks.Metrics.from_dict(other["metrics"])
                selected = list(
                    range(case.get("start_step", 1), case["expected_steps"] + 1)
                )
                metric_names = [
                    k
                    for k in spec.get("metrics", ["loss"])
                    if k
                    not in (
                        "gradient",
                        "gradients",
                        "parameter",
                        "updated_parameters",
                        "optimizer",
                    )
                ]
                metric_names = [
                    "lr" if k == "learning_rate" else k for k in metric_names
                ]
                metric_checks = [checks.compare_metrics(
                    reference, metrics[0], dtype=case["dtype"], metrics=[metric], steps=selected,
                    **case.get("resolved_tolerances", {}).get(kind, {}).get(metric, {})) for metric in metric_names]
                outcome = checks.result(_status(metric_checks) if metric_checks else "PASS",
                                        [reason for check in metric_checks for reason in check.get("reasons", [])], metrics=metric_checks)
                tensor_kinds = {
                    "gradients": "gradient",
                    "updated_parameters": "parameter",
                }
                required = [
                    tensor_kinds.get(k, k)
                    for k in spec.get("metrics", [])
                    if k
                    in (
                        "gradient",
                        "gradients",
                        "parameter",
                        "updated_parameters",
                        "optimizer",
                    )
                ]
                if required:
                    reference_case = self.case_map[spec["reference_case"]]
                    snapshot_outcome = self.tensor_checks(
                        case, kind,
                        other["snapshot_dir"],
                        candidate_stages[0]["snapshot_dir"],
                        steps=selected,
                        required=required,
                        expected_world_sizes=(
                            reference_case["nproc_per_node"]
                            * reference_case["nthread_per_process"],
                            case["nproc_per_node"] * case["nthread_per_process"],
                        ),
                        check_initial=kind == "equivalent",
                    )
                    outcomes.append(dict(snapshot_outcome, kind=kind + "_tensors"))
            elif kind == "lora_frozen":
                snapshot_dir = Path(candidate_stages[0]["snapshot_dir"])
                outcome = checks.check_lora_frozen(
                    snapshot_dir / "step_0",
                    snapshot_dir / ("step_" + str(case["expected_steps"])),
                )
            elif kind == "lr_schedule":
                outcome = checks.check_lr_schedule(metrics[0], case["resolved_args"],
                    **case.get("resolved_tolerances", {}).get(kind, {}).get("lr", {}))
            elif kind == "memory_growth":
                aliases = {"warmup_steps": "warmup", "window_steps": "window"}
                outcome = checks.check_memory_growth(
                    metrics[0],
                    **{aliases.get(k, k): v for k, v in spec.items() if k != "kind"}
                )
            elif kind == "oracle":
                oracle_dir = Path(candidate_stages[0]["snapshot_dir"]).parent / "oracle"
                arguments = case["resolved_args"]
                oracle_command = [
                    sys.executable,
                    str(ROOT / "fixtures" / "reference.py"),
                    "--initial-snapshot",
                    str(
                        Path(candidate_stages[0]["snapshot_dir"])
                        / "step_0"
                        / "rank_0"
                        / "manifest.json"
                    ),
                    "--tokens",
                    self.assets.resolve(arguments["input_bin"], self.cases_dir),
                    "--output",
                    str(oracle_dir),
                    "--steps",
                    str(case["expected_steps"]),
                    "--batch-size",
                    str(case["batch"]["global_samples"]),
                    "--micro-batch-size",
                    str(case["batch"]["micro_batch_samples"]),
                    "--sequence-length",
                    str(case["batch"]["sequence_length"]),
                    "--learning-rate",
                    str(arguments["learning_rate"]),
                    "--candidate-snapshots",
                    candidate_stages[0]["snapshot_dir"],
                ]
                oracle_process = run_process(
                    oracle_command,
                    cwd=REPO,
                    log_path=oracle_dir / "oracle.log",
                    timeout=case["timeout_seconds"],
                )
                if oracle_process["status"] != "PASS":
                    outcomes.append(
                        dict(
                            checks.result(
                                oracle_process["status"],
                                ["Independent oracle failed: " + oracle_process["log"]],
                            ),
                            kind=kind,
                        )
                    )
                    continue
                oracle_metrics = checks.load_metrics(
                    oracle_dir / "metrics.jsonl.rank_0.jsonl",
                    expected_steps=range(1, case["expected_steps"] + 1),
                    dtype=case["dtype"],
                )
                outcome = checks.compare_metrics(
                    oracle_metrics, metrics[0], dtype=case["dtype"],
                    **case.get("resolved_tolerances", {}).get(kind, {}).get("loss", {})
                )
                outcomes.append(
                    dict(
                        self.tensor_checks(
                            case, kind,
                            oracle_dir,
                            candidate_stages[0]["snapshot_dir"],
                            steps=range(1, case["expected_steps"] + 1),
                            required=("parameter", "gradient"),
                            check_initial=False,
                        ),
                        kind="oracle_tensors",
                    )
                )
            elif kind in ("history", "performance"):
                continue
            else:
                outcome = checks.result("ERROR", ["Unsupported check kind: " + kind])
            outcomes.append(dict(outcome, kind=kind))
        return outcomes

    def run_case(self, case):
        started = time.monotonic()
        reproduce_args = copy.copy(self.args)
        history = getattr(self, "history", None)
        if (
            not self.args.record_history
            and history is not None
            and history.reference is not None
            and case["id"] in history.previous_cases
        ):
            reproduce_args.reference_run = history.reference
        result = dict(
            id=case["id"],
            kind="training",
            status="ERROR",
            reasons=[],
            stages=[],
            checks=[],
            reproduce=command_for_case(
                reproduce_args, case, self.assets, REPO, self.pool.gpu_ids
            ),
        )
        directory = safe_case_path(self.cases_dir, case["id"])
        directory.mkdir(parents=True, exist_ok=True)
        try:
            blocked = [
                cid
                for cid in case["dependencies"]
                if "metrics" not in self.done.get(cid, {})
            ]
            if blocked:
                result.update(
                    status="BLOCKED",
                    reasons=[
                        "No valid dependency training result: " + ", ".join(blocked)
                    ],
                )
                return result
            performance = "performance" in case or self.plan.get("suite_metadata", {}).get("exclusive", False)
            with self.pool.lease(case["gpus"], exclusive=performance) as gpu_ids:
                count = case.get(
                    "repetitions", case.get("performance", {}).get("repetitions", 1)
                )
                for repeat in range(count):
                    stage_dir = directory / ("run-" + str(repeat + 1))
                    stage = self.train(case, stage_dir, gpu_ids, repeat=repeat)
                    result["stages"].append(stage)
                    if stage["status"] != "PASS":
                        result.update(status=stage["status"], reasons=stage["reasons"])
                        return result
                result["checks"] = self.evaluate(case, result["stages"])
                result["status"] = _status(result["checks"])
                result["reasons"] = [
                    r for ch in result["checks"] for r in ch.get("reasons", [])
                ]
                candidate = next(s for s in result["stages"] if not s["reference"])
                result["metrics"] = candidate["metrics"]
                result["snapshot_dir"] = candidate["snapshot_dir"]
        except Exception as error:
            result.update(
                status="ERROR", reasons=["{}: {}".format(type(error).__name__, error)]
            )
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            atomic_json(directory / "result.json", result)
        return result

    def run_cases(self):
        pending = list(self.plan["cases"])
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.args.jobs)
        try:
            active = {}
            while pending or active:
                active_cases = [self.case_map[cid] for cid in active.values()]
                used_gpus = sum(c["gpus"] for c in active_cases)
                active_exclusive = any(
                    "performance" in c or self.plan.get("suite_metadata", {}).get("exclusive", False)
                    for c in active_cases
                )
                for case in list(pending):
                    if len(active) >= self.args.jobs:
                        break
                    exclusive = "performance" in case or self.plan.get("suite_metadata", {}).get("exclusive", False)
                    if (
                        active_exclusive
                        or exclusive
                        and active
                        or used_gpus + case["gpus"] > len(self.pool.gpu_ids)
                    ):
                        continue
                    if all(d in self.done for d in case["dependencies"]):
                        pending.remove(case)
                        print("[RUN] " + case["id"], flush=True)
                        active[executor.submit(self.run_case, case)] = case["id"]
                        used_gpus += case["gpus"]
                        active_exclusive = exclusive
                if not active:
                    raise RuntimeError("Dependency scheduler cannot progress")
                completed, _ = concurrent.futures.wait(
                    active, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in completed:
                    cid = active.pop(future)
                    self.done[cid] = future.result()
                    print(
                        "[{}] {} {}".format(
                            self.done[cid]["status"],
                            cid,
                            "; ".join(self.done[cid]["reasons"])[:500],
                        ),
                        flush=True,
                    )
        except BaseException:
            terminate_all()
            raise
        finally:
            executor.shutdown(wait=True)
        if not self.args.keep_checkpoints and all(
            item["status"] in ("PASS", "RECORDED") for item in self.done.values()
        ):
            self.cleanup_checkpoints()
        return [self.done[c["id"]] for c in self.plan["cases"]]

    def cleanup_checkpoints(self):
        owner = self.checkpoint_dir / "owner.json"
        if self.checkpoint_dir.is_symlink() or not owner.is_file():
            raise ValueError("Checkpoint cleanup requires an owned run directory")
        if json.loads(owner.read_text()).get("run_output") != str(
            self.output.resolve()
        ):
            raise ValueError("Checkpoint owner does not match this run")
        for name in ("candidate", "reference"):
            directory = self.checkpoint_dir / name
            if directory.exists():
                if directory.is_symlink():
                    raise ValueError("Refusing to clean symlinked checkpoint directory")
                shutil.rmtree(directory)
        atomic_json(
            self.output / "checkpoint-retention.json",
            {"status": "removed_after_success", "directory": str(self.checkpoint_dir)},
        )


def unit_tests(args, output, pool):
    results = []
    for label in (["cpu"] if args.backend == "cpu" else ["cpu", args.backend]):
        directory = output / "unit" / label
        directory.mkdir(parents=True, exist_ok=True)
        with pool.lease(0 if label == "cpu" else len(pool.gpu_ids)) as devices:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            if devices:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            argv = [
                "ctest",
                "--test-dir",
                str(args.build_dir),
                "-L",
                "^" + label + "$",
                "--output-on-failure",
                "--no-tests=error",
                "--output-junit",
                str(directory / "junit.xml"),
                "-j",
                str(min(args.build_jobs, 8) if label == "cpu" else 1),
            ]
            with tempfile.TemporaryDirectory(prefix="infinitrain-unit-") as temporary:
                # Existing checkpoint tests use fixed names under temp_directory_path.
                # Isolate concurrent runs without changing every C++ test fixture.
                env["TMPDIR"] = temporary
                process = run_process(
                    argv,
                    cwd=REPO,
                    log_path=directory / "ctest.log",
                    timeout=args.build_timeout,
                    env=env,
                )
                if process["status"] != "PASS":
                    shutil.copytree(
                        temporary, directory / "failure-tmp", dirs_exist_ok=True
                    )
            results.append(
                dict(
                    id="unit/" + label,
                    kind="unit",
                    status=process["status"],
                    reasons=[process["reason"]] if process.get("reason") else [],
                    process=process,
                    elapsed_seconds=process["elapsed_seconds"],
                )
            )
    return results


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser()
    args = cli.parse_args(argv)
    run_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output, plan, run = None, None, None

    def failed_report(reason, status="ERROR"):
        if output is None or plan is None:
            return
        completed = list(run.done.values()) if run is not None else []
        known = {item["id"] for item in completed}
        for case in plan["cases"]:
            if case["id"] not in known:
                completed.append(
                    {
                        "id": case["id"],
                        "kind": "training",
                        "status": "BLOCKED",
                        "reasons": [reason],
                    }
                )
        completed.append(
            {
                "id": "infrastructure",
                "kind": "infrastructure",
                "status": status,
                "reasons": [reason],
            }
        )
        try:
            write_reports(output, plan, completed, {"error": reason})
        except OSError as report_error:
            print(
                "Could not write failure report to {}: {}".format(
                    output, report_error
                ),
                file=sys.stderr,
            )

    try:
        if (
            any(
                value == "--local-config" or value.startswith("--local-config=")
                for value in argv
            )
            and not args.local_config.is_file()
        ):
            raise ValueError(
                "Explicit local configuration not found: " + str(args.local_config)
            )
        _, fixture_overrides = _configuration(args)
        if (
            min(args.jobs, args.build_jobs, args.build_timeout, args.resource_timeout)
            < 1
        ):
            raise ValueError("Job counts and timeouts must be positive")
        if args.strict and args.record_history:
            raise ValueError(
                "--record-history is an explicit history rebase and cannot satisfy --strict"
            )
        if args.record_history and args.reference_run:
            raise ValueError("--record-history cannot combine with an explicit reference run")
        if args.rerun_failed:
            previous = json.loads(args.rerun_failed.read_text())
            args.suite = previous["suite"]
            args.case = [
                r["id"]
                for r in previous["results"]
                if r.get("kind") == "training" and r["status"] not in ("PASS", "RECORDED")
            ]
            if not args.case:
                raise ValueError("Previous run has no failed/unverified training cases")
        overrides = {key: json.loads(value) for key, value in _pairs(args.set).items()}
        if args.changed and (args.model or args.tag or args.case or args.rerun_failed):
            raise ValueError(
                "--changed cannot be combined with filters that erase fixed anchors"
            )
        plan = planning.load_plan(
            args.suite,
            models=args.model,
            tags=args.tag,
            cases=args.case,
            backend=args.backend,
            config_root=args.config_root,
            overrides=overrides,
            alignment=args.alignment,
        )
        _select_build_mode(args, plan)
        for case in plan["cases"]:
            case["argv_template"] = _argv(case, args.build_dir, case["resolved_args"])
        if args.changed:
            base = probe(["git", "-C", str(REPO), "merge-base", "HEAD", args.changed])
            if not base:
                raise ValueError(
                    "Cannot resolve merge base for --changed " + args.changed
                )
            changed = probe(["git", "-C", str(REPO), "diff", "--name-only", base])
            untracked = probe(
                ["git", "-C", str(REPO), "ls-files", "--others", "--exclude-standard"]
            )
            plan = planning.select_changed(
                plan,
                sorted(
                    set((changed or "").splitlines() + (untracked or "").splitlines())
                ),
            )
        if args.strict and (plan["coverage"]["partial"] or args.skip_unit):
            raise ValueError("Strict suite gates cannot omit cases or unit tests")
        if args.list or args.dry_run:
            if args.dry_run or args.explain:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
            else:
                for case in plan["cases"]:
                    print(
                        "{}  GPUs={}  checks={}".format(
                            case["id"],
                            case["gpus"],
                            ",".join(ch["kind"] for ch in case["checks"]),
                        )
                    )
                print(
                    "{} cases; partial={}".format(
                        len(plan["cases"]), plan["coverage"]["partial"]
                    )
                )
            return 0
        reexec = _reexec_container(args)
        if reexec is not None:
            return reexec
        # A failed multi-GB training process must not fill the artifact disk with
        # an unbounded core file. Structured logs and tensor snapshots are retained.
        _, core_hard_limit = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (0, core_hard_limit))
        output = (
            args.output or REPO / "artifacts" / "test-runs" / unique_run_id()
        ).resolve()
        if (output / "resolved_plan.json").exists():
            raise ValueError("Output already contains a run; choose a new directory")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "resolved_plan.json", plan)
        print("Artifacts: " + str(output), flush=True)
        assets = Assets(
            args.config_root,
            REPO / ".cache" / "regression",
            args.data_root,
            fixture_overrides,
        )
        errors = assets.prepare(plan["cases"])
        if not errors:
            required_storage = checkpoint_storage_budget(
                plan, assets
            )
            storage_error = storage_preflight(
                args.checkpoint_root or output, required_storage
            )
            atomic_json(
                output / "storage.json",
                {
                    "estimated_checkpoint_bytes": required_storage,
                    "checkpoint_root": str(
                        args.checkpoint_root or output / "checkpoints"
                    ),
                },
            )
            if storage_error:
                errors.append(storage_error)
        gpu_ids = (
            args.gpus.split(",")
            if args.gpus is not None
            else discover_gpus() if args.backend != "cpu" else []
        )
        required_gpus = max(c["gpus"] for c in plan["cases"])
        if required_gpus > len(gpu_ids):
            errors.append(
                "Suite needs up to {} GPUs, available {}".format(
                    required_gpus, len(gpu_ids)
                )
            )
        if errors:
            atomic_json(
                output / "preflight.json", {"status": "ERROR", "reasons": errors}
            )
            failed_report("; ".join(errors))
            for error in errors:
                print("[PREFLIGHT ERROR] " + error, file=sys.stderr)
            return 2
        atomic_json(
            output / "fixtures.json", {"paths": assets.paths, "hashes": assets.hashes}
        )
        if not args.no_build:
            print("[BUILD] " + str(args.build_dir), flush=True)
            built = build_project(
                REPO,
                args.build_dir,
                args.backend,
                output / "build",
                args.build_jobs,
                args.build_timeout,
                profile_mode=args.profile_mode,
                build_tests=bool(plan["suite_metadata"].get("ctest")),
            )
            if built["status"] != "PASS":
                atomic_json(output / "build-result.json", built)
                failed_report(built.get("reason", "Build failed; see " + built["log"]))
                print("[BUILD FAILED] " + built["log"], file=sys.stderr)
                return 2
        for build in [args.build_dir]:
            missing = [
                name
                for name in {c["binary"] for c in plan["cases"]}
                if not (Path(build) / name).is_file()
            ]
            if (
                any(c["nproc_per_node"] > 1 for c in plan["cases"])
                and not (Path(build) / "infini_run").is_file()
            ):
                missing.append("infini_run")
            if missing:
                raise ValueError(
                    "Missing build binaries in {}: {}".format(build, missing)
                )
        run = Run(args, plan, assets, output, gpu_ids)
        metadata = {
            "source": run.source,
            "environment": run.environment,
            "unit_tests_omitted": args.skip_unit,
            "partial": plan["coverage"]["partial"],
            "build_provenance": run.build_provenance,
            "profile_mode": run.profile_mode,
            "checkpoint_directory": str(run.checkpoint_dir),
            "config_root": str(args.config_root.resolve()),
            "started_at": run_started_at,
            "strict": bool(args.strict),
            "selection_scope": run.selection_scope,
            "alignment": {
                "mode": args.alignment,
                "history_root": str(Path(args.history_root).resolve()),
                "reference_run": str(Path(args.reference_run).resolve()) if args.reference_run else None,
                "selection": run.history_selection,
            },
        }
        atomic_json(output / "metadata.json", metadata)
        results = []
        if plan["suite_metadata"].get("ctest") and not args.skip_unit:
            print("[UNIT] CTest", flush=True)
            results.extend(unit_tests(args, output, run.pool))
        results.extend(run.run_cases())
        summary = write_reports(output, plan, results, metadata)
        print(
            "Results: {}\nSummary: {}".format(summary["counts"], output / "summary.md"),
            flush=True,
        )
        return 0 if summary["success"] else 1
    except (ValueError, OSError, planning.PlanError) as error:
        failed_report(str(error))
        print("Configuration error: " + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        terminate_all()
        failed_report("Run interrupted")
        print(
            "Interrupted; active process groups are terminated and artifacts retained",
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        failed_report("{}: {}".format(type(error).__name__, error))
        print(
            "Execution error: {}: {}".format(type(error).__name__, error),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())

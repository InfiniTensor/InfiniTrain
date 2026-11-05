"""Compare a training run with one fixed previous successful run's raw logs.

This module has no golden approval or tensor-state workflow. First runs and new
cases are explicitly RECORDED; invalid evidence never becomes a comparison PASS.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import statistics

try:
    from . import checks
except ImportError:
    import checks


_IGNORED_ARGS = {
    "metrics_file",
    "regression_snapshot_dir",
    "save",
    "lora_save_path",
    "input_bin",
    "input_val_bin",
    "tokenizer_bin",
    "llmc_filepath",
}
_PERF_ENV_FIELDS = (
    "backend",
    "system",
    "machine",
    "cpu",
    "gpu_inventory",
    "gpu_topology",
    "compiler",
    "cuda_compiler",
    "nccl",
    "build_options",
    "container_image",
    "selected_gpu_ids",
    "performance_profile",
)


def case_identity(case):
    """Training semantics only; editing assertions does not erase log history."""
    arguments = case.get("resolved_args", case.get("args"))
    if not isinstance(arguments, dict):
        raise ValueError("Case requires resolved training arguments")
    semantic_args = {
        key: value for key, value in arguments.items() if key not in _IGNORED_ARGS
    }
    # Checkpoint paths are relocated per run. Their symbolic source and step,
    # when available, describe the input; dropping them would hide resume bugs.
    for key in ("load", "lora_load_path"):
        if key in arguments:
            semantic_args[key] = case.get("args", {}).get(key, arguments[key])
    identity = {
        "model": case.get("model"),
        "binary": case.get("binary"),
        "args": semantic_args,
        "nproc_per_node": case.get("nproc_per_node", 1),
        "nthread_per_process": case.get("nthread_per_process", 1),
        "dtype": checks.normalize_dtype(case["dtype"]),
        "start_step": case.get("start_step", 1),
        "expected_steps": case["expected_steps"],
    }
    json.dumps(identity, sort_keys=True, allow_nan=False)
    return identity


def _output(status, reasons=(), *, reference=None, loss=None, performance=None, **data):
    result = checks.result(status, reasons, **data)
    result["reference_run"] = str(reference) if reference is not None else None
    result["loss"] = loss or checks.result("NOT_COMPARED", reasons)
    result["performance"] = performance or checks.result("NOT_COMPARED", reasons)
    return result


def _hashes(fixtures):
    if not isinstance(fixtures, dict):
        raise ValueError("Fixture hashes must be a mapping")
    return fixtures.get("hashes", fixtures)


def _candidate_stages(stages):
    if not isinstance(stages, list) or not stages:
        raise ValueError("Missing training stages")
    candidate = [stage for stage in stages if not stage.get("reference", False)]
    if not candidate:
        raise ValueError("No candidate training stages")
    for index, stage in enumerate(candidate):
        process = stage.get("process", {})
        if (
            stage.get("status") != "PASS"
            or process.get("status") != "PASS"
            or process.get("returncode") != 0
        ):
            raise ValueError("Training stage did not finish successfully")
        if stage.get("repeat", index) != index:
            raise ValueError("Training repetition IDs are missing, duplicated, or out of order")
    return candidate


def _completed_at(summary, manifest, path):
    raw = manifest.get("completed_at") or summary.get("metadata", {}).get("completed_at")
    if raw is None and not manifest:
        # Pre-history archives did not record completion time. Keep those usable
        # without letting filesystem copies reorder newer timestamped archives.
        return datetime.fromtimestamp((path / "summary.json").stat().st_mtime, timezone.utc)
    if not isinstance(raw, str):
        raise ValueError("Successful previous run has no completion timestamp")
    completed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if completed.tzinfo is None:
        raise ValueError("Completion timestamp must include a timezone")
    return completed.astimezone(timezone.utc)


def _validate_threshold(value, label, *, below_one=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (below_one and value >= 1)
    ):
        raise ValueError(
            label + " must be finite and nonnegative"
            + (" and below one" if below_one else "")
        )


class History:
    def __init__(
        self, root, current_output, suite, backend, reference_run=None, *,
        loss_atol_fp32=1e-5, loss_atol_bf16=1e-2, performance_threshold=.20,
        expected_case_ids=None, selection_scope=None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.current_output = Path(current_output).expanduser().resolve()
        self.suite, self.backend = suite, backend
        self.explicit = reference_run is not None
        self.requested = (
            Path(reference_run).expanduser().resolve()
            if reference_run is not None else None
        )
        self.loss_atol_fp32, self.loss_atol_bf16 = loss_atol_fp32, loss_atol_bf16
        self.performance_threshold = performance_threshold
        self.expected_case_ids = set(expected_case_ids or [])
        self.selection_scope = selection_scope
        _validate_threshold(loss_atol_fp32, "FP32 loss tolerance")
        _validate_threshold(loss_atol_bf16, "BF16 loss tolerance")
        _validate_threshold(performance_threshold, "Performance threshold", below_one=True)
        self.reference = None
        self.selection = None
        self.previous_cases, self.previous_results = {}, {}
        self.previous_fixtures, self.previous_environment = {}, {}

    def _matches_scope(self, summary):
        metadata = summary.get("metadata", {})
        previous_scope = metadata.get("selection_scope")
        if self.selection_scope is not None and previous_scope is not None:
            # A full-suite archive also covers a filtered invocation. The
            # reverse would let a small rerun replace full-suite comparison data.
            return self.selection_scope == previous_scope or not any(previous_scope.values())
        # Older archives may only distinguish full and filtered runs. A full
        # run can introduce new cases; a filtered archive must cover the request.
        if metadata.get("partial") is False:
            return True
        previous_ids = {
            item.get("id") for item in summary.get("results", [])
            if item.get("kind") == "training"
        }
        return not self.expected_case_ids or self.expected_case_ids <= previous_ids

    def _summary(self, path):
        summary = json.loads((path / "summary.json").read_text())
        if summary.get("suite") != self.suite:
            raise ValueError("Reference run uses a different suite")
        environment = summary.get("metadata", {}).get("environment", {})
        if environment.get("backend") != self.backend:
            raise ValueError("Reference run uses a different backend")
        if summary.get("complete") is not True or summary.get("success") is not True:
            raise ValueError("Reference run is incomplete or unsuccessful")
        results = summary.get("results")
        if (
            not isinstance(results, list)
            or not results
            or any(result.get("status") not in ("PASS", "RECORDED") for result in results)
        ):
            raise ValueError("Reference run contains failed or unverified results")
        return summary

    def select(self):
        """Freeze reference selection once; new completed runs cannot change it."""
        if self.selection is not None:
            return self.selection
        try:
            if self.explicit:
                if self.requested == self.current_output:
                    raise ValueError("Current run cannot be its own reference")
                if not self.requested.is_dir():
                    raise ValueError(
                        "Explicit reference run directory does not exist: "
                        + str(self.requested)
                    )
                chosen, summary = self.requested, self._summary(self.requested)
            else:
                candidates = []
                if self.root.is_dir():
                    for path in self.root.iterdir():
                        if not path.is_dir() or path.resolve() == self.current_output:
                            continue
                        if not (path / "summary.json").is_file():
                            manifest = (
                                json.loads((path / "run.json").read_text())
                                if (path / "run.json").is_file() else {}
                            )
                            if (
                                manifest.get("success") is True
                                and manifest.get("suite") == self.suite
                                and manifest.get("backend") == self.backend
                            ):
                                raise ValueError(
                                    "Successful previous run is missing summary.json: "
                                    + str(path)
                                )
                            continue
                        try:
                            summary = json.loads((path / "summary.json").read_text())
                        except ValueError:
                            manifest = (
                                json.loads((path / "run.json").read_text())
                                if (path / "run.json").is_file() else {}
                            )
                            if (
                                manifest.get("suite") == self.suite
                                and manifest.get("backend") == self.backend
                                and manifest.get("success") is True
                            ):
                                raise ValueError(
                                    "Successful previous run has malformed summary.json: "
                                    + str(path)
                                )
                            continue
                        backend = summary.get("metadata", {}).get("environment", {}).get("backend")
                        if summary.get("suite") != self.suite or backend != self.backend:
                            continue
                        if summary.get("complete") is not True or summary.get("success") is not True:
                            continue
                        if not self._matches_scope(summary):
                            continue
                        manifest = (
                            json.loads((path / "run.json").read_text())
                            if (path / "run.json").is_file() else {}
                        )
                        completed = _completed_at(summary, manifest, path)
                        candidates.append((completed, str(path), path))
                if not candidates:
                    self.selection = _output(
                        "RECORDED",
                        ["No completed successful previous run; this run records the first comparison data"],
                    )
                    return self.selection
                _, _, chosen = max(candidates, key=lambda candidate: candidate[:2])
                summary = self._summary(chosen)
            plan = json.loads((chosen / "resolved_plan.json").read_text())
            cases = plan.get("cases")
            if not isinstance(cases, list) or not cases:
                raise ValueError("Reference resolved plan has no cases")
            self.previous_cases = {case["id"]: case for case in cases}
            training = [item for item in summary["results"] if item.get("kind") == "training"]
            self.previous_results = {item["id"]: item for item in training}
            if (
                len(self.previous_cases) != len(cases)
                or len(self.previous_results) != len(training)
                or self.previous_cases.keys() != self.previous_results.keys()
            ):
                raise ValueError("Reference run has missing/duplicate/mismatched case results")
            self.previous_fixtures = _hashes(json.loads((chosen / "fixtures.json").read_text()))
            self.previous_environment = summary["metadata"]["environment"]
            self.reference = chosen
            self.selection = _output("PASS", reference=chosen, cases=len(cases))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            self.selection = _output(
                "ERROR", ["Invalid previous run: " + str(error)], reference=self.requested
            )
        return self.selection

    def _log_path(self, stage, case_id, index):
        raw = stage.get("process", {}).get("log")
        if not isinstance(raw, str) or not raw:
            raise ValueError("Previous stage has no raw log path")
        declared = Path(raw)
        if ".." in declared.parts:
            raise ValueError("Previous raw log path contains traversal")
        candidates = []
        if not declared.is_absolute():
            candidates.append(self.reference / declared)
        # Re-anchor a moved run's original absolute path at its cases directory.
        if declared.is_absolute() and "cases" in declared.parts:
            marker = len(declared.parts) - 1 - list(reversed(declared.parts)).index("cases")
            candidates.append(self.reference.joinpath(*declared.parts[marker:]))
        # Do not fall back to a guessed log when an explicit path is corrupt.
        # Such a fallback could silently compare a different repetition.
        root = self.reference.resolve()
        for candidate in candidates:
            resolved = candidate.resolve()
            if root not in resolved.parents:
                continue
            if resolved.is_file():
                return resolved
        raise ValueError("Previous raw training log is missing: " + raw)

    def _read_log(self, path, case):
        dtype = checks.normalize_dtype(case["dtype"])
        # Legacy text has dtype in the recorded argv, not on every loss line.
        # Verify it when present so a renamed BF16 log cannot impersonate FP32.
        lines = Path(path).read_text().splitlines()
        json_headers = [line for line in lines if line.startswith("[ARGV_JSON]")]
        if json_headers:
            if len(json_headers) != 1 or not json_headers[0].startswith("[ARGV_JSON] "):
                raise ValueError("Raw log must contain one valid ARGV_JSON header")
            words = json.loads(json_headers[0][len("[ARGV_JSON] "):])
            if not isinstance(words, list) or not words or any(
                not isinstance(word, str) for word in words
            ):
                raise ValueError("Raw log ARGV_JSON must be a nonempty list of strings")
            commands = [words]
        else:
            commands = [
                shlex.split(line[len("[COMMAND] "):])
                for line in lines if line.startswith("[COMMAND] ")
            ]
        for words in commands:
            declared = []
            for index, word in enumerate(words):
                if word.startswith("--dtype="):
                    declared.append(word.split("=", 1)[1])
                elif word == "--dtype" and index + 1 < len(words):
                    declared.append(words[index + 1])
            if any(checks.normalize_dtype(value) != dtype for value in declared):
                raise ValueError("Raw log dtype differs from case dtype")
        return checks.load_metrics(
            path,
            expected_steps=range(case.get("start_step", 1), case["expected_steps"] + 1),
            dtype=dtype,
            required=("loss", "tokens_per_second"),
        )

    def compare(
        self, case, stages, fixtures, environment, *, loss_atol_fp32=None,
        loss_atol_bf16=None, performance_threshold=None,
    ):
        selected = self.select()
        try:
            candidate_stages = _candidate_stages(stages)
            expected_repetitions = case.get(
                "repetitions", case.get("performance", {}).get("repetitions", 1)
            )
            if (
                type(expected_repetitions) is not int
                or expected_repetitions < 1
                or len(candidate_stages) != expected_repetitions
            ):
                raise ValueError("Current run repetition count differs from case contract")
            # Both sides use the same rounded raw-log precision and historical tok/s definition.
            candidates = [self._read_log(stage["process"]["log"], case) for stage in candidate_stages]
            if selected["status"] == "ERROR":
                return selected
            if self.reference is None:
                return _output("RECORDED", selected["reasons"])
            identifier = case["id"]
            if identifier not in self.previous_cases:
                return _output(
                    "ERROR" if self.explicit else "RECORDED",
                    ["Case is absent from the selected previous run"],
                    reference=self.reference,
                )
            previous_case = self.previous_cases[identifier]
            if case_identity(case) != case_identity(previous_case):
                return _output(
                    "NOT_COMPARABLE", ["Training configuration differs from the previous run"],
                    reference=self.reference,
                )
            actual_hashes = _hashes(fixtures)
            fixture_ids = set(case.get("fixtures", []))
            if fixture_ids != set(previous_case.get("fixtures", [])):
                return _output(
                    "NOT_COMPARABLE", ["Fixture inventory differs from the previous run"],
                    reference=self.reference,
                )
            for fixture in fixture_ids:
                if fixture not in actual_hashes or fixture not in self.previous_fixtures:
                    raise ValueError("Missing fixture hash: " + fixture)
                if actual_hashes[fixture] != self.previous_fixtures[fixture]:
                    return _output(
                        "NOT_COMPARABLE", ["Fixture content changed: " + fixture],
                        reference=self.reference,
                    )
            previous_stages = _candidate_stages(self.previous_results[identifier].get("stages"))
            if len(previous_stages) != expected_repetitions:
                raise ValueError("Previous run repetition count differs from case contract")
            references = [
                self._read_log(self._log_path(stage, identifier, index), previous_case)
                for index, stage in enumerate(previous_stages)
            ]
            dtype = checks.normalize_dtype(case["dtype"])
            if dtype == "bfloat16":
                atol = self.loss_atol_bf16 if loss_atol_bf16 is None else loss_atol_bf16
            else:
                atol = self.loss_atol_fp32 if loss_atol_fp32 is None else loss_atol_fp32
            threshold = (
                self.performance_threshold if performance_threshold is None
                else performance_threshold
            )
            _validate_threshold(atol, "Loss tolerance")
            _validate_threshold(threshold, "Performance threshold", below_one=True)
            loss_checks = [
                checks.compare_metrics(reference, candidate, dtype=dtype, atol=atol, rtol=0)
                for reference, candidate in zip(references, candidates)
            ]
            loss_reasons = [reason for result in loss_checks for reason in result["reasons"]]
            loss = checks.result(
                "FAIL" if loss_reasons else "PASS", loss_reasons,
                repetitions=loss_checks, atol=atol, rtol=0,
            )
            incompatible = [
                field for field in _PERF_ENV_FIELDS
                if field not in environment or field not in self.previous_environment
                or environment[field] != self.previous_environment[field]
            ]
            if incompatible:
                performance = checks.result(
                    "NOT_COMPARABLE",
                    ["Performance environment differs or is missing: " + ", ".join(incompatible)],
                )
            elif any(len(run.records) < 2 for run in references + candidates):
                performance = checks.result(
                    "NOT_COMPARABLE",
                    ["Performance needs at least two steps to exclude the first warmup step"],
                )
            else:
                reference_means = [
                    statistics.mean(row["tokens_per_second"] for row in run.records[1:])
                    for run in references
                ]
                candidate_means = [
                    statistics.mean(row["tokens_per_second"] for row in run.records[1:])
                    for run in candidates
                ]
                reference_mean = statistics.mean(reference_means)
                candidate_mean = statistics.mean(candidate_means)
                change = candidate_mean / reference_mean - 1
                reasons = (
                    [f"Throughput regression {-change:.2%} exceeds {threshold:.2%}"]
                    if change < -threshold else []
                )
                performance = checks.result(
                    "FAIL" if reasons else "PASS", reasons,
                    reference_mean=reference_mean, candidate_mean=candidate_mean,
                    reference_repeat_means=reference_means, candidate_repeat_means=candidate_means,
                    signed_change=change, threshold=threshold, warmup_steps=1,
                )
            if "FAIL" in (loss["status"], performance["status"]):
                status = "FAIL"
            elif performance["status"] == "NOT_COMPARABLE":
                status = "NOT_COMPARABLE"
            else:
                status = "PASS"
            return _output(
                status, loss["reasons"] + performance["reasons"],
                reference=self.reference, loss=loss, performance=performance,
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
            return _output("ERROR", [str(error)], reference=self.reference)

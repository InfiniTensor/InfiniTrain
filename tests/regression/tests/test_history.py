"""Tests for last-run raw-log loss and throughput alignment."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from history import History, case_identity


def case(case_id="case", *, save="artifact:old/checkpoints", steps=3):
    return {
        "id": case_id,
        "model": "gpt2",
        "binary": "gpt2",
        "dtype": "float32",
        "start_step": 1,
        "expected_steps": steps,
        "fixtures": ["fixture.weights"],
        "nproc_per_node": 1,
        "nthread_per_process": 1,
        "resolved_args": {
            "dtype": "float32",
            "save": save,
            "num_iteration": steps,
        },
    }


def log(path, losses, throughput, command='binary --dtype=float32'):
    lines = [
        "[RUN_ID] previous",
        "[UTC_TIME] 2026-10-10T00:00:00+00:00",
        "[GIT_COMMIT] " + "0" * 40,
        "[CASE_ID] case",
        "[COMMAND] " + command,
    ]
    lines.extend(
        "step {}/{} | train loss {:.6f} | {} tok/s".format(
            step, len(losses), loss, tps
        )
        for step, (loss, tps) in enumerate(zip(losses, throughput), 1)
    )
    Path(path).write_text("\n".join(lines) + "\n")


def stage(path, **overrides):
    result = {
        "status": "PASS", "reference": False, "repeat": 0,
        "process": {"log": str(path), "status": "PASS", "returncode": 0},
    }
    result.update(overrides)
    return result


def change_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.current = self.root / "current"
        self.previous = self.root / "previous"
        self.current.mkdir()
        self.previous.mkdir()
        self.environment = {
            "backend": "cuda",
            "system": "Linux",
            "machine": "x86_64",
            "gpu_inventory": ["gpu"],
            "gpu_topology": "topology",
            "compiler": "compiler",
            "cuda_compiler": "cuda",
            "nccl": "2",
            "build_options": {"CMAKE_BUILD_TYPE": "Release"},
            "cpu": "cpu",
            "container_image": "image",
            "selected_gpu_ids": ["0"],
            "performance_profile": None,
        }
        self.make_previous()

    def tearDown(self):
        self.temporary.cleanup()

    def make_previous(self):
        example = case(save="artifact:previous-private/checkpoints")
        stage_log = self.previous / "cases" / "case" / "run-1" / "train.log"
        stage_log.parent.mkdir(parents=True)
        log(stage_log, [3, 2, 1], [100, 110, 108])
        summary = {
            "suite": "pr",
            "complete": True,
            "success": True,
            "metadata": {"environment": self.environment},
            "results": [
                {
                    "id": "case",
                    "kind": "training",
                    "status": "PASS",
                    "stages": [stage(stage_log)],
                }
            ],
        }
        (self.previous / "summary.json").write_text(json.dumps(summary))
        (self.previous / "resolved_plan.json").write_text(
            json.dumps({"cases": [example]})
        )
        (self.previous / "fixtures.json").write_text(
            json.dumps({"hashes": {"fixture.weights": "a" * 64}})
        )
        (self.previous / "run.json").write_text(
            json.dumps({
                "suite": "pr", "backend": "cuda", "success": True,
                "completed_at": "2026-10-10T00:01:00+00:00",
            })
        )

    def history(self, explicit=False, **kwargs):
        return History(
            self.root,
            self.current,
            "pr",
            "cuda",
            self.previous if explicit else None,
            **kwargs,
        )

    def compare_candidate(self, losses, throughput):
        candidate_log = self.current / "case.log"
        log(candidate_log, losses, throughput)
        stages = [stage(candidate_log)]
        return self.history().compare(
            case(),
            stages,
            {"fixture.weights": "a" * 64},
            self.environment,
        )

    def test_first_run_is_recorded_not_false_pass(self):
        with tempfile.TemporaryDirectory() as empty:
            history = History(Path(empty), self.current, "pr", "cuda")
            self.assertEqual(history.select()["status"], "RECORDED")

    def test_loss_and_throughput_compare_and_artifact_paths_do_not_matter(self):
        outcome = self.compare_candidate([3, 2, 1], [101, 109, 107])
        self.assertEqual(outcome["status"], "PASS")
        self.assertEqual(outcome["loss"]["status"], "PASS")
        self.assertEqual(outcome["performance"]["status"], "PASS")

    def test_loss_regression_fails(self):
        outcome = self.compare_candidate([3, 2.1, 1], [100, 110, 108])
        self.assertEqual(outcome["loss"]["status"], "FAIL")
        self.assertEqual(outcome["status"], "FAIL")

    def test_throughput_regression_fails(self):
        outcome = self.compare_candidate([3, 2, 1], [100, 87, 86])
        self.assertEqual(outcome["performance"]["status"], "FAIL")
        self.assertEqual(outcome["status"], "FAIL")

    def test_legacy_throughput_cli_matches_history_arithmetic_mean(self):
        scripts = Path(__file__).resolve().parents[3] / "scripts"
        sys.path.insert(0, str(scripts))
        from compare_tps import compare_files

        reference_log = self.previous / "cases/case/run-1/train.log"
        log(reference_log, [3, 2, 1], [100000, 100, 900])
        reference_dir = self.root / "legacy-reference"
        candidate_dir = self.root / "legacy-candidate"
        reference_dir.mkdir()
        candidate_dir.mkdir()
        shutil.copyfile(reference_log, reference_dir / "case.log")
        # Nonuniform samples distinguish arithmetic from harmonic averaging;
        # deliberately different first steps also verify warmup exclusion.
        for rates, expected_mean, expected_status in (
            ([1, 500, 500], 500, "PASS"),
            ([1, 300, 400], 350, "FAIL"),
        ):
            with self.subTest(rates=rates):
                outcome = self.compare_candidate([3, 2, 1], rates)
                performance = outcome["performance"]
                self.assertEqual(performance["status"], expected_status)
                self.assertEqual(performance["data"]["reference_mean"], 500)
                self.assertEqual(performance["data"]["candidate_mean"], expected_mean)
                candidate_log = self.current / "case.log"
                result = compare_files(reference_log, candidate_log, 0.20)
                self.assertEqual(result[3], 500)
                self.assertEqual(result[4], expected_mean)
                self.assertEqual(result[5], performance["data"]["signed_change"])
                self.assertEqual(result[6:], (2, 2))
                self.assertEqual(result[1], expected_status == "FAIL")
                shutil.copyfile(candidate_log, candidate_dir / "case.log")
                cli = subprocess.run(
                    [sys.executable, str(scripts / "compare_tps.py"),
                     str(reference_dir), str(candidate_dir)],
                    capture_output=True, text=True,
                )
                self.assertEqual(cli.returncode, int(expected_status == "FAIL"),
                                 cli.stdout + cli.stderr)

    def test_configuration_change_is_not_comparable(self):
        candidate_log = self.current / "case.log"
        log(candidate_log, [4, 3, 2, 1], [100, 110, 108, 107])
        changed = case()
        changed["resolved_args"]["num_iteration"] = 4
        changed["expected_steps"] = 4
        outcome = self.history().compare(
            changed,
            [stage(candidate_log)],
            {"fixture.weights": "a" * 64},
            self.environment,
        )
        self.assertEqual(outcome["status"], "NOT_COMPARABLE")

    def test_environment_change_prevents_performance_comparison(self):
        candidate_log = self.current / "case.log"
        log(candidate_log, [3, 2, 1], [100, 110, 108])
        changed_environment = dict(self.environment, gpu_inventory=["other-gpu"])
        outcome = self.history().compare(
            case(),
            [stage(candidate_log)],
            {"fixture.weights": "a" * 64},
            changed_environment,
        )
        self.assertEqual(outcome["performance"]["status"], "NOT_COMPARABLE")

    def copy_run(self, name, completed):
        target = self.root / name
        shutil.copytree(self.previous, target)
        change_json(target / "run.json", lambda value: value.update(completed_at=completed))
        return target

    def candidate(self, history=None, example=None, candidate_stage=None):
        candidate_log = self.current / "case.log"
        log(candidate_log, [3, 2, 1], [100, 110, 108])
        return (history or self.history()).compare(
            example or case(), [candidate_stage or stage(candidate_log)],
            {"fixture.weights": "a" * 64}, self.environment,
        )

    def test_auto_uses_completion_instant_not_mtime_or_lexical_timezone(self):
        older = self.copy_run("mtime-newest", "2026-10-10T10:00:00+10:00")
        os.utime(older / "summary.json", (2_000_000_000, 2_000_000_000))
        self.assertEqual(self.history().select()["reference_run"], str(self.previous))

    def test_reference_is_frozen_before_new_run_completes(self):
        history = self.history()
        history.select()
        self.copy_run("later", "2026-10-10T01:00:00Z")
        self.assertEqual(history.select()["reference_run"], str(self.previous))
        self.assertEqual(self.candidate(history)["status"], "PASS")

    def test_current_run_cannot_select_itself(self):
        history = History(self.root, self.previous, "pr", "cuda", self.previous)
        self.assertEqual(history.select()["status"], "ERROR")
        self.assertEqual(History(self.root, self.previous, "pr", "cuda").select()["status"], "RECORDED")

    def test_failed_or_incomplete_new_run_does_not_replace_success(self):
        for flag in ("success", "complete"):
            with self.subTest(flag=flag):
                newer = self.copy_run("later-" + flag, "2026-10-10T01:00:00Z")
                change_json(newer / "summary.json", lambda value: value.update({flag: False}))
                self.assertEqual(self.history().select()["reference_run"], str(self.previous))

    def test_corrupt_latest_success_is_error_not_fallback(self):
        newer = self.copy_run("later", "2026-10-10T01:00:00Z")
        (newer / "resolved_plan.json").write_text("{")
        self.assertEqual(self.history().select()["status"], "ERROR")

    def test_corrupt_latest_success_summary_is_error_not_first_recording(self):
        (self.previous / "summary.json").write_text("{")
        self.assertEqual(self.history().select()["status"], "ERROR")

    def test_naive_completion_timestamp_is_rejected(self):
        change_json(
            self.previous / "run.json",
            lambda value: value.update(completed_at="2026-10-10T00:00:00"),
        )
        self.assertEqual(self.history().select()["status"], "ERROR")

    def test_legacy_archive_can_use_mtime_when_completion_was_not_recorded(self):
        (self.previous / "run.json").unlink()
        self.assertEqual(self.candidate()["status"], "PASS")

    def test_full_and_filtered_runs_have_separate_reference_scopes(self):
        full_scope = {"models": [], "tags": [], "cases": [], "changed": None}
        change_json(
            self.previous / "summary.json",
            lambda value: value["metadata"].update(selection_scope=full_scope),
        )
        newer = self.copy_run("filtered", "2026-10-10T01:00:00Z")
        change_json(
            newer / "summary.json",
            lambda value: value["metadata"].update(
                selection_scope=dict(full_scope, models=["gpt2"])
            ),
        )
        history = self.history(selection_scope=full_scope, expected_case_ids=["case", "new-case"])
        self.assertEqual(history.select()["reference_run"], str(self.previous))

    def test_legacy_filtered_archive_cannot_replace_full_reference(self):
        change_json(self.previous / "summary.json", lambda value: value["metadata"].update(partial=False))
        newer = self.copy_run("filtered", "2026-10-10T01:00:00Z")
        change_json(newer / "summary.json", lambda value: value["metadata"].update(partial=True))
        history = self.history(expected_case_ids=["case", "new-case"])
        self.assertEqual(history.select()["reference_run"], str(self.previous))

    def test_filtered_invocation_can_reuse_full_archive_but_not_other_filter(self):
        full_scope = {"models": [], "tags": [], "cases": [], "changed": None}
        change_json(
            self.previous / "summary.json",
            lambda value: value["metadata"].update(selection_scope=full_scope),
        )
        other_filter = self.copy_run("other-filter", "2026-10-10T01:00:00Z")
        change_json(
            other_filter / "summary.json",
            lambda value: value["metadata"].update(
                selection_scope=dict(full_scope, models=["qwen3"])
            ),
        )
        requested_scope = dict(full_scope, models=["gpt2"])
        history = self.history(selection_scope=requested_scope, expected_case_ids=["case"])
        self.assertEqual(history.select()["reference_run"], str(self.previous))
        matching_filter = self.copy_run("matching-filter", "2026-10-10T00:30:00Z")
        change_json(
            matching_filter / "summary.json",
            lambda value: value["metadata"].update(selection_scope=requested_scope),
        )
        history = self.history(selection_scope=requested_scope, expected_case_ids=["case"])
        self.assertEqual(history.select()["reference_run"], str(matching_filter))
        newest_full = self.copy_run("newest-full", "2026-10-10T02:00:00Z")
        history = self.history(selection_scope=requested_scope, expected_case_ids=["case"])
        self.assertEqual(history.select()["reference_run"], str(newest_full))

    def test_new_case_is_recorded_but_explicit_reference_missing_case_is_error(self):
        self.assertEqual(self.candidate(example=case("new-case"))["status"], "RECORDED")
        self.assertEqual(self.candidate(self.history(explicit=True), case("new-case"))["status"], "ERROR")

    def test_checkpoint_source_and_step_remain_in_identity(self):
        first, second = case(), case()
        first["args"] = {"load": "artifact:save-case/checkpoint_step_000003"}
        second["args"] = dict(first["args"])
        first["resolved_args"]["load"] = "/previous/checkpoint_step_000003"
        second["resolved_args"]["load"] = "/current/checkpoint_step_000003"
        self.assertEqual(case_identity(first), case_identity(second))
        second["args"]["load"] = "artifact:other-case/checkpoint_step_000003"
        self.assertNotEqual(case_identity(first), case_identity(second))
        second["args"]["load"] = "artifact:save-case/checkpoint_step_000004"
        self.assertNotEqual(case_identity(first), case_identity(second))
        second["args"]["load"] = ""
        self.assertNotEqual(case_identity(first), case_identity(second))

    def test_relocated_archive_reads_local_logs_not_original_directory(self):
        moved = self.copy_run("moved", "2026-10-10T01:00:00Z")
        (self.previous / "cases/case/run-1/train.log").write_text("corrupt original")
        self.assertEqual(self.candidate()["status"], "PASS")
        self.assertEqual(self.history().select()["reference_run"], str(moved))

    def test_log_escape_or_missing_declared_path_cannot_use_guessed_log(self):
        for raw in (
            "../previous/cases/case/run-1/train.log",
            "cases/case/run-99/train.log",
            "/outside/train.log",
        ):
            with self.subTest(raw=raw):
                change_json(
                    self.previous / "summary.json",
                    lambda value: value["results"][0]["stages"][0]["process"].update(log=raw),
                )
                self.assertEqual(self.candidate()["status"], "ERROR")

    def test_log_symlink_cannot_escape_archive(self):
        raw = self.previous / "cases/case/run-1/train.log"
        outside = self.root / "outside.log"
        raw.rename(outside)
        raw.symlink_to(outside)
        self.assertEqual(self.candidate()["status"], "ERROR")

    def test_previous_failed_process_with_complete_log_is_rejected(self):
        change_json(
            self.previous / "summary.json",
            lambda value: value["results"][0]["stages"][0]["process"].update(returncode=1),
        )
        self.assertEqual(self.candidate()["status"], "ERROR")

    def test_candidate_failed_process_is_rejected_even_with_complete_log(self):
        failed = stage(self.current / "case.log")
        failed["process"].update(returncode=1)
        self.assertEqual(self.candidate(candidate_stage=failed)["status"], "ERROR")

    def test_invalid_logs_never_pass(self):
        for content in (
            "",
            "step 1/3 | train loss nan | 100 tok/s\n",
            "step 1/3 | train loss 3 | 100 tok/s\n",
        ):
            with self.subTest(content=content):
                (self.previous / "cases/case/run-1/train.log").write_text(content)
                self.assertEqual(self.candidate()["status"], "ERROR")

    def test_missing_fixture_and_changed_fixture_are_not_pass(self):
        (self.previous / "fixtures.json").write_text(json.dumps({"hashes": {}}))
        self.assertEqual(self.candidate()["status"], "ERROR")
        (self.previous / "fixtures.json").write_text(json.dumps({"hashes": {"fixture.weights": "b" * 64}}))
        self.assertEqual(self.candidate()["status"], "NOT_COMPARABLE")

    def test_warmup_throughput_is_excluded(self):
        outcome = self.compare_candidate([3, 2, 1], [1, 110, 108])
        self.assertEqual(outcome["status"], "PASS")
        self.assertEqual(outcome["performance"]["data"]["candidate_mean"], 109)

    def test_loss_uses_absolute_tolerance_not_relative_tolerance(self):
        raw = self.previous / "cases/case/run-1/train.log"
        log(raw, [10, 10, 10], [100, 110, 108])
        self.assertEqual(self.compare_candidate([10, 10.00002, 10], [100, 110, 108])["status"], "FAIL")

    def test_bfloat16_tolerance_is_selected_from_config_not_filename(self):
        example = case()
        example["dtype"] = "bfloat16"
        example["resolved_args"]["dtype"] = "bfloat16"
        change_json(self.previous / "resolved_plan.json", lambda value: value.update(cases=[example]))
        raw = self.previous / "cases/case/run-1/train.log"
        log(raw, [3, 2, 1], [100, 110, 108], command="binary --dtype=bfloat16")
        candidate_log = self.current / "misleading-fp32.log"
        log(candidate_log, [3, 2.005, 1], [100, 110, 108], command="binary --dtype=bfloat16")
        outcome = self.history().compare(
            example, [stage(candidate_log)],
            {"fixture.weights": "a" * 64}, self.environment,
        )
        self.assertEqual(outcome["status"], "PASS")
        self.assertEqual(outcome["loss"]["data"]["atol"], .01)
        log(candidate_log, [3, 2.02, 1], [100, 110, 108], command="binary --dtype=bfloat16")
        outcome = self.history().compare(
            example, [stage(candidate_log)],
            {"fixture.weights": "a" * 64}, self.environment,
        )
        self.assertEqual(outcome["status"], "FAIL")

    def test_manifest_cannot_hide_missing_or_duplicate_case_results(self):
        change_json(
            self.previous / "summary.json",
            lambda value: value["results"].append(value["results"][0]),
        )
        self.assertEqual(self.history().select()["status"], "ERROR")

    def test_repeated_stage_ids_are_rejected(self):
        candidate_log = self.current / "case.log"
        log(candidate_log, [3, 2, 1], [100, 110, 108])
        example = dict(case(), repetitions=2)
        outcome = self.history().compare(
            example, [stage(candidate_log), stage(candidate_log)],
            {"fixture.weights": "a" * 64}, self.environment,
        )
        self.assertEqual(outcome["status"], "ERROR")

    def test_json_argv_preserves_newline_paths_without_parsing_readable_command(self):
        raw = self.previous / "cases/case/run-1/train.log"
        argv = ["binary", "--input_bin", "/data/new\nline's file.bin", "--dtype=float32"]
        log(raw, [3, 2, 1], [100, 110, 108], command=shlex.join(argv))
        raw.write_text("[ARGV_JSON] " + json.dumps(argv) + "\n" + raw.read_text())
        self.assertEqual(self.candidate()["status"], "PASS")

    def test_invalid_json_argv_cannot_fall_back_to_readable_command(self):
        raw = self.previous / "cases/case/run-1/train.log"
        for header in ("{", "{}", "[]", '["binary", 1]', '"binary"'):
            with self.subTest(header=header):
                log(raw, [3, 2, 1], [100, 110, 108])
                raw.write_text("[ARGV_JSON] " + header + "\n" + raw.read_text())
                self.assertEqual(self.candidate()["status"], "ERROR")

    def test_json_argv_dtype_mismatch_is_rejected(self):
        raw = self.previous / "cases/case/run-1/train.log"
        raw.write_text('[ARGV_JSON] ["binary", "--dtype", "bfloat16"]\n' + raw.read_text())
        self.assertEqual(self.candidate()["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()

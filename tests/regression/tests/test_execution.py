"""Exercise process failures, isolation and reporting without requiring GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from execution import (
    ResourcePool,
    atomic_json,
    file_sha256,
    run_process,
    safe_case_path,
)
from reporting import write_reports
from runner import Run, checkpoint_storage_budget, storage_preflight
from execution import command_text


class ProcessTests(unittest.TestCase):
    @staticmethod
    def process_exited(pid):
        try:
            return (Path("/proc") / str(pid) / "stat").read_text().split()[2] == "Z"
        except (FileNotFoundError, ProcessLookupError):
            return True

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_nonzero_exit_and_literal_arguments(self):
        literal = "$(touch SHOULD_NOT_EXIST); `echo nope`"
        outcome = run_process(
            [
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); sys.exit(7)",
                literal,
            ],
            cwd=self.root,
            log_path=self.root / "run.log",
            timeout=5,
        )
        self.assertEqual(outcome["status"], "FAIL")
        self.assertEqual(outcome["returncode"], 7)
        self.assertIn(literal, (self.root / "run.log").read_text())
        self.assertFalse((self.root / "SHOULD_NOT_EXIST").exists())

    def test_process_log_records_run_context_before_command(self):
        outcome = run_process(
            [sys.executable, "-c", "print(0)"],
            cwd=self.root,
            log_path=self.root / "context.log",
            timeout=5,
            headers={
                "RUN_ID": "run",
                "UTC_TIME": "2026-10-10T00:00:00+00:00",
                "GIT_COMMIT": "0" * 40,
                "CASE_ID": "case",
            },
        )
        self.assertEqual(outcome["status"], "PASS")
        lines = (self.root / "context.log").read_text().splitlines()
        self.assertEqual(lines[:5], [
            "[RUN_ID] run",
            "[UTC_TIME] 2026-10-10T00:00:00+00:00",
            "[GIT_COMMIT] " + "0" * 40,
            "[CASE_ID] case",
            "[COMMAND] " + command_text([sys.executable, "-c", "print(0)"]),
        ])

    def test_timeout_kills_spawned_group(self):
        pid_file = self.root / "child.pid"
        code = (
            "import subprocess,time,pathlib,sys; "
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"]); '
            "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(120)"
        )
        outcome = run_process(
            [sys.executable, "-c", code, str(pid_file)],
            cwd=self.root,
            log_path=self.root / "timeout.log",
            timeout=0.4,
        )
        self.assertEqual(outcome["status"], "ERROR")
        self.assertIn("timeout", outcome["reason"])
        pid = int(pid_file.read_text())
        # An orphan zombie has already terminated; it awaits the host's reaper.
        self.assertTrue(self.process_exited(pid))

    def test_missing_executable_is_error(self):
        outcome = run_process(
            [str(self.root / "missing")],
            cwd=self.root,
            log_path=self.root / "missing.log",
            timeout=1,
        )
        self.assertEqual(outcome["status"], "ERROR")

    def test_failed_launcher_cannot_leave_sigterm_ignoring_rank(self):
        pid_file = self.root / "rank.pid"
        child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"
        code = (
            'import subprocess,time,pathlib,sys; p=subprocess.Popen([sys.executable,"-c",sys.argv[2]]); '
            "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(.2); sys.exit(1)"
        )
        outcome = run_process(
            [sys.executable, "-c", code, str(pid_file), child],
            cwd=self.root,
            log_path=self.root / "leader.log",
            timeout=5,
        )
        self.assertEqual(outcome["status"], "FAIL")
        pid = int(pid_file.read_text())
        for _ in range(20):
            if self.process_exited(pid):
                break
            time.sleep(0.01)
        self.assertTrue(self.process_exited(pid))

    def test_fixture_hash_cache_detects_changed_bytes(self):
        path = self.root / "fixture.bin"
        path.write_bytes(b"aaaa")
        first = file_sha256(path, self.root / "cache")
        stamp = path.stat().st_mtime_ns
        path.write_bytes(b"bbbb")
        os.utime(str(path), ns=(stamp, stamp))
        self.assertNotEqual(first, file_sha256(path, self.root / "cache"))

    def test_paths_cannot_escape_artifacts(self):
        for invalid in ("../outside", "a/../../outside", "/absolute", "a//b", "a/./b"):
            with self.assertRaises(ValueError):
                safe_case_path(self.root, invalid)
        self.assertEqual(
            safe_case_path(self.root, "model/case"), self.root / "model" / "case"
        )

    def test_resource_leases_release_and_exclude_other_runners(self):
        first = ResourcePool(["0", "1"], self.root / "locks", wait_seconds=0.1)
        second = ResourcePool(["0", "1"], self.root / "locks", wait_seconds=0.1)
        with first.lease(1) as assigned:
            with second.lease(1) as other:
                self.assertNotEqual(assigned, other)
            with self.assertRaises(RuntimeError):
                with second.lease(2):
                    self.fail("must not acquire occupied GPU")
        with second.lease(2) as all_gpus:
            self.assertEqual(all_gpus, ["0", "1"])

    def test_atomic_json_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            atomic_json(self.root / "bad.json", {"loss": float("nan")})
        self.assertFalse((self.root / "bad.json").exists())


class ReportingTests(unittest.TestCase):
    def test_incomparable_and_missing_case_cannot_pass(self):
        plan = {
            "suite": "pr",
            "cases": [{"id": "a"}, {"id": "b"}],
            "coverage": {"matrices": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            report = write_reports(
                tmp,
                plan,
                [
                    {
                        "id": "a",
                        "kind": "training",
                        "status": "NOT_COMPARABLE",
                        "reasons": ["no comparable history"],
                    }
                ],
                {},
            )
            self.assertFalse(report["success"])
            self.assertFalse(report["complete"])
            self.assertIn("NOT_COMPARABLE", (Path(tmp) / "junit.xml").read_text())
            coverage = json.loads((Path(tmp) / "coverage.json").read_text())
            self.assertEqual(coverage["failed_or_unverified_case_ids"], ["a", "b"])


class CheckpointOwnershipTests(unittest.TestCase):
    def test_cleanup_only_after_dependent_resume_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = Run.__new__(Run)
            run.output = root / "report"
            run.output.mkdir()
            run.checkpoint_dir = root / "scratch" / "owned"
            run.checkpoint_dir.mkdir(parents=True)
            atomic_json(
                run.checkpoint_dir / "owner.json",
                {"run_output": str(run.output.resolve())},
            )
            unrelated = root / "scratch" / "unrelated.bin"
            unrelated.write_text("keep")
            source = {"id": "source", "dependencies": [], "gpus": 0, "checks": []}
            resume = {
                "id": "resume",
                "dependencies": ["source"],
                "gpus": 0,
                "checks": [],
            }
            run.plan = {"cases": [source, resume]}
            run.case_map = {c["id"]: c for c in run.plan["cases"]}
            run.done = {}
            run.pool = SimpleNamespace(gpu_ids=[])
            run.args = SimpleNamespace(jobs=2, keep_checkpoints=False)
            checkpoint = (
                run.checkpoint_dir
                / "candidate"
                / "source"
                / "checkpoints"
                / "state.bin"
            )

            def execute(case):
                if case["id"] == "source":
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_text("saved state")
                else:
                    self.assertEqual(checkpoint.read_text(), "saved state")
                return {"id": case["id"], "status": "PASS", "reasons": []}

            run.run_case = execute
            outcomes = run.run_cases()
            self.assertEqual(len(outcomes), 2)
            self.assertFalse(checkpoint.exists())
            self.assertEqual(unrelated.read_text(), "keep")
            self.assertTrue((run.checkpoint_dir / "owner.json").exists())

    def test_wrong_owner_never_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Run.__new__(Run)
            run.output = Path(temporary) / "report"
            run.checkpoint_dir = Path(temporary) / "scratch"
            run.checkpoint_dir.mkdir()
            atomic_json(
                run.checkpoint_dir / "owner.json", {"run_output": "someone else's run"}
            )
            with self.assertRaises(ValueError):
                run.cleanup_checkpoints()
            self.assertTrue(run.checkpoint_dir.exists())

    def test_storage_budget_includes_all_retained_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            weight = Path(temporary) / "weights"
            weight.write_bytes(b"x" * 1024)
            case = {
                "resolved_args": {
                    "save": "artifact:a/checkpoints",
                    "save_interval": 2,
                    "max_checkpoint_keep": 3,
                },
                "fixtures": ["tiny.gpt2.weights"],
                "expected_steps": 10,
            }
            assets = SimpleNamespace(paths={"tiny.gpt2.weights": str(weight)})
            required = checkpoint_storage_budget(
                {"cases": [case]}, assets, reference=True
            )
            self.assertGreaterEqual(required, 1024 * 4 * 3 * 2)
            self.assertIn("--checkpoint-root", storage_preflight(temporary, 10**30))


if __name__ == "__main__":
    unittest.main()

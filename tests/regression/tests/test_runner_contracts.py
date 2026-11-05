"""Independent regression gate/report/provenance tests; no GPU required."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checks
import reporting
import runner


class RunnerContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_removed_baseline_options_are_rejected(self):
        for argv in (
            ["--baseline-dir", "/tmp/baselines"],
            ["--propose-baselines"],
            ["--alignment", "baseline"],
            ["baseline", "accept"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.parser().parse_args(argv)
                self.assertEqual(error.exception.code, 2)
        for mode in ("history", "deep"):
            self.assertEqual(
                runner.parser().parse_args(["--alignment", mode]).alignment, mode
            )

    def test_duplicate_results_cannot_hide_missing_case(self):
        plan = {"suite": "pr", "cases": [{"id": "a"}, {"id": "b"}], "coverage": {}}
        results = [{"id": "a", "kind": "training", "status": "PASS"} for _ in range(2)]
        summary = reporting.write_reports(self.root, plan, results, {})
        self.assertFalse(summary["success"])
        self.assertFalse(summary["complete"])

    def test_empty_results_and_plan_never_report_success(self):
        summary = reporting.write_reports(
            self.root, {"suite": "pr", "cases": [], "coverage": {}}, [], {}
        )
        self.assertFalse(summary["success"])

    def test_first_history_run_records_but_does_not_claim_comparison(self):
        plan = {"suite": "pr", "cases": [{"id": "case"}], "coverage": {}}
        result = {
            "id": "case",
            "kind": "training",
            "status": "RECORDED",
            "reasons": ["first run"],
            "stages": [
                {
                    "reference": False,
                    "repeat": 0,
                    "process": {
                        "argv": ["binary", "--case=case"],
                        "cwd": "/repo",
                        "log": "/repo/train.log",
                        "elapsed_seconds": 1,
                    },
                }
            ],
        }
        summary = reporting.write_reports(
            self.root, plan, [result], {"started_at": "2026-10-10T00:00:00+00:00"}
        )
        self.assertTrue(summary["success"])
        manifest = json.loads((self.root / "run.json").read_text())
        self.assertEqual(
            manifest["cases"][0]["commands"][0]["argv"], ["binary", "--case=case"]
        )
        self.assertTrue(manifest["completed_at"])
        coverage = json.loads((self.root / "coverage.json").read_text())
        self.assertEqual(coverage["recorded_case_ids"], ["case"])
        self.assertEqual(coverage["failed_or_unverified_case_ids"], ["case"])

    def test_strict_gate_rejects_record_only_history(self):
        plan = {"suite": "pr", "cases": [{"id": "case"}], "coverage": {}}
        result = {
            "id": "case",
            "kind": "training",
            "status": "RECORDED",
            "reasons": ["first run"],
            "stages": [],
        }
        summary = reporting.write_reports(self.root, plan, [result], {"strict": True})
        self.assertFalse(summary["success"])


    def test_multiprocess_launcher_is_part_of_verified_build(self):
        args = SimpleNamespace(
            checkpoint_root=None,
            resource_timeout=1,
            build_dir=self.root / "build",
            backend="cuda",
            perf_environment=None,
            history_root=self.root / "history",
            reference_run=None,
            loss_atol_fp32=1e-5,
            loss_atol_bf16=1e-2,
            performance_threshold=0.2,
        )
        plan = {
            "suite": "pr",
            "cases": [
                {
                    "id": "case",
                    "binary": "gpt2",
                    "nproc_per_node": 2,
                    "nthread_per_process": 1,
                }
            ]
        }
        with mock.patch.object(runner, "ResourcePool"), mock.patch.object(
            runner, "environment_fingerprint", return_value={}
        ), mock.patch.object(
            runner, "source_metadata", return_value={}
        ), mock.patch.object(
            runner, "verify_build_manifest", return_value={"verified": False}
        ) as verify:
            runner.Run(args, plan, None, self.root, [])
        self.assertIn("infini_run", verify.call_args.args[1])


    def test_checkpoint_space_accounts_for_data_parallel_copies(self):
        weights = self.root / "weights.bin"
        weights.write_bytes(b"x" * 1024)
        assets = SimpleNamespace(paths={"model.weights": str(weights)})
        case = {
            "id": "checkpoint",
            "fixtures": ["model.weights"],
            "expected_steps": 1,
            "topology": {"dp": 1},
            "resolved_args": {
                "save": "artifact:case/checkpoints",
                "save_interval": 1,
                "max_checkpoint_keep": 1,
                "save_optimizer_state": True,
                "zero_stage": 0,
            },
        }
        single = runner.checkpoint_storage_budget({"cases": [case]}, assets)
        case["topology"]["dp"] = 8
        replicated = runner.checkpoint_storage_budget({"cases": [case]}, assets)
        self.assertGreaterEqual(replicated, single * 8)

    def owned_run(self):
        run = runner.Run.__new__(runner.Run)
        run.output = self.root / "output"
        run.output.mkdir()
        run.checkpoint_dir = self.root / "scratch"
        run.checkpoint_dir.mkdir()
        (run.checkpoint_dir / "owner.json").write_text(
            json.dumps({"run_output": str(run.output.resolve())})
        )
        (run.checkpoint_dir / "candidate").mkdir()
        (run.checkpoint_dir / "candidate" / "weights").write_text("owned checkpoint")
        return run

    def test_checkpoint_cleanup_preserves_unowned_paths(self):
        run = self.owned_run()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "weights").write_text("user data")
        run.cleanup_checkpoints()
        self.assertEqual((outside / "weights").read_text(), "user data")
        self.assertFalse((run.checkpoint_dir / "candidate").exists())
        self.assertTrue((run.checkpoint_dir / "owner.json").is_file())

    def test_checkpoint_cleanup_rejects_bad_owner_and_symlink(self):
        run = self.owned_run()
        (run.checkpoint_dir / "owner.json").write_text(
            json.dumps({"run_output": "/some/other/run"})
        )
        with self.assertRaises(ValueError):
            run.cleanup_checkpoints()
        self.assertTrue((run.checkpoint_dir / "candidate" / "weights").is_file())
        (run.checkpoint_dir / "owner.json").write_text(
            json.dumps({"run_output": str(run.output.resolve())})
        )
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "weights").write_text("user data")
        (run.checkpoint_dir / "reference").symlink_to(outside)
        with self.assertRaises(ValueError):
            run.cleanup_checkpoints()
        self.assertEqual((outside / "weights").read_text(), "user data")


if __name__ == "__main__":
    unittest.main()

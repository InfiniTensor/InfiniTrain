"""Default log mode must not secretly execute diagnostic reference workloads."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import planning
import runner


class LogPlanTests(unittest.TestCase):
    def test_default_cases_drop_diagnostic_references_keep_checkpoint_sources(self):
        plan = planning.load_plan("pr", alignment="history")
        self.assertEqual(len(plan["cases"]), 87)
        cases = {case["id"]: case for case in plan["cases"]}
        for case in cases.values():
            self.assertEqual(case["checks"], [{"kind": "finite_and_complete"}, {"kind": "history"}])
            for dep in case["dependencies"]:
                self.assertIn(dep, cases)
            if case["id"].endswith("/reference"):
                self.assertTrue(any(other["args"].get("load", "").startswith("artifact:" + case["id"] + "/") for other in cases.values()))
        self.assertFalse(plan["coverage"]["partial"])

    def test_classic_training_commands_preserved(self):
        deep = planning.load_plan("nightly", cases=["classic_*"], alignment="deep")
        logs = planning.load_plan("nightly", cases=["classic_*"], alignment="history")
        self.assertEqual(len(logs["cases"]), 212)
        self.assertEqual({c["id"]: c["resolved_args"] for c in deep["cases"]},
                         {c["id"]: c["resolved_args"] for c in logs["cases"]})

    def test_checkpoint_filter_includes_producer_and_other_filters_stay_partial(self):
        plan = planning.load_plan("pr", cases=["tiny_checkpoint/gpt2/dtype=float32"], alignment="history")
        self.assertEqual(len(plan["cases"]), 2)
        self.assertEqual(plan["cases"][1]["dependencies"], [plan["cases"][0]["id"]])
        self.assertTrue(plan["coverage"]["partial"])
        for suite in ("smoke", "pr", "nightly", "perf", "release"):
            self.assertFalse(planning.load_plan(suite, alignment="history")["coverage"]["partial"])


class LogWorkflowTests(unittest.TestCase):
    def test_record_compare_regression_and_failed_run_does_not_replace_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local.json"
            local.write_text("{}")
            build = root / "build"
            build.mkdir()
            control = build / "control.json"
            control.write_text(json.dumps({"loss": 1.0, "tps": 100}))
            executable = build / "gpt2"
            executable.write_text(
                "#!" + sys.executable + "\n"
                "import json,sys\nfrom pathlib import Path\n"
                "assert not any(a.startswith(('--metrics_file=', '--regression_snapshot_dir=')) for a in sys.argv)\n"
                "c=json.loads(Path(__file__).with_name('control.json').read_text())\n"
                "for step in range(1,5):\n"
                " print('step %d/4 | train loss %.6f | lr 1e-4 | (1.0 ms | %d tok/s)' % (step,c['loss'],c['tps']))\n"
            )
            executable.chmod(0o755)
            environment = {
                "backend": "cpu", "system": "Linux", "machine": "test",
                "cpu": "test", "gpu_inventory": [], "gpu_topology": None,
                "compiler": "test", "cuda_compiler": None, "nccl": None,
                "build_options": {}, "container_image": "test",
            }
            common = ["--suite", "smoke", "--model", "gpt2", "--backend", "cpu",
                      "--local-config", str(local), "--config-root", str(runner.ROOT),
                      "--build-dir", str(build), "--history-root", str(root / "runs"),
                      "--checkpoint-root", str(root / "checkpoints"), "--no-build", "--skip-unit"]
            def invoke(name, *extra):
                output = root / "runs" / name
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = runner.main(common + ["--output", str(output)] + list(extra))
                return code, json.loads((output / "summary.json").read_text())
            with mock.patch.object(runner, "REPO", root), mock.patch.object(
                runner, "environment_fingerprint", side_effect=lambda *args: dict(environment)
            ), mock.patch.object(runner, "source_metadata", return_value={"commit": "a" * 40, "dirty": True}):
                code, first = invoke("first")
                self.assertEqual(code, 0)
                self.assertEqual(first["counts"], {"RECORDED": 1})
                code, second = invoke("second")
                self.assertEqual(code, 0)
                self.assertEqual(second["counts"], {"PASS": 1})
                history = next(c for c in second["results"][0]["checks"] if c["kind"] == "history")
                self.assertEqual(Path(history["reference_run"]).name, "first")
                control.write_text(json.dumps({"loss": 1.1, "tps": 10}))
                code, failed = invoke("regressed")
                self.assertEqual(code, 1)
                history = next(c for c in failed["results"][0]["checks"] if c["kind"] == "history")
                self.assertEqual(history["loss"]["status"], "FAIL")
                self.assertEqual(history["performance"]["status"], "FAIL")
                control.write_text(json.dumps({"loss": 1.0, "tps": 100}))
                code, recovered = invoke("recovered")
                self.assertEqual(code, 0)
                history = next(c for c in recovered["results"][0]["checks"] if c["kind"] == "history")
                self.assertEqual(Path(history["reference_run"]).name, "second")
                result = recovered["results"][0]
                reproduce = shlex.split(result["reproduce"])
                self.assertEqual(Path(reproduce[reproduce.index("--reference-run") + 1]).name, "second")
                process = result["stages"][0]["process"]
                text = Path(process["log"]).read_text()
                self.assertIn("[GIT_COMMIT] " + "a" * 40, text)
                self.assertIn("[GIT_DIRTY] true", text)
                self.assertIn("[COMPLETED_AT]", text)
                self.assertIn("[EXIT_CODE] 0", text)
                self.assertEqual(json.loads(next(line[12:] for line in text.splitlines() if line.startswith("[ARGV_JSON] "))), process["argv"])
                self.assertFalse(list((root / "runs").rglob("*.npy")))
                manifest = json.loads((root / "runs/recovered/run.json").read_text())
                self.assertEqual(manifest["source"]["commit"], "a" * 40)
                self.assertTrue(manifest["started_at"])
                self.assertTrue(manifest["completed_at"])
                self.assertEqual(manifest["cases"][0]["commands"][0]["argv"], process["argv"])


if __name__ == "__main__":
    unittest.main()

"""Check integration tests using real artifact files and real runner evaluation."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checks
import runner
import tolerances
from test_checks import snapshot, npy


class RunnerCheckIntegration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = runner.Run.__new__(runner.Run)
        self.run.args = SimpleNamespace(
            alignment="deep",
            perf_environment="test-a100",
            record_history=False,
            loss_atol_fp32=1e-5,
            loss_atol_bf16=1e-2,
            performance_threshold=.20,
        )
        self.run.history = mock.Mock()
        self.run.history.compare.return_value = checks.result("PASS")
        self.run.assets = SimpleNamespace(hashes={})
        self.run.environment = {}
        self.run.done = {}
        self.run.case_map = {}
        self.case = {
            "id": "candidate",
            "dtype": "float32",
            "start_step": 1,
            "expected_steps": 2,
            "nproc_per_node": 1,
            "nthread_per_process": 1,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def metrics(self, last_loss=1):
        return checks.Metrics(
            [
                {"step": 1, "loss": 1, "lr": 0.001, "consumed_samples": 2},
                {"step": 2, "loss": last_loss, "lr": 0.001, "consumed_samples": 4},
            ],
            "float32",
        ).to_dict()

    def stage(self, *, reference=False, last_loss=1, snapshots=None):
        return {
            "reference": reference,
            "metrics": self.metrics(last_loss),
            "snapshot_dir": str(snapshots or self.root / "candidate"),
        }

    def assert_failed(self, outcomes):
        self.assertTrue(any(item["status"] != "PASS" for item in outcomes), outcomes)



    def test_backend_gradient_override_never_relaxes_parameters(self):
        ref, cand = self.root / "reference", self.root / "candidate"
        for step in range(3):
            for root in (ref, cand):
                snapshot(root / f"step_{step}", step=step)
                if step:
                    directory = root / f"step_{step}" / "rank_0"
                    values = [.1, .2] if root == ref else [.101, .2]
                    npy(directory / "gradient.npy", [2], values)
                    manifest = directory / "manifest.json"
                    document = json.loads(manifest.read_text())
                    document["tensors"].append({"kind": "gradient", "name": "weight", "shape": [2],
                                                "dtype": "float32", "path": "gradient.npy"})
                    manifest.write_text(json.dumps(document))
        self.run.done["reference"] = {"metrics": self.metrics(), "snapshot_dir": str(ref)}
        self.run.case_map["reference"] = dict(self.case)
        self.case["checks"] = [{"kind": "equivalent", "reference_case": "reference", "metrics": ["loss", "gradient", "parameter"]}]
        registry = copy.deepcopy(tolerances.load_registry())
        registry["policies"]["distributed_equivalence"]["backends"] = {
            "cuda": {"gradient": {"float32": {"atol": .01, "rtol": 0}}}}
        def apply(backend):
            self.case["resolved_tolerances"] = {"equivalent": {
                metric: tolerances.resolve("distributed_equivalence", "float32", metric, backend, registry)
                for metric in ("loss", "gradient", "parameter")}}
        apply("cpu")
        self.assert_failed(self.run.evaluate(self.case, [self.stage()]))
        apply("cuda")
        self.assertTrue(all(check["status"] == "PASS" for check in self.run.evaluate(self.case, [self.stage()])))
        npy(cand / "step_1" / "rank_0" / "tensor.npy", [2], [1.001, 2.])
        outcomes = self.run.evaluate(self.case, [self.stage()])
        self.assert_failed(outcomes)
        self.assertTrue(any("parameter" in reason for outcome in outcomes for reason in outcome.get("reasons", [])))

    def test_tp_gradient_policy_does_not_inherit_parameter_tolerance(self):
        ref, cand = self.root / "reference", self.root / "candidate"
        for root, ranks in ((ref, 1), (cand, 2)):
            for step in (0, 1):
                for rank in range(ranks):
                    snapshot(root / f"step_{step}", step=step, rank=rank)
                    directory = root / f"step_{step}" / f"rank_{rank}"
                    manifest = directory / "manifest.json"
                    document = json.loads(manifest.read_text())
                    document.update(dp_rank=0, tp_rank=rank, pp_rank=0, world_size=ranks)
                    if step:
                        npy(directory / "gradient.npy", [2], [.1 + rank * .001, .2])
                        document["tensors"].append({"kind": "gradient", "name": "weight", "shape": [2],
                                                    "dtype": "float32", "path": "gradient.npy"})
                    manifest.write_text(json.dumps(document))
        self.case["resolved_tolerances"] = {"equivalent": {
            "gradient": {"atol": .01, "rtol": 0}, "parameter": {"atol": 0, "rtol": 0}}}
        result = self.run.tensor_checks(self.case, "equivalent", ref, cand, steps=[1],
                                       required=["gradient", "parameter"], expected_world_sizes=(1, 2), check_initial=True)
        self.assertEqual(result["status"], "PASS", result)

    def test_tensor_equivalence_checks_intermediate_steps_and_initialization(self):
        ref, cand = self.root / "reference", self.root / "candidate"
        for step in range(3):
            snapshot(ref / f"step_{step}", step=step)
            snapshot(cand / f"step_{step}", step=step)
        self.run.done["reference"] = {
            "metrics": self.metrics(),
            "snapshot_dir": str(ref),
        }
        self.run.case_map["reference"] = self.case
        self.case["checks"] = [
            {
                "kind": "equivalent",
                "reference_case": "reference",
                "metrics": ["loss", "parameter"],
            }
        ]
        self.assertTrue(
            all(
                item["status"] == "PASS"
                for item in self.run.evaluate(self.case, [self.stage()])
            )
        )
        snapshot(cand / "step_1", values=(2.0, 1.0), step=1)
        self.assert_failed(self.run.evaluate(self.case, [self.stage()]))
        snapshot(cand / "step_1", step=1)
        snapshot(cand / "step_0", values=(2.0, 1.0), step=0)
        self.assert_failed(self.run.evaluate(self.case, [self.stage()]))

    def test_missing_rank_snapshot_fails_even_when_loss_matches(self):
        ref, cand = self.root / "reference", self.root / "candidate"
        for step in range(3):
            snapshot(ref / f"step_{step}", step=step)
            snapshot(cand / f"step_{step}", step=step)
        self.run.done["reference"] = {
            "metrics": self.metrics(),
            "snapshot_dir": str(ref),
        }
        self.run.case_map["reference"] = dict(self.case)
        self.case["nthread_per_process"] = 2
        self.case["checks"] = [
            {
                "kind": "equivalent",
                "reference_case": "reference",
                "metrics": ["loss", "parameter"],
            }
        ]
        self.assert_failed(self.run.evaluate(self.case, [self.stage()]))

    def test_lora_requires_unchanged_base_and_changed_adapter(self):
        root = self.root / "candidate"
        for step in (0, 2):
            directory = snapshot(root / f"step_{step}", step=step) / "rank_0"
            npy(directory / "adapter.npy", [2], [0.1, 0.2 if step == 0 else 0.3])
            document = json.loads((directory / "manifest.json").read_text())
            document["tensors"].append(
                {
                    "name": "layer.lora_A",
                    "kind": "parameter",
                    "shape": [2],
                    "path": "adapter.npy",
                }
            )
            (directory / "manifest.json").write_text(json.dumps(document))
        self.case["checks"] = [{"kind": "lora_frozen"}]
        self.assertTrue(
            all(item["status"] == "PASS" for item in self.run.evaluate(self.case, [self.stage()]))
        )
        npy(root / "step_2" / "rank_0" / "tensor.npy", [2], [2.0, 1.0])
        self.assert_failed(self.run.evaluate(self.case, [self.stage()]))





    def test_resume_wrong_data_cursor_fails_even_when_loss_matches(self):
        self.run.done["reference"] = {"metrics": self.metrics()}
        self.case.update(
            start_step=2,
            checks=[
                {
                    "kind": "resume",
                    "reference_case": "reference",
                    "metrics": ["loss", "lr", "consumed_samples"],
                }
            ],
        )
        stage = self.stage()
        stage["metrics"]["records"] = stage["metrics"]["records"][1:]
        stage["metrics"]["records"][0]["consumed_samples"] = 6
        self.assert_failed(self.run.evaluate(self.case, [stage]))


if __name__ == "__main__":
    unittest.main()

"""Adversarial artifact tests: regressions must not become empty green checks."""

import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks


def npy(path, shape, values, descriptor="<f4"):
    header = repr({"descr": descriptor, "fortran_order": False, "shape": tuple(shape)})
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    path.write_bytes(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header.encode()
        + struct.pack(
            descriptor[0]
            + {"f4": "f", "f8": "d", "i8": "q", "i4": "i", "u8": "Q"}[descriptor[1:]]
            * len(values),
            *values,
        )
    )


def snapshot(
    root,
    values=(1.0, 2.0),
    *,
    name="weight",
    kind="parameter",
    rank=0,
    shard=None,
    step=1,
    shape=None,
):
    directory = root / f"rank_{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    shape = list(shape or [len(values)])
    npy(directory / "tensor.npy", shape, values)
    tensor = {
        "name": name,
        "kind": kind,
        "shape": shape,
        "path": "tensor.npy",
        "dtype": "float32",
    }
    if shard:
        tensor["shard"] = shard
    (directory / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "step": step, "rank": rank, "tensors": [tensor]}
        )
    )
    return root


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def log(self, text, name="metrics.log"):
        path = self.root / name
        path.write_text(text)
        return path

    def test_empty_metrics_is_failure(self):
        for text in ("", "training started\n"):
            with self.assertRaises(checks.MetricError):
                checks.load_metrics(self.log(text))

    def test_nan_inf_and_scientific(self):
        for token in ("NaN", "-inf", "+Infinity"):
            with self.assertRaises(checks.MetricError):
                checks.load_metrics(self.log(f"step 1/1 | train loss {token}\n"))
        run = checks.load_metrics(
            self.log(
                "step 1/1 | train loss -2.12e-3 | lr 2E-5 | (1.2e1 ms | 3e2 tok/s)"
            )
        )
        self.assertEqual(run.records[0]["loss"], -0.00212)
        self.assertEqual(run.records[0]["tokens_per_second"], 300)

    def test_truncated_missing_and_duplicate_steps(self):
        for steps in ((1, 2), (1, 3), (1, 2, 2, 3), (3, 2, 1)):
            with self.assertRaises(checks.MetricError):
                checks.load_metrics(
                    self.log(
                        "\n".join(f"step {step}/3 | train loss 1.0" for step in steps)
                    )
                )

    def test_resumed_explicit_interval(self):
        path = self.log("step 3/4 | train loss 1\nstep 4/4 | train loss 1")
        self.assertEqual(
            len(checks.load_metrics(path, expected_steps=range(3, 5)).records), 2
        )
        with self.assertRaises(checks.MetricError):
            checks.load_metrics(path)

    def test_runtime_optimizer_boolean_metadata_is_not_a_metric(self):
        row = {
            "schema_version": 1,
            "type": "step",
            "model": "gpt2",
            "step": 1,
            "dtype": "float32",
            "loss": 4.8,
            "lr": 0.0001,
            "global_tokens": 32,
            "step_ms": 3.5,
            "peak_memory_bytes": 1048576,
            "consumed_samples": 2,
            "rank": 0,
            "dp": 1,
            "tp": 1,
            "pp": 1,
            "sp": 1,
            "vpp": 1,
            "dp_rank": 0,
            "tp_rank": 0,
            "pp_rank": 0,
            "world_size": 1,
            "optimizer_type": "sgd",
            "optimizer_stateful": False,
        }
        path = self.log(json.dumps(row), "optimizer.jsonl")
        self.assertEqual(
            checks.load_metrics(path).records[0]["optimizer_stateful"], False
        )
        row["optimizer_stateful"] = 0
        with self.assertRaises(checks.MetricError):
            checks.load_metrics(self.log(json.dumps(row), "optimizer.jsonl"))

    def test_dtype_metadata_conflict(self):
        path = self.log('{"step":1,"dtype":"bfloat16","loss":1}', "metrics.jsonl")
        with self.assertRaises(checks.MetricError):
            checks.load_metrics(path, dtype="float32")
        run = checks.load_metrics(path)
        result = checks.compare_metrics(run, run, dtype="float32")
        self.assertEqual(result["status"], "FAIL")

    def test_json_tail_corruption(self):
        path = self.log('{"step":1,"loss":1}\n{"step":', "metrics.jsonl")
        with self.assertRaises(checks.MetricError):
            checks.load_metrics(path)

    def test_absolute_and_relative_tolerance(self):
        a = checks.Metrics([{"step": 1, "loss": 100.0}], "float32")
        b = checks.Metrics([{"step": 1, "loss": 100.05}], "float32")
        self.assertEqual(
            checks.compare_metrics(a, b, dtype="float32", atol=0.001, rtol=0.001)[
                "status"
            ],
            "PASS",
        )
        self.assertEqual(
            checks.compare_metrics(a, b, dtype="float32", atol=0.001, rtol=0)["status"],
            "FAIL",
        )
        b.records[0]["loss"] = float("nan")
        self.assertEqual(
            checks.compare_metrics(a, b, dtype="float32")["status"], "FAIL"
        )

    def test_both_missing_and_integer_mismatch_fail(self):
        a = checks.Metrics([{"step": 1, "loss": 1, "consumed_samples": 1000000}])
        b = checks.Metrics([{"step": 1, "loss": 1, "consumed_samples": 1000001}])
        self.assertEqual(
            checks.compare_metrics(
                a, b, dtype="bfloat16", metrics=("consumed_samples",)
            )["status"],
            "FAIL",
        )
        self.assertEqual(
            checks.compare_metrics(a, a, dtype="float32", metrics=("lr",))["status"],
            "FAIL",
        )

    def distributed(self, *, complete=True, timing=10, loss=2.0, rank=0):
        prefix = self.root / "metrics"
        row = {
            "step": 1,
            "rank": rank,
            "dtype": "float32",
            "loss": loss,
            "lr": 0.001,
            "global_tokens": 32,
            "step_ms": timing,
            "consumed_samples": 2,
            "peak_memory_bytes": 1024,
            "pp": 1,
            "pp_rank": 0,
        }
        text = json.dumps(row) + "\n"
        if complete:
            text += json.dumps({"type": "complete", "step": 1, "rank": rank}) + "\n"
        Path(str(prefix) + f".rank_{rank}.jsonl").write_text(text)
        return prefix

    def test_all_ranks_must_complete(self):
        prefix = self.distributed()
        self.distributed(rank=1, complete=False)
        with self.assertRaises(checks.MetricError):
            checks.load_distributed_metrics(
                prefix, expected_steps=[1], dtype="float32", world_size=2
            )
        self.distributed(rank=1, timing=20)
        run = checks.load_distributed_metrics(
            prefix, expected_steps=[1], dtype="float32", world_size=2
        )
        self.assertEqual(run.records[0]["step_ms"], 20)
        self.assertEqual(run.records[0]["tokens_per_second"], 1600)

    def test_rank_reduction_loss_disagreement(self):
        prefix = self.distributed()
        self.distributed(rank=1, loss=3)
        with self.assertRaises(checks.MetricError):
            checks.load_distributed_metrics(
                prefix, expected_steps=[1], dtype="float32", world_size=2
            )

    def test_same_norm_different_tensor_fails(self):
        left = snapshot(self.root / "a", values=(1.0, 2.0))
        right = snapshot(self.root / "b", values=(2.0, 1.0))
        checked = checks.compare_snapshots(
            left, right, dtype="float32", required_kinds=("parameter",)
        )
        self.assertEqual(checked["status"], "FAIL")

    def test_packed_tp_reconstruction(self):
        left = snapshot(self.root / "a", values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        right = snapshot(
            self.root / "b",
            values=(1.0, 3.0, 5.0),
            shard={"axis": 0, "index": 0, "count": 2, "part_sizes": [2, 2, 2]},
        )
        snapshot(
            right,
            values=(2.0, 4.0, 6.0),
            rank=1,
            shard={"axis": 0, "index": 1, "count": 2, "part_sizes": [2, 2, 2]},
        )
        self.assertEqual(
            checks.compare_snapshots(
                left, right, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "PASS",
        )
        (right / "rank_1" / "manifest.json").unlink()
        self.assertEqual(
            checks.compare_snapshots(
                left, right, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "FAIL",
        )

    def test_nonleading_axis_reconstruction(self):
        left = snapshot(
            self.root / "a",
            values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
            shape=[2, 4],
        )
        right = snapshot(
            self.root / "b",
            values=(1.0, 2.0, 5.0, 6.0),
            shape=[2, 2],
            shard={"axis": 1, "index": 0, "count": 2},
        )
        snapshot(
            right,
            values=(3.0, 4.0, 7.0, 8.0),
            shape=[2, 2],
            rank=1,
            shard={"axis": 1, "index": 1, "count": 2},
        )
        self.assertEqual(
            checks.compare_snapshots(
                left, right, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "PASS",
        )

    def test_corrupt_tensor_and_replica(self):
        left = snapshot(self.root / "a")
        right = snapshot(self.root / "b")
        snapshot(right, rank=1, values=(1.0, 3.0))
        self.assertEqual(
            checks.compare_snapshots(
                left, right, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "FAIL",
        )
        (left / "rank_0" / "tensor.npy").write_bytes(b"broken")
        self.assertEqual(
            checks.compare_snapshots(
                left, left, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "FAIL",
        )

    def test_stateless_sgd_requires_explicit_contract(self):
        left, right = snapshot(self.root / "left"), snapshot(self.root / "right")
        kwargs = dict(dtype="float32", required_kinds=("parameter", "optimizer"))
        self.assertEqual(
            checks.compare_snapshots(left, right, **kwargs)["status"], "FAIL"
        )
        for directory in (left, right):
            path = directory / "rank_0" / "manifest.json"
            document = json.loads(path.read_text())
            document.update(optimizer_type="sgd", optimizer_stateful=False)
            path.write_text(json.dumps(document))
        self.assertEqual(
            checks.compare_snapshots(left, right, **kwargs)["status"], "PASS"
        )
        path = right / "rank_0" / "manifest.json"
        document = json.loads(path.read_text())
        document.update(optimizer_type="adam", optimizer_stateful=True)
        path.write_text(json.dumps(document))
        self.assertEqual(
            checks.compare_snapshots(left, right, **kwargs)["status"], "FAIL"
        )

    def bundle(
        self, root, values=(1.0, 2.0), *, descriptor="<f4", dtype="float32", second=True
    ):
        directory = root / "rank_0"
        directory.mkdir(parents=True, exist_ok=True)
        npy(directory / "temporary.npy", [len(values)], values, descriptor)
        record = (directory / "temporary.npy").read_bytes()
        (directory / "temporary.npy").unlink()
        tensors = [
            {
                "kind": "parameter",
                "name": "weight",
                "shape": [len(values)],
                "dtype": dtype,
                "path": "tensors.bin",
                "offset": 0,
                "length": len(record),
            }
        ]
        if second:
            tensors.append(dict(tensors[0], name="other", offset=len(record)))
        (directory / "tensors.bin").write_bytes(record * (2 if second else 1))
        (directory / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "rank": 0, "step": 1, "tensors": tensors})
        )
        return root

    def edit_bundle(self, root, mutation):
        path = root / "rank_0" / "manifest.json"
        document = json.loads(path.read_text())
        mutation(document)
        path.write_text(json.dumps(document))

    def test_bundle_read_once_and_legacy_compatibility(self):
        bundled = self.bundle(self.root / "bundle", second=False)
        legacy = snapshot(self.root / "legacy")
        original = Path.read_bytes
        with mock.patch.object(
            Path, "read_bytes", autospec=True, side_effect=original
        ) as reads:
            checks._load_snapshot(bundled)
            self.assertEqual(
                sum(
                    call.args[0].name == "tensors.bin" for call in reads.call_args_list
                ),
                1,
            )
        self.assertEqual(
            checks.compare_snapshots(
                legacy, bundled, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "PASS",
        )

    def test_bundle_resolves_each_unique_path_once(self):
        bundled = self.bundle(self.root / "bundle")
        path = bundled / "rank_0" / "manifest.json"
        document = json.loads(path.read_text())
        first = document["tensors"][0]
        # Hundreds of tensor metadata entries can intentionally alias one NPY.
        document["tensors"] += [
            dict(first, name=f"alias_{index}", alias_of="weight")
            for index in range(200)
        ]
        path.write_text(json.dumps(document))
        original = Path.resolve
        with mock.patch.object(
            Path, "resolve", autospec=True, side_effect=original
        ) as resolutions:
            checks._load_snapshot(bundled)
            self.assertEqual(
                sum(
                    call.args[0].name == "tensors.bin"
                    for call in resolutions.call_args_list
                ),
                1,
            )
            self.assertEqual(
                sum(
                    call.args[0].name == "rank_0" for call in resolutions.call_args_list
                ),
                1,
            )

    def test_cached_bundle_paths_still_reject_traversal_and_symlink_escape(self):
        bundled = self.bundle(self.root / "bundle")
        outside = self.root / "outside.bin"
        outside.write_bytes((bundled / "rank_0" / "tensors.bin").read_bytes())
        for raw_path in ("../../outside.bin", "escape.bin"):
            bundled = self.bundle(self.root / "bundle")
            symlink = bundled / "rank_0" / "escape.bin"
            if not symlink.exists():
                symlink.symlink_to(outside)
            self.edit_bundle(
                bundled, lambda doc: doc["tensors"][0].update(path=raw_path)
            )
            with self.assertRaisesRegex(checks.MetricError, "escapes"):
                checks._load_snapshot(bundled)

    def test_bundle_truncation_and_invalid_offsets(self):
        left = self.bundle(self.root / "left")
        kwargs = dict(dtype="float32", required_kinds=("parameter",))
        for offset, length in ((-1, 20), (True, 20), (0, -1), (0, 10**9)):
            right = self.bundle(self.root / "right")
            self.edit_bundle(
                right,
                lambda doc: doc["tensors"][0].update(offset=offset, length=length),
            )
            self.assertEqual(
                checks.compare_snapshots(left, right, **kwargs)["status"], "FAIL"
            )
        right = self.bundle(self.root / "right")
        path = right / "rank_0" / "tensors.bin"
        path.write_bytes(path.read_bytes()[:-1])
        self.assertEqual(
            checks.compare_snapshots(left, right, **kwargs)["status"], "FAIL"
        )

    def test_bundle_overlap_only_allowed_for_declared_alias(self):
        root = self.bundle(self.root / "bundle")
        self.edit_bundle(root, lambda doc: doc["tensors"][1].update(offset=0))
        with self.assertRaisesRegex(checks.MetricError, "Overlapping"):
            checks._load_snapshot(root)
        self.edit_bundle(root, lambda doc: doc["tensors"][1].update(alias_of="weight"))
        document, tensors = checks._load_snapshot(root)
        self.assertEqual(
            tensors[("parameter", "weight")], tensors[("parameter", "other")]
        )

    def test_bundle_manifest_shape_and_dtype_must_match_payload(self):
        left = self.bundle(self.root / "left")
        for patch in ({"dtype": "float64"}, {"shape": [1, 2]}):
            right = self.bundle(self.root / "right")
            self.edit_bundle(right, lambda doc: doc["tensors"][0].update(patch))
            self.assertEqual(
                checks.compare_snapshots(
                    left, right, dtype="float32", required_kinds=("parameter",)
                )["status"],
                "FAIL",
            )

    def test_integer_snapshot_comparison_is_exact_above_float_precision(self):
        left = self.bundle(
            self.root / "left",
            values=(2**60,),
            descriptor="<i8",
            dtype="int64",
            second=False,
        )
        right = self.bundle(
            self.root / "right",
            values=(2**60 + 1,),
            descriptor="<i8",
            dtype="int64",
            second=False,
        )
        checked = checks.compare_snapshots(
            left, right, dtype="bfloat16", required_kinds=("parameter",)
        )
        self.assertEqual(checked["status"], "FAIL")
        self.assertEqual(checked["data"]["max_absolute_error"], 1)
        self.assertIn(str(2**60 + 1), checked["reasons"][0])

    def test_float64_preserved_and_different_storage_dtype_rejected(self):
        left = self.bundle(
            self.root / "left",
            values=(1.0,),
            descriptor="<f8",
            dtype="float64",
            second=False,
        )
        right = self.bundle(
            self.root / "right",
            values=(1.0 + 2**-40,),
            descriptor="<f8",
            dtype="float64",
            second=False,
        )
        self.assertEqual(
            checks.compare_snapshots(
                left,
                right,
                dtype="float32",
                atol=0,
                rtol=0,
                required_kinds=("parameter",),
            )["status"],
            "FAIL",
        )
        right = self.bundle(self.root / "right", values=(1.0,), second=False)
        self.assertEqual(
            checks.compare_snapshots(
                left, right, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "FAIL",
        )

    def replicated(self, root, values, *, across_tp):
        for rank, value in enumerate(values):
            snapshot(root, values=(value,), rank=rank)
            path = root / f"rank_{rank}" / "manifest.json"
            document = json.loads(path.read_text())
            document.update(
                world_size=len(values),
                dp_rank=0 if across_tp else rank,
                tp_rank=rank if across_tp else 0,
                pp_rank=0,
            )
            path.write_text(json.dumps(document))
        return root

    def test_tp_roundoff_uses_existing_tolerance_but_dp_stays_exact(self):
        reference = snapshot(self.root / "reference", values=(1.0,))
        candidate = self.replicated(
            self.root / "tp", [1.0, 1.0 + 2**-23], across_tp=True
        )
        self.assertEqual(
            checks.compare_snapshots(
                reference, candidate, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "PASS",
        )
        dp = self.replicated(self.root / "dp", [1.0, 1.0 + 2**-23], across_tp=False)
        self.assertEqual(
            checks.compare_snapshots(
                reference, dp, dtype="float32", required_kinds=("parameter",)
            )["status"],
            "FAIL",
        )
        self.assertEqual(
            checks.compare_snapshots(
                reference,
                candidate,
                dtype="float32",
                atol=0,
                rtol=0,
                required_kinds=("parameter",),
            )["status"],
            "FAIL",
        )

    def test_every_tp_replica_compared_directly_with_reference(self):
        reference = snapshot(self.root / "reference", values=(0.0,))
        candidate = self.replicated(self.root / "tp", [0.75e-5, 1.5e-5], across_tp=True)
        # Replicas are close to each other, but rank 1 is outside reference tolerance.
        checks._load_snapshot(candidate, replica_tolerance=(1e-5, 0))
        result = checks.compare_snapshots(
            reference,
            candidate,
            dtype="float32",
            atol=1e-5,
            rtol=0,
            required_kinds=("parameter",),
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("replica 1" in reason for reason in result["reasons"]))
        self.replicated(candidate, [0.0, 1.0], across_tp=True)
        with self.assertRaisesRegex(checks.MetricError, "exceed configured tolerance"):
            checks._load_snapshot(candidate, replica_tolerance=(1e-5, 0))

    def test_resume_requires_state(self):
        ref = checks.Metrics(
            [
                {"step": s, "loss": 1, "lr": 0.001, "consumed_samples": s * 2}
                for s in range(1, 5)
            ]
        )
        resumed = checks.Metrics(ref.records[2:])
        self.assertEqual(
            checks.compare_resume(ref, resumed, dtype="float32", resume_step=2)[
                "status"
            ],
            "FAIL",
        )

    def test_lr_oracle_boundary(self):
        args = {
            "learning_rate": 0.1,
            "min_lr": 0.0,
            "lr_warmup_iters": 2,
            "lr_decay_iters": 4,
            "num_iteration": 5,
            "lr_decay_style": "linear",
        }
        expected = [0.0, 0.05, 0.1, 0.05, 0.0]
        run = checks.Metrics(
            [{"step": i + 1, "lr": value} for i, value in enumerate(expected)]
        )
        self.assertEqual(checks.check_lr_schedule(run, args)["status"], "PASS")
        run.records[2]["lr"] = 0.05
        self.assertEqual(checks.check_lr_schedule(run, args)["status"], "FAIL")


class PerformanceTests(unittest.TestCase):
    def runs(self, times=(10, 10, 10), memory=100):
        return [
            checks.Metrics(
                [
                    {
                        "step": step,
                        "step_ms": time,
                        "global_tokens": 32,
                        "peak_memory_bytes": memory,
                    }
                    for step in range(1, 7)
                ]
            )
            for time in times
        ]

    def compare(self, ref, cand, **kwargs):
        return checks.compare_performance(ref, cand, warmup=2, window=4, **kwargs)

    def test_fast_is_pass_slow_is_fail(self):
        ref = self.runs()
        self.assertEqual(self.compare(ref, self.runs((8, 8, 8)))["status"], "PASS")
        self.assertEqual(self.compare(ref, self.runs((12, 12, 12)))["status"], "FAIL")

    def test_noise_not_pass(self):
        self.assertEqual(
            self.compare(self.runs(), self.runs((9, 10, 11)))["status"], "INCONCLUSIVE"
        )

    def test_memory_and_missing_measurement(self):
        self.assertEqual(
            self.compare(self.runs(), self.runs(memory=120))["status"], "FAIL"
        )
        self.assertEqual(
            self.compare(self.runs(), self.runs((10, 10)))["status"], "FAIL"
        )
        missing = self.runs()
        missing[0].records.pop()
        self.assertEqual(self.compare(self.runs(), missing)["status"], "FAIL")

    def test_warmup_is_excluded(self):
        candidate = self.runs()
        for run in candidate:
            run.records[0]["step_ms"] = 10000
        self.assertEqual(self.compare(self.runs(), candidate)["status"], "PASS")


class MemoryGrowthTests(unittest.TestCase):
    def check_values(self, values):
        metrics = checks.Metrics(
            [
                {"step": step + 1, "peak_memory_bytes": value}
                for step, value in enumerate(values)
            ]
        )
        return checks.check_memory_growth(
            metrics, warmup=2, window=4, max_growth_bytes=10, max_growth_fraction=0.01
        )

    def test_stable_and_warmup_growth_pass(self):
        self.assertEqual(self.check_values([10, 20] + [100] * 12)["status"], "PASS")

    def test_sustained_growth_fails(self):
        self.assertEqual(
            self.check_values([10, 20] + list(range(100, 160, 5)))["status"], "FAIL"
        )

    def test_single_allocator_jump_is_inconclusive(self):
        self.assertEqual(
            self.check_values([10, 20] + [100] * 6 + [150] * 6)["status"],
            "INCONCLUSIVE",
        )

    def test_too_short_or_zero_is_not_pass(self):
        self.assertEqual(self.check_values([100] * 6)["status"], "FAIL")
        self.assertEqual(self.check_values([0] * 14)["status"], "FAIL")




class LegacyCLITests(unittest.TestCase):
    def test_false_green_cases_fail_exit(self):
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.mkdir()
            b.mkdir()
            for script in ("compare_loss.py", "compare_tps.py"):
                for content in ("", "step 1/1 | train loss NaN | 1 tok/s"):
                    (a / "case.log").write_text(content)
                    (b / "case.log").write_text(content)
                    output = subprocess.run(
                        [sys.executable, str(scripts / script), str(a), str(b)],
                        capture_output=True,
                    )
                    self.assertNotEqual(output.returncode, 0, output.stdout)
                (b / "case.log").unlink()
                output = subprocess.run(
                    [sys.executable, str(scripts / script), str(a), str(b)],
                    capture_output=True,
                )
                self.assertNotEqual(output.returncode, 0)

    def test_bf16_alias_uses_bf16_threshold(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from compare_loss import get_dtype_from_filename

        self.assertEqual(
            get_dtype_from_filename("qwen3_3d_ddp_tp2_pp2_bf16.log"), "bfloat16"
        )


if __name__ == "__main__":
    unittest.main()

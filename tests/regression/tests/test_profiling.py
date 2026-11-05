"""Profiling workloads, build isolation, and real profiler text contracts."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import planning
import profiling
import runner
import execution


class ProfilingTests(unittest.TestCase):
    def test_builder_uses_explicit_profiling_and_unit_test_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for enabled in (False, True):
                with mock.patch.object(execution, "source_metadata", return_value={"commit": "abc"}), mock.patch.object(
                    execution, "run_process", return_value={"status": "FAIL"}
                ) as run:
                    execution.build_project(root, root / "build", "cpu", root / "logs",
                                            profile_mode=enabled, build_tests=not enabled)
                    argv = run.call_args.args[0]
                    self.assertIn("-DPROFILE_MODE=" + ("ON" if enabled else "OFF"), argv)
                    self.assertIn("-DBUILD_TEST=" + ("OFF" if enabled else "ON"), argv)
                    self.assertEqual(sum(a.startswith("-DPROFILE_MODE=") for a in argv), 1)

    def test_perf_is_only_basic_and_ddp_in_two_precisions(self):
        plan = planning.load_plan("perf", alignment="history")
        self.assertTrue(plan["suite_metadata"]["profile_mode"])
        self.assertEqual(len(plan["cases"]), 14)
        self.assertEqual(len({c["id"] for c in plan["cases"]}), 14)
        for case in plan["cases"]:
            self.assertIn(case["dtype"], ("float32", "bfloat16"))
            self.assertEqual(case.get("repetitions", 1), 1)
            self.assertEqual(case["expected_steps"], 10)
            topology, args = case["topology"], case["resolved_args"]
            self.assertEqual((topology["tp"], topology["pp"], topology["vpp"]), (1, 1, 1))
            self.assertFalse(args.get("sequence_parallel", False))
            self.assertFalse(args.get("activation_recompute", False))
            self.assertEqual(args.get("zero_stage", 0), 0)
            self.assertEqual(args.get("lora_rank", 0), 0)
            self.assertEqual(topology["dp"], 2 if case["scenario"] == "kernel_ddp" else 1)
            if case["model"] == "mixtral":
                self.assertEqual(case["scenario"], "kernel_basic")
        cpu = planning.load_plan("perf", backend="cpu", alignment="history")
        self.assertEqual(len(cpu["cases"]), 4)
        release = planning.load_plan("release", alignment="history")
        self.assertEqual(len(release["cases"]), 303)
        self.assertFalse(any("kernel" in c["tags"] for c in release["cases"]))

    def test_ddp_keeps_same_per_device_workload(self):
        cases = planning.load_plan("perf")["cases"]
        basic = {(c["model"], c["dtype"]): c for c in cases if c["scenario"] == "kernel_basic"}
        for case in cases:
            if case["scenario"] != "kernel_ddp":
                continue
            base = basic[case["model"], case["dtype"]]
            for key in ("micro_batch_samples", "sequence_length", "accumulation_steps"):
                self.assertEqual(case["batch"][key], base["batch"][key])
            self.assertEqual(case["fixtures"], base["fixtures"])

    def test_default_build_is_separate_and_explicit_path_is_honored(self):
        plan = planning.load_plan("perf")
        args = SimpleNamespace(build_dir=Path("/build/cuda"), build_dir_explicit=False)
        runner._select_build_mode(args, plan)
        self.assertEqual(args.build_dir, Path("/build/cuda-profile"))
        args = SimpleNamespace(build_dir=Path("/custom"), build_dir_explicit=True)
        runner._select_build_mode(args, plan)
        self.assertEqual(args.build_dir, Path("/custom"))

    def test_no_build_requires_verified_profiling_binary(self):
        environment = {"build_options": {"PROFILE_MODE": "ON"}}
        provenance = {"verified": True, "manifest": {"environment": environment}}
        profiling.check_build_mode(environment, provenance, True)
        for env, prov in (({}, provenance), (environment, {}),
                          (environment, {"verified": True, "manifest": {"environment": {}}})):
            with self.assertRaisesRegex(ValueError, "PROFILE_MODE=ON"):
                profiling.check_build_mode(env, prov, True)
        with self.assertRaisesRegex(ValueError, "PROFILE_MODE=OFF"):
            profiling.check_build_mode(environment, provenance, False)

    def make_files(self, root, ranks=(0, 1)):
        for rank in ranks:
            (root / f"gpt2.report.rank{rank}").write_text(
                f"=== Profiler Report for Rank {rank} ===\nDevice Total(us)\n")
            (root / f"gpt2.records.log.rank{rank}").write_text(
                f"=== Kernel Call Log for Rank {rank} ===\n"
                "Tag: Untagged\n0 2026-10-16 01:00:00 Init cuda 999 999 20\n"
                "Tag: Step_0\n0 2026-10-16 01:00:00 Matmul cuda 100 80 20\n"
                "Tag: Step_1\n0 2026-10-16 01:00:00 Matmul cuda 20 10 20\n"
                "1 2026-10-16 01:00:00 Matmul cuda 30 20 20\n")

    def test_collects_all_devices_and_excludes_first_step_from_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_files(root)
            result = profiling.collect(root, "gpt2", 2, 1, 2)
            self.assertEqual(len(result["operators"]), 2)
            for row in result["operators"]:
                self.assertEqual(row["count"], 2)
                self.assertEqual(row["device_total_us"], 30)
                self.assertEqual(row["device_mean_us"], 15)
                self.assertEqual(row["host_mean_us"], 25)
            self.assertEqual(len(result["files"]), 4)
            self.assertEqual(json.loads((root / "profiles.json").read_text()), result)

    def test_missing_empty_or_truncated_profiler_output_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                profiling.collect(root, "gpt2", 1, 1, 2)
            self.make_files(root, (0,))
            with self.assertRaises(ValueError):
                profiling.collect(root, "gpt2", 2, 1, 2)
            records = root / "gpt2.records.log.rank0"
            for value in ("", "Tag: Step_0\n0 2026-10-16 01:00:00 Op cpu 10 0 0\n", "broken"):
                records.write_text(value)
                with self.assertRaises(ValueError):
                    profiling.collect(root, "gpt2", 1, 1, 2)


if __name__ == "__main__":
    unittest.main()

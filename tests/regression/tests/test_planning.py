"""Planning tests use the original shell configuration as migration oracle."""

import copy
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning import (
    ROOT,
    PlanError,
    batch_args,
    load_plan,
    pair_keys,
    pairwise,
    validate_case,
)


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = load_plan("nightly")
        cls.classic = [
            c for c in cls.plan["cases"] if c["scenario"].startswith("classic_")
        ]
        cls.old = json.loads((ROOT.parents[1] / "scripts/test_config.json").read_text())

    def test_every_original_command_preserved(self):
        by_id = {c["id"]: c for c in self.classic}
        self.assertEqual(
            Counter(c["model"] for c in self.classic),
            {"gpt2": 86, "llama3": 86, "qwen3": 28, "mixtral": 12},
        )
        count = 0
        for model in ["gpt2", "llama3", "qwen3", "mixtral"]:
            groups = self.old["variables"][model.upper() + "_TEST_GROUPS"].split(",")
            for group in self.old["test_groups"]:
                if group["tag"] not in groups:
                    continue
                for test in group["tests"]:
                    cid = "classic_%s/%s/%s" % (group["tag"], model, test["id"])
                    case = by_id[cid]
                    count += 1
                    expected = copy.deepcopy(test["args"])
                    for key, value in list(expected.items()):
                        if isinstance(value, str) and "LORA_WEIGHTS_DIR" in value:
                            expected[key] = "fixture:classic.%s.lora" % model
                        elif key == "save":
                            expected[key] = "artifact:%s/checkpoints" % cid
                        elif key == "load":
                            dep = case["dependencies"][0]
                            expected[key] = "artifact:%s/checkpoints/%s" % (
                                dep,
                                "iter_%07d" % int(value.rsplit("_", 1)[1]),
                            )
                            original_source = next(
                                t
                                for t in group["tests"]
                                if t["id"] == by_id[dep]["legacy_id"]
                            )
                            self.assertEqual(
                                original_source["args"]["save"], value.rsplit("/", 1)[0]
                            )
                    actual = {
                        k: v
                        for k, v in case["args"].items()
                        if k not in ["input_bin", "llmc_filepath", "device"]
                    }
                    self.assertEqual(actual, expected, cid)
                    self.assertEqual(
                        case["nproc_per_node"],
                        group.get("infini_run_args", {}).get("nproc_per_node", 1),
                    )
                    self.assertEqual(
                        case["nthread_per_process"],
                        test["args"].get("nthread_per_process", 1),
                    )
                    self.assertEqual(
                        case["gpus"],
                        case["nproc_per_node"] * case["nthread_per_process"],
                    )
        self.assertEqual(count, 198)

    def test_no_machine_paths_and_resumes_topological(self):
        positions = {c["id"]: i for i, c in enumerate(self.plan["cases"])}
        for case in self.classic:
            for value in case["args"].values():
                if isinstance(value, str):
                    self.assertFalse(value.startswith("/"), case["id"])
            for dep in case["dependencies"]:
                self.assertLess(positions[dep], positions[case["id"]])
            if "load" in case["args"]:
                self.assertEqual(case["start_step"], 31)
                self.assertEqual(case["expected_steps"], 50)


class PlanningTests(unittest.TestCase):
    def test_all_suites_and_deterministic_ids(self):
        for suite in ["smoke", "pr", "nightly", "perf", "release"]:
            first = load_plan(suite)
            second = load_plan(suite)
            self.assertEqual(first, second)
            self.assertEqual(
                len({c["id"] for c in first["cases"]}), len(first["cases"])
            )
            self.assertTrue(all(len(c["config_hash"]) == 64 for c in first["cases"]))

    def test_all_legal_pairs_and_high_risk_points(self):
        plan = load_plan("pr")
        reports = [
            r for r in plan["coverage"]["matrices"] if r["algorithm"] == "greedy-v1"
        ]
        self.assertTrue(reports)
        for report in reports:
            self.assertFalse(report["uncovered_pairs"])
            self.assertEqual(report["required_pairs"], report["covered_pairs"])
            self.assertGreater(
                report["legal_candidates"], report["selected_candidates"]
            )
            self.assertFalse(report["unavailable_high_risk"])
            chosen = [
                c["factors"]
                for c in plan["cases"]
                if c["scenario"] == report["scenario"] and "reference" not in c["tags"]
            ]
            for anchor in report["high_risk"]:
                self.assertIn(anchor, chosen)

    def test_pairwise_constraints_define_obligations(self):
        rows = [
            {"a": 0, "b": 0, "c": 0},
            {"a": 0, "b": 1, "c": 1},
            {"a": 1, "b": 0, "c": 1},
        ]
        selected, required = pairwise(rows)
        self.assertEqual(set(required), set().union(*(pair_keys(r) for r in selected)))
        self.assertNotIn('[ ["a",1],["b",1] ]'.replace(" ", ""), required)
        with self.assertRaises(PlanError):
            pairwise(rows, [{"a": 1, "b": 1, "c": 1}])

    def test_filter_keeps_references_and_reports_gaps(self):
        plan = load_plan("pr", models="gpt2", tags="zero2")
        self.assertTrue(plan["coverage"]["partial"])
        ids = {c["id"] for c in plan["cases"]}
        self.assertTrue(
            any(c["selection_reason"] == ["dependency closure"] for c in plan["cases"])
        )
        for case in plan["cases"]:
            self.assertTrue(set(case["dependencies"]) <= ids)
        self.assertTrue(any(r["uncovered_pairs"] for r in plan["coverage"]["matrices"]))
        with self.assertRaises(PlanError):
            load_plan("pr", models="missing")

    def test_resume_filter_adds_continuous_source(self):
        plan = load_plan("nightly", cases="classic_checkpoint/gpt2/ckpt_resume")
        self.assertEqual(len(plan["cases"]), 2)
        self.assertEqual(plan["cases"][1]["dependencies"], [plan["cases"][0]["id"]])

    def test_cpu_exclusions_explicit(self):
        plan = load_plan("pr", backend="cpu")
        self.assertTrue(all(c["gpus"] == 0 for c in plan["cases"]))
        self.assertTrue(
            any(r["unavailable_high_risk"] for r in plan["coverage"]["matrices"])
        )
        self.assertTrue(
            any(
                "GPU backend" in e["reason"]
                for r in plan["coverage"]["matrices"]
                for e in r["excluded"]
            )
        )

    def test_batch_units(self):
        b = {"micro_batch_samples": 2, "sequence_length": 64, "accumulation_steps": 4}
        self.assertEqual(
            batch_args("dense", b, {"dp": 2}),
            {"batch_size": 2, "total_batch_size": 1024, "sequence_length": 64},
        )
        self.assertEqual(
            batch_args("mixtral", b, {"dp": 1}),
            {"micro_batch_size": 2, "global_batch_size": 8, "sequence_length": 64},
        )
        self.assertEqual(
            batch_args("mixtral", b, {"dp": 2}),
            {"micro_batch_size": 2, "global_batch_size": 16, "sequence_length": 64},
        )
        with self.assertRaises(PlanError):
            batch_args("dense", {"micro_batch_samples": 0}, {})

    def test_invalid_capabilities_and_types(self):
        plan = load_plan("smoke")
        base = plan["cases"][0]
        model = plan["models"][base["model"]]
        for patch in [
            {"zero_stage": 3},
            {"sequence_parallel": True},
            {"unknown_argument": 1},
            {"batch_size": True},
            {"total_batch_size": 17},
            {"dtype": "fp8"},
        ]:
            case = copy.deepcopy(base)
            case["args"].update(patch)
            case["dtype"] = case["args"].get("dtype", case["dtype"])
            with self.subTest(patch=patch), self.assertRaises(PlanError):
                validate_case(case, model)
        mixtral = next(c for c in plan["cases"] if c["model"] == "mixtral")
        mixtral["nproc_per_node"] = 2
        with self.assertRaises(PlanError):
            validate_case(mixtral, plan["models"]["mixtral"])

    def test_unknown_config_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ["scenarios", "models", "suites"]:
                shutil.copytree(ROOT / d, root / d)
            path = root / "scenarios/tiny_smoke.json"
            value = json.loads(path.read_text())
            value["typo"] = True
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(PlanError, "unknown fields"):
                load_plan("smoke", config_root=root)

    def test_new_scenario_needs_no_scheduler_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ["scenarios", "models", "suites"]:
                shutil.copytree(ROOT / d, root / d)
            case = json.loads((root / "scenarios/tiny_smoke.json").read_text())
            case.update(
                id="new_feature",
                models=["llama3"],
                matrix={"dtype": ["float32", "bfloat16"]},
            )
            (root / "scenarios/new_feature.json").write_text(json.dumps(case))
            result = load_plan(
                "smoke", config_root=root, tags=["anchor"], cases="new_feature/*"
            )
            self.assertEqual(len(result["cases"]), 2)

    def test_hash_changes_with_semantics(self):
        a = load_plan("smoke")["cases"][0]
        b = load_plan("smoke", overrides={"learning_rate": 0.002})["cases"][0]
        self.assertEqual(a["id"], b["id"])
        self.assertNotEqual(a["config_hash"], b["config_hash"])


class ChangedAndQuarantineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = load_plan("pr")

    def test_model_change_preserves_all_anchors_and_dependencies(self):
        from planning import select_changed

        result = select_changed(self.plan, ["example/llama3/main.cc"])
        self.assertLess(len(result["cases"]), len(self.plan["cases"]))
        ids = {case["id"] for case in result["cases"]}
        anchors = {
            case["id"] for case in self.plan["cases"] if "anchor" in case["tags"]
        }
        self.assertTrue(anchors <= ids)
        self.assertTrue(
            all(
                case["id"] in ids
                for case in self.plan["cases"]
                if case["model"] == "llama3"
            )
        )
        for case in result["cases"]:
            self.assertTrue(set(case["dependencies"]) <= ids)
        self.assertTrue(result["coverage"]["partial"])
        self.assertTrue(
            any(r["uncovered_pairs"] for r in result["coverage"]["matrices"])
        )
        self.assertNotIn("change_selection", self.plan["coverage"])

    def test_common_unknown_empty_and_nightly_conservative(self):
        from planning import select_changed

        for paths in [
            [],
            ["infini_train/tensor.cc"],
            ["new/unknown.cc"],
            ["example/gpt2/main.cc", "CMakeLists.txt"],
        ]:
            result = select_changed(self.plan, paths)
            self.assertEqual(len(result["cases"]), len(self.plan["cases"]))
            self.assertTrue(result["coverage"]["change_selection"]["fallback"])
        nightly = load_plan("nightly")
        self.assertEqual(
            len(select_changed(nightly, ["example/llama3/main.cc"])["cases"]),
            len(nightly["cases"]),
        )

    def test_changed_rejects_preexisting_partial_selection(self):
        from planning import select_changed

        with self.assertRaisesRegex(PlanError, "unfiltered"):
            select_changed(load_plan("pr", models="llama3"), ["example/llama3/main.cc"])

    def test_quarantine_is_owned_expiring_annotation_never_omission(self):
        from datetime import date, timedelta

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ["scenarios", "models", "suites"]:
                shutil.copytree(ROOT / directory, root / directory)
            item = {
                "case_id": "tiny_smoke/gpt2/default",
                "owner": "training-team",
                "reason": "Tracked intermittent infrastructure issue",
                "issue": "#123",
                "expires": (date.today() + timedelta(days=7)).isoformat(),
            }
            path = root / "quarantines.json"

            def write(entry):
                path.write_text(
                    json.dumps({"schema_version": 1, "quarantines": [entry]})
                )

            write(item)
            result = load_plan("smoke", config_root=root)
            self.assertEqual(len(result["cases"]), 4)
            self.assertFalse(result["coverage"]["partial"])
            self.assertEqual(result["cases"][0]["quarantine"], item)
            self.assertTrue(result["coverage"]["quarantines"][0]["selected"])
            filtered = load_plan("smoke", config_root=root, models="llama3")
            self.assertFalse(filtered["coverage"]["quarantines"][0]["selected"])
            for patch in [
                {"owner": ""},
                {"reason": ""},
                {"issue": ""},
                {"expires": "not-a-date"},
                {"expires": "2000-01-01"},
                {"case_id": "missing/case"},
            ]:
                write(dict(item, **patch))
                with self.subTest(patch=patch), self.assertRaises(PlanError):
                    load_plan("smoke", config_root=root)

    def test_release_memory_drift_is_explicit(self):
        result = load_plan("release")
        long_cases = [
            case for case in result["cases"] if case["scenario"] == "long_stability"
        ]
        self.assertTrue(long_cases)
        for case in long_cases:
            check = next(c for c in case["checks"] if c["kind"] == "memory_growth")
            self.assertEqual(check["metric"], "peak_memory_bytes")
            self.assertGreater(
                case["expected_steps"],
                check["warmup_steps"] + 2 * check["window_steps"],
            )


class AdversarialPlanningTests(unittest.TestCase):
    def test_invalid_matrix_values_fail_even_with_other_valid_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ["scenarios", "models", "suites"]:
                shutil.copytree(ROOT / directory, root / directory)
            path = root / "scenarios/adversarial.json"
            base = {
                "schema_version": 1,
                "id": "adversarial",
                "models": ["gpt2"],
                "profile": "tiny",
                "tags": [],
                "suites": ["smoke"],
            }
            bad = [
                {"matrix": {"dtype": ["float32", "bad_dtype"]}},
                {"matrix": {"zero_stage": [0, True]}},
                {"matrix": {"zero_stage": [0, "1"]}},
                {"matrix": {"zero_stage": [0, 3]}},
                {"matrix": {"learning_rate": [0.001, float("nan")]}},
                {"matrix": {"learning_rate": [0.001, float("inf")]}},
                {"matrix": {"lr_decay_style": ["linear", "unknown"]}},
                {"matrix": {"sequence_parallel": [False, "true"]}},
                {"matrix": {"dtype": ["float32", "float32"]}},
                {"matrix": {"does_not_exist": [0, 1]}},
                {"matrix": {"tp": [1, 0]}},
                {"matrix": {"launch": ["threads", "shell"]}},
                {"matrix": {"num_iteration": [2, 2**32]}},
                {
                    "matrix": {"num_iteration": [2, 3]},
                    "args": {"learning_rate": float("inf")},
                },
                {"matrix": {"num_iteration": [2, 3]}, "args": {"unknown": 1}},
                {
                    "matrix": {"num_iteration": [2, 3]},
                    "reference": {"single_device": "yes"},
                },
                {"checks": [{"kind": "pretend_success"}]},
            ]
            for patch in bad:
                with self.subTest(patch=patch):
                    path.write_text(json.dumps(dict(base, **patch)))
                    with self.assertRaises(PlanError):
                        load_plan("smoke", config_root=root)

    def test_case_ids_and_manifest_binary_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ["scenarios", "models", "suites"]:
                shutil.copytree(ROOT / directory, root / directory)
            path = root / "scenarios/adversarial.json"
            base = {
                "schema_version": 1,
                "id": "adversarial",
                "models": ["gpt2"],
                "profile": "tiny",
                "tags": [],
                "suites": ["smoke"],
            }
            for bad_id in ["../escape", "safe/../../escape", "safe\\escape"]:
                path.write_text(
                    json.dumps(dict(base, templates=[{"variants": [{"id": bad_id}]}]))
                )
                with self.subTest(case=bad_id), self.assertRaises(PlanError):
                    load_plan("smoke", config_root=root)
            path.unlink()
            model_path = root / "models/gpt2.json"
            model = json.loads(model_path.read_text())
            model["binary"] = "../escape"
            model_path.write_text(json.dumps(model))
            with self.assertRaises(PlanError):
                load_plan("smoke", config_root=root)

    def test_capacity_and_layer_constraints_are_explicit_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ["scenarios", "models", "suites"]:
                shutil.copytree(ROOT / directory, root / directory)
            scenario = {
                "schema_version": 1,
                "id": "capacity",
                "models": ["gpt2"],
                "profile": "tiny",
                "tags": [],
                "suites": ["smoke"],
                "matrix": {"sequence_length": [16, 128]},
            }
            path = root / "scenarios/capacity.json"
            path.write_text(json.dumps(scenario))
            plan = load_plan("smoke", config_root=root)
            report = next(
                r for r in plan["coverage"]["matrices"] if r["scenario"] == "capacity"
            )
            self.assertEqual(report["legal_candidates"], 1)
            self.assertIn("positional capacity", report["excluded"][0]["reason"])
            scenario["matrix"] = {"pp": [1, 2], "vpp": [1, 3]}
            path.write_text(json.dumps(scenario))
            plan = load_plan("smoke", config_root=root)
            report = next(
                r for r in plan["coverage"]["matrices"] if r["scenario"] == "capacity"
            )
            self.assertTrue(
                any("divide layer count" in x["reason"] for x in report["excluded"])
            )

    def test_overrides_preserve_reference_and_resume_semantics(self):
        plan = load_plan(
            "pr",
            cases="tiny_zero/llama3/dtype=float32/zero_stage=1",
            overrides={"num_iteration": 8, "zero_stage": 2, "learning_rate": 0.0002},
        )
        self.assertEqual(len(plan["cases"]), 2)
        reference, candidate = plan["cases"]
        self.assertEqual(reference["resolved_args"]["zero_stage"], 0)
        self.assertEqual(candidate["resolved_args"]["zero_stage"], 2)
        self.assertEqual(reference["expected_steps"], 8)
        self.assertEqual(candidate["expected_steps"], 8)
        self.assertEqual(
            reference["batch"]["global_tokens"], candidate["batch"]["global_tokens"]
        )
        for case in plan["cases"]:
            self.assertEqual(
                case["gpus"], case["nproc_per_node"] * case["nthread_per_process"]
            )
        for override in [
            {"nthread_per_process": 2},
            {"tensor_parallel": 2},
            {"total_batch_size": 123},
        ]:
            with self.subTest(override=override), self.assertRaisesRegex(
                PlanError, "managed"
            ):
                load_plan("smoke", overrides=override)
        with self.assertRaisesRegex(PlanError, "resume start"):
            load_plan("pr", overrides={"num_iteration": 2})

    def test_manifest_types_do_not_confuse_gflags_validators(self):
        for model in load_plan("smoke")["models"].values():
            self.assertNotIn("validator", model["argument_types"].values())
            if model["adapter"] == "dense":
                self.assertEqual(model["argument_types"]["zero_stage"], "int32")
                self.assertEqual(model["argument_types"]["device"], "string")


class CheckContractTests(unittest.TestCase):
    def test_empty_unknown_and_malformed_check_specs_fail_planning(self):
        plan = load_plan("smoke")
        original = next(case for case in plan["cases"] if case["model"] == "gpt2")
        model = plan["models"]["gpt2"]
        invalid = [
            [],
            None,
            [{"kind": "unknown"}],
            [{"kind": "oracle", "ignored": True}],
            [{"kind": "oracle"}, {"kind": "oracle"}],
            [{"kind": "equivalent", "reference_case": "bad/../path"}],
            [{"kind": "equivalent", "reference_case": "other", "metrics": []}],
            [
                {
                    "kind": "equivalent",
                    "reference_case": "other",
                    "metrics": ["loss", "loss"],
                }
            ],
            [
                {
                    "kind": "equivalent",
                    "reference_case": "other",
                    "tolerance": float("nan"),
                }
            ],
            [
                {
                    "kind": "resume",
                    "reference_case": "other",
                    "tolerance": {"atol": float("inf")},
                }
            ],
            [{"kind": "lora_frozen"}],
        ]
        for specs in invalid:
            case = copy.deepcopy(original)
            case["checks"] = specs
            with self.subTest(specs=specs), self.assertRaises(PlanError):
                validate_case(case, model)
        for timeout in [0, -1, True, 0.5, float("inf")]:
            case = copy.deepcopy(original)
            case["timeout_seconds"] = timeout
            with self.subTest(timeout=timeout), self.assertRaises(PlanError):
                validate_case(case, model)

    def test_performance_window_repeat_and_budget_contract(self):
        plan = load_plan("perf")
        # Optional repeated benchmarking contracts remain supported; perf itself
        # now captures kernel profiles with one execution per configuration.
        original = copy.deepcopy(plan["cases"][0])
        original["args"]["num_iteration"] = 40
        original["performance"] = {"warmup_steps": 10, "measure_steps": 30, "repetitions": 3}
        original["repetitions"] = 3
        original["checks"] = [{"kind": "finite_and_complete"},
                              dict(kind="performance", **original["performance"])]
        model = plan["models"][original["model"]]
        invalid = [
            {"warmup_steps": -1},
            {"measure_steps": 0},
            {"measure_steps": 0.5},
            {"measure_steps": 31},
            {"repetitions": 2},
            {"repetitions": True},
            {"threshold": float("nan")},
            {"max_jitter": float("inf")},
            {"memory_threshold": -0.1},
            {"ignored_budget": 1},
        ]
        for patch in invalid:
            case = copy.deepcopy(original)
            case["performance"].update(patch)
            with self.subTest(patch=patch), self.assertRaises(PlanError):
                validate_case(case, model)
        case = copy.deepcopy(original)
        case["repetitions"] = 4
        with self.assertRaisesRegex(PlanError, "disagree"):
            validate_case(case, model)
        case = copy.deepcopy(original)
        case["args"]["num_iteration"] = 39
        with self.assertRaisesRegex(PlanError, "warmup"):
            validate_case(case, model)

    def test_memory_window_and_metric_contract(self):
        plan = load_plan("release")
        original = next(c for c in plan["cases"] if c["scenario"] == "long_stability")
        model = plan["models"][original["model"]]
        for patch in [
            {"window_steps": 0},
            {"window_steps": 1},
            {"warmup_steps": 200},
            {"max_growth_bytes": float("inf")},
            {"max_growth_fraction": -1},
            {"min_increasing_fraction": 1.01},
            {"metric": "unavailable_memory"},
        ]:
            case = copy.deepcopy(original)
            next(
                spec for spec in case["checks"] if spec["kind"] == "memory_growth"
            ).update(patch)
            with self.subTest(patch=patch), self.assertRaises(PlanError):
                validate_case(case, model)

    def test_oracle_constraints_do_not_rewrite_candidate(self):
        plan = load_plan("pr")
        original = next(
            c
            for c in plan["cases"]
            if c["scenario"] == "tiny_oracle" and c["model"] == "gpt2"
        )
        model = plan["models"]["gpt2"]
        for patch in [
            {"dtype": "bfloat16"},
            {"lora_rank": 2},
            {"zero_stage": 1},
            {"lr_decay_style": "cosine"},
            {"lr_warmup_iters": 2},
            {"load": "artifact:other/checkpoints"},
            {"total_batch_size": 63},
            {"nthread_per_process": 2},
        ]:
            case = copy.deepcopy(original)
            case["args"].update(patch)
            before = copy.deepcopy(case["args"])
            with self.subTest(patch=patch), self.assertRaises(PlanError):
                validate_case(case, model)
            self.assertEqual(case["args"], before)
        case = copy.deepcopy(original)
        case["args"]["llmc_filepath"] = "fixture:unversioned.weights"
        with self.assertRaisesRegex(PlanError, "fixed"):
            validate_case(case, model)

    def test_explicit_reference_joins_dependency_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ["scenarios", "models", "suites"]:
                shutil.copytree(ROOT / directory, root / directory)
            scenario = {
                "schema_version": 1,
                "id": "explicit_reference",
                "models": ["gpt2"],
                "profile": "tiny",
                "tags": [],
                "suites": ["smoke"],
                "checks": [
                    {"kind": "finite_and_complete"},
                    {
                        "kind": "equivalent",
                        "reference_case": "tiny_smoke/gpt2/default",
                        "metrics": ["loss"],
                    },
                ],
            }
            path = root / "scenarios/explicit_reference.json"
            path.write_text(json.dumps(scenario))
            result = load_plan("smoke", config_root=root, cases="explicit_reference/*")
            self.assertEqual(
                [c["id"] for c in result["cases"]],
                ["tiny_smoke/gpt2/default", "explicit_reference/gpt2/default"],
            )
            scenario["checks"][1]["reference_case"] = "missing/reference"
            path.write_text(json.dumps(scenario))
            with self.assertRaisesRegex(PlanError, "missing dependency"):
                load_plan("smoke", config_root=root)


class DistributedOptimizerCapabilityTests(unittest.TestCase):
    def test_positive_zero_requires_dp_and_all_dense_models_have_both_stages(self):
        plan = load_plan("pr")
        covered = set()
        for case in plan["cases"]:
            stage = case["resolved_args"].get("zero_stage", 0)
            if stage:
                self.assertGreater(case["topology"]["dp"], 1, case["id"])
                covered.add((case["model"], stage, case["dtype"]))
        for model in ["gpt2", "llama3", "qwen3"]:
            for stage in [1, 2]:
                for dtype in ["float32", "bfloat16"]:
                    self.assertIn((model, stage, dtype), covered)
        self.assertTrue(
            any(
                "requires DP > 1" in item["reason"]
                for report in plan["coverage"]["matrices"]
                for item in report["excluded"]
            )
        )
        for report in plan["coverage"]["matrices"]:
            self.assertFalse(report["uncovered_pairs"])

    def test_single_rank_zero_is_rejected_before_launch(self):
        plan = load_plan("smoke")
        case = copy.deepcopy(
            next(case for case in plan["cases"] if case["model"] == "llama3")
        )
        case["args"]["zero_stage"] = 1
        with self.assertRaisesRegex(PlanError, "requires DP > 1"):
            validate_case(case, plan["models"]["llama3"])


class AssetProvenancePlanningTests(unittest.TestCase):
    def test_all_managed_path_overrides_rejected_with_fixture_guidance(self):
        for key in [
            "input_bin",
            "input_val_bin",
            "llmc_filepath",
            "tokenizer_bin",
            "lora_load_path",
            "lora_save_path",
            "load",
            "save",
            "metrics_file",
            "regression_snapshot_dir",
        ]:
            with self.subTest(key=key), self.assertRaisesRegex(PlanError, "--fixture"):
                load_plan(
                    "smoke", models="gpt2", overrides={key: "/tmp/unfingerprinted"}
                )

    def test_scenario_cannot_silently_replace_profile_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ["scenarios", "models", "suites"]:
                shutil.copytree(ROOT / directory, root / directory)
            scenario = {
                "schema_version": 1,
                "id": "asset_override",
                "models": ["gpt2"],
                "profile": "tiny",
                "tags": [],
                "suites": ["smoke"],
            }
            path = root / "scenarios/asset_override.json"
            for key, value in [
                ("input_bin", "/tmp/other"),
                ("llmc_filepath", "fixture:other.weights"),
                ("lora_load_path", "/tmp/adapter"),
                ("save", "/tmp/shared-checkpoint"),
            ]:
                scenario["args"] = {key: value}
                path.write_text(json.dumps(scenario))
                with self.subTest(key=key), self.assertRaises(PlanError):
                    load_plan("smoke", config_root=root)
            scenario["args"] = {"lora_load_path": "fixture:custom.adapter"}
            path.write_text(json.dumps(scenario))
            case = load_plan("smoke", config_root=root, cases="asset_override/*")[
                "cases"
            ][0]
            self.assertIn("custom.adapter", case["fixtures"])

    def test_actual_extra_asset_bytes_change_history_fingerprint(self):
        from assets import Assets

        plan = load_plan("smoke", models="gpt2")
        case = copy.deepcopy(plan["cases"][0])
        case["args"]["lora_load_path"] = "fixture:custom.adapter"
        case["fixtures"] = (
            []
        )  # Stale metadata must be recomputed from effective inputs.
        validate_case(case, plan["models"]["gpt2"])
        self.assertIn("custom.adapter", case["fixtures"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "adapter.bin"
            asset.write_bytes(b"first")
            keys = []
            for content in [b"first", b"other"]:
                asset.write_bytes(content)
                assets = Assets(
                    ROOT, root / "cache", overrides={"custom.adapter": str(asset)}
                )
                self.assertEqual(assets.prepare([case]), [])
                keys.append(dict(assets.hashes))
            self.assertNotEqual(keys[0], keys[1])


class BackendDtypeCapabilityTests(unittest.TestCase):
    def test_cpu_excludes_bf16_without_coercion_and_cuda_keeps_it(self):
        cpu = load_plan("pr", backend="cpu")
        self.assertTrue(
            all(
                case["dtype"] == "float32"
                and case["resolved_args"]["dtype"] == "float32"
                for case in cpu["cases"]
            )
        )
        exclusions = [
            item
            for report in cpu["coverage"]["matrices"]
            for item in report["excluded"]
        ]
        self.assertTrue(
            any(
                item["point"].get("dtype") == "bfloat16"
                and "FP32-only" in item["reason"]
                for item in exclusions
            )
        )
        gpu = load_plan("pr", backend="cuda")
        for model in ["gpt2", "llama3", "qwen3", "mixtral"]:
            self.assertEqual(
                {case["dtype"] for case in gpu["cases"] if case["model"] == model},
                {"float32", "bfloat16"},
            )
        self.assertTrue(
            all(not report["uncovered_pairs"] for report in gpu["coverage"]["matrices"])
        )

    def test_explicit_cpu_bf16_request_errors(self):
        with self.assertRaisesRegex(PlanError, "FP32-only"):
            load_plan("smoke", backend="cpu", overrides={"dtype": "bfloat16"})


class LongOracleCapabilityTests(unittest.TestCase):
    def test_canonical_long_run_keeps_200_steps_accumulation_and_oracle(self):
        plan = load_plan("release")
        cases = [case for case in plan["cases"] if case["scenario"] == "long_stability"]
        self.assertEqual(
            {case["model"] for case in cases}, {"gpt2", "llama3", "qwen3", "mixtral"}
        )
        for case in cases:
            self.assertEqual(case["expected_steps"], 200)
            self.assertEqual(case["batch"]["accumulation_steps"], 4)
            self.assertEqual(case["batch"]["micro_batch_samples"], 2)
            self.assertEqual(case["dtype"], "float32")
            self.assertIn("oracle", {spec["kind"] for spec in case["checks"]})
            self.assertIn("memory_growth", {spec["kind"] for spec in case["checks"]})

    def test_epoch_wrap_supported_but_one_complete_microbatch_required(self):
        plan = load_plan("pr")
        case = copy.deepcopy(
            next(
                case
                for case in plan["cases"]
                if case["scenario"] == "tiny_oracle" and case["model"] == "gpt2"
            )
        )
        case["args"]["num_iteration"] = 1000
        case["args"]["total_batch_size"] = 128
        validate_case(case, plan["models"]["gpt2"])
        self.assertEqual(case["batch"]["accumulation_steps"], 4)
        case["args"]["batch_size"] = 2048
        case["args"]["total_batch_size"] = 32768
        with self.assertRaisesRegex(PlanError, "no complete microbatch"):
            validate_case(case, plan["models"]["gpt2"])




class ToleranceRegistryPlanningTests(unittest.TestCase):
    def copy_config(self, directory):
        root=Path(directory)
        for name in ["scenarios","models","suites"]:
            shutil.copytree(ROOT/name,root/name)
        shutil.copyfile(ROOT/"tolerances.json",root/"tolerances.json")
        return root

    def test_default_policy_maps_preserve_legacy_hash_payload(self):
        from planning import config_hash
        plan=load_plan("release")
        for case in plan["cases"]:
            original={"model":case["model"],"profile":case["profile"],
                      "args":case["resolved_args"],"nproc_per_node":case["nproc_per_node"],
                      "fixtures":case["fixtures"],"checks":case["checks"]}
            self.assertEqual(case["config_hash"],config_hash(original),case["id"])
        scheduler=next(case for case in plan["cases"] if case["scenario"]=="tiny_scheduler")
        self.assertEqual(scheduler["resolved_tolerances"]["lr_schedule"]["lr"],{"atol":1e-9,"rtol":1e-5})
        resumed=next(case for case in plan["cases"] if "resume" in case["resolved_tolerances"] and "consumed_samples" in case["resolved_tolerances"]["resume"])
        self.assertEqual(resumed["resolved_tolerances"]["resume"]["consumed_samples"],{"atol":0,"rtol":0})

    def test_editing_existing_policy_changes_only_affected_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.copy_config(tmp)
            pattern="tiny_accumulation/gpt2/dtype=float32"
            before={case["id"]:case for case in load_plan("pr",cases=pattern,config_root=root)["cases"]}
            path=root/"tolerances.json";registry=json.loads(path.read_text())
            registry["policies"]["distributed_equivalence"]["backends"]={"cuda":{"loss":{"float32":{"atol":2e-5,"rtol":1e-5}}}}
            path.write_text(json.dumps(registry))
            after={case["id"]:case for case in load_plan("pr",cases=pattern,config_root=root)["cases"]}
            self.assertNotEqual(before[pattern]["config_hash"],after[pattern]["config_hash"])
            self.assertEqual(before[pattern]["checks"],after[pattern]["checks"])
            self.assertEqual(before[pattern+"/reference"]["config_hash"],after[pattern+"/reference"]["config_hash"])
            cpu=load_plan("pr",cases=pattern,config_root=root,backend="cpu")["cases"][-1]
            ordinary_cpu=load_plan("pr",cases=pattern,backend="cpu")["cases"][-1]
            self.assertEqual(cpu["config_hash"],ordinary_cpu["config_hash"])
            self.assertEqual(after[pattern]["resolved_tolerances"]["equivalent"]["loss"]["atol"],2e-5)

    def test_new_json_policy_requires_no_python_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.copy_config(tmp)
            path=root/"tolerances.json";registry=json.loads(path.read_text())
            policy=copy.deepcopy(registry["policies"]["distributed_equivalence"])
            policy["metrics"]={"loss":{"float32":{"atol":2e-6,"rtol":3e-6}}}
            policy["backends"]={"cuda":{"gradient":{"float32":{"atol":4e-6,"rtol":5e-6}}}}
            registry["policies"]["new_numeric_policy"]=policy;path.write_text(json.dumps(registry))
            scenario={"schema_version":1,"id":"new_policy_case","models":["gpt2"],"profile":"tiny",
                      "suites":["smoke"],"tags":[],"batch":{"micro_batch_samples":1,"sequence_length":16,"accumulation_steps":4},
                      "reference":{"single_device":True,"tolerance":"new_numeric_policy","metrics":["loss","gradients","updated_parameters"]}}
            (root/"scenarios/new_policy.json").write_text(json.dumps(scenario))
            candidate=load_plan("smoke",config_root=root,cases="new_policy_case/*")["cases"][-1]
            self.assertEqual(candidate["tolerance_policies"]["equivalent"],"new_numeric_policy")
            self.assertEqual(candidate["resolved_tolerances"]["equivalent"]["loss"],{"atol":2e-6,"rtol":3e-6})
            self.assertEqual(candidate["resolved_tolerances"]["equivalent"]["gradient"],{"atol":4e-6,"rtol":5e-6})
            self.assertIn("parameter",candidate["resolved_tolerances"]["equivalent"])
            scenario["reference"]["tolerance"]="unregistered"
            (root/"scenarios/new_policy.json").write_text(json.dumps(scenario))
            with self.assertRaisesRegex(PlanError,"tolerance"):
                load_plan("smoke",config_root=root)


if __name__ == "__main__":
    unittest.main()

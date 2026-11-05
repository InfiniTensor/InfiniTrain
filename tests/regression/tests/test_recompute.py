"""Recompute cases must differ from their controls only in recompute flags."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning import PlanError, load_plan, validate_case


RECOMPUTE_FLAGS = {
    "activation_recompute", "recompute_granularity",
    "recompute_method", "recompute_num_layers",
}


class RecomputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = load_plan("nightly", alignment="history")
        cls.cases = {c["id"]: c for c in cls.plan["cases"]}

    def assert_matching_control(self, case, control):
        self.assertTrue(case["resolved_args"]["activation_recompute"])
        self.assertFalse(control["resolved_args"]["activation_recompute"])
        strip = lambda c: {k: v for k, v in c["resolved_args"].items() if k not in RECOMPUTE_FLAGS}
        self.assertEqual(strip(case), strip(control), case["id"])
        for field in ("fixtures", "batch", "topology", "nproc_per_node", "expected_steps"):
            self.assertEqual(case[field], control[field], (case["id"], field))

    def test_master_recompute_matches_basic_or_zero_controls(self):
        cases = [c for c in self.cases.values() if c["scenario"] == "classic_recompute"]
        self.assertEqual(len(cases), 28)
        for case in cases:
            name = case["legacy_id"]
            base = name.split("_recompute_", 1)[0]
            group = "classic_basic"
            if case["dtype"] == "bfloat16":
                base += "_bfloat16"
            if "_distopt" in name:
                group = "classic_zero"
                base += "_distopt"
            self.assert_matching_control(case, self.cases[f'{group}/{case["model"]}/{base}'])

    def test_every_qwen_classic_has_both_recompute_methods(self):
        controls = [c for c in self.cases.values() if c["scenario"] == "classic_qwen3"]
        self.assertEqual(len(controls), 7)
        for control in controls:
            for method in ("uniform", "block"):
                cid = "classic_recompute_qwen3/qwen3/" + control["legacy_id"] + "_recompute_" + method + "2"
                self.assert_matching_control(self.cases[cid], control)
                self.assertEqual(self.cases[cid]["resolved_args"]["recompute_method"], method)

    def test_tiny_cases_match_existing_accumulation_controls(self):
        cases = [c for c in self.cases.values() if c["scenario"] == "tiny_recompute"]
        self.assertEqual(len(cases), 12)
        for case in cases:
            control = self.cases[f'tiny_accumulation/{case["model"]}/dtype={case["dtype"]}']
            self.assert_matching_control(case, control)
            self.assertFalse(case["dependencies"])

    def test_deep_reference_preserves_batch_and_topology(self):
        plan = load_plan("pr", tags=["recompute"], alignment="deep")
        cases = {c["id"]: c for c in plan["cases"]}
        self.assertEqual(len(cases), 24)
        for case in cases.values():
            if case["id"].endswith("/reference"):
                continue
            self.assert_matching_control(case, cases[case["id"] + "/reference"])

    def test_invalid_recompute_settings_fail_planning(self):
        original = next(c for c in self.cases.values() if c["scenario"] == "tiny_recompute")
        model = self.plan["models"][original["model"]]
        for patch in ({"recompute_num_layers": 0}, {"recompute_method": "typo"},
                      {"recompute_granularity": "selective"}, {"recompute_granularity": "none"}):
            case = copy.deepcopy(original)
            case["args"].update(patch)
            with self.subTest(patch=patch), self.assertRaises(PlanError):
                validate_case(case, model)


if __name__ == "__main__":
    unittest.main()

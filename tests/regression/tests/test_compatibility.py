"""Exercise legacy wrappers against the real planner without builds or GPUs."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "scripts/run_models_and_profile.bash"


class CompatibilityTests(unittest.TestCase):
    def run_wrapper(self, *args, cwd=None):
        return subprocess.run(
            ["bash", str(WRAPPER), *args],
            cwd=str(cwd or REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def plan(self, *args, cwd=None):
        result = self.run_wrapper(*args, "--dry-run", cwd=cwd)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_default_includes_master_classics_and_qwen_recompute(self):
        cases = self.plan()["cases"]
        self.assertEqual(len(cases), 212)
        self.assertTrue(all(c["scenario"].startswith("classic_") for c in cases))

    def test_recompute_group_includes_qwen(self):
        cases = self.plan("--only-run", "recompute")["cases"]
        self.assertEqual(len(cases), 42)
        self.assertEqual({c["model"] for c in cases}, {"gpt2", "llama3", "qwen3"})
        self.assertEqual(len(self.plan("--only-run", "recompute", "--model", "qwen3")["cases"]), 14)

    def test_group_union_preserved(self):
        cases = self.plan("--only-run", " basic,zero ")["cases"]
        self.assertEqual(len(cases), 60)
        self.assertEqual(
            {c["scenario"] for c in cases}, {"classic_basic", "classic_zero"}
        )

    def test_equals_form_and_model_forwarding(self):
        cases = self.plan("--only-run=basic,zero", "--model", "llama3")["cases"]
        self.assertEqual(len(cases), 30)
        self.assertEqual({c["model"] for c in cases}, {"llama3"})

    def test_default_config_path_from_any_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = self.plan(
                "--test-config", "test_config.json", "--only-run", "moe", cwd=tmp
            )["cases"]
            self.assertEqual(len(cases), 12)
        self.assertEqual(
            len(
                self.plan(
                    "--test-config",
                    str(REPO / "scripts/test_config.json"),
                    "--only-run",
                    "qwen3",
                )["cases"]
            ),
            7,
        )

    def test_bad_or_custom_input_is_error(self):
        for args in [
            ("--only-run", ""),
            ("--only-run", "basic,typo"),
            ("--only-run",),
            ("--test-config",),
            ("--test-config", "missing.json"),
            ("--does-not-exist",),
        ]:
            with self.subTest(args=args):
                self.assertNotEqual(self.run_wrapper(*args).returncode, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.json"
            path.write_text("{}")
            result = self.run_wrapper("--test-config", str(path))
            self.assertEqual(result.returncode, 2)
            self.assertIn("Migrate", result.stderr)

    def test_classic_fixture_paths_match_original(self):
        legacy = json.loads((REPO / "scripts/test_config.json").read_text())[
            "variables"
        ]
        mapping = json.loads(
            (REPO / "tests/regression/fixtures/classic.json").read_text()
        )["files"]
        for fid, item in mapping.items():
            _, model, kind = fid.split(".")
            suffix = {
                "tokens": "INPUT_BIN",
                "weights": "LLMC_FILEPATH",
                "lora": "LORA_WEIGHTS_DIR",
            }[kind]
            self.assertEqual(
                item["path"],
                legacy[model.upper() + "_" + suffix].split("/data/llmc/", 1)[1],
            )
            self.assertFalse(Path(item["path"]).is_absolute())
            self.assertNotIn("..", Path(item["path"]).parts)


if __name__ == "__main__":
    unittest.main()

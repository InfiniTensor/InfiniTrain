"""Developer reproduction commands preserve semantics without shell injection."""

from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction import command_for_case
import runner


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "project with spaces"
        self.repo.mkdir()
        self.args = SimpleNamespace(
            suite="pr",
            backend="cpu",
            build_dir=self.repo / ".cache" / "cpu build",
            config_root=self.repo / "tests" / "regression",
            history_root=self.repo / "history",
            set=["num_iteration=7", 'lr_decay_style="cosine"'],
            reference_run=None,
            checkpoint_root=None,
            perf_environment=None,
            docker_container=None,
            record_history=False,
            output=self.repo / "old-output",
            strict=True,
            rerun_failed=self.repo / "old-summary.json",
            model=["gpt2"],
            tag=["tiny"],
            jobs=8,
        )
        self.case = {
            "id": "tiny/gpt2/dtype=float32",
            "fixtures": ["tiny.weights", "tiny.tokens"],
        }
        self.assets = SimpleNamespace(
            paths={
                "tiny.weights": str(self.repo / ".cache" / "model weights.bin"),
                "tiny.tokens": str(self.repo / ".cache" / "tokens.bin"),
                "unused.weights": str(self.repo / "unused.bin"),
            }
        )

    def tearDown(self):
        self.tmp.cleanup()

    def parse(self, gpu_ids=()):
        command = command_for_case(
            self.args, self.case, self.assets, self.repo, gpu_ids
        )
        words = shlex.split(command)
        self.assertEqual(words[0], "./scripts/run_all_tests.sh")
        return command, words, runner.parser().parse_args(words[1:])

    def test_cpu_overrides_and_effective_paths_are_preserved(self):
        command, words, parsed = self.parse()
        self.assertEqual(parsed.backend, "cpu")
        self.assertEqual(parsed.set, self.args.set)
        self.assertEqual(parsed.build_dir, Path(".cache/cpu build"))
        self.assertEqual(parsed.config_root, Path("tests/regression"))
        self.assertEqual(parsed.history_root, Path("history"))
        self.assertEqual(parsed.case, [self.case["id"]])
        self.assertEqual(parsed.jobs, 1)
        self.assertTrue(parsed.no_build and parsed.skip_unit)
        self.assertIsNone(parsed.output)
        self.assertIsNone(parsed.rerun_failed)
        self.assertFalse(parsed.strict)
        self.assertIsNone(parsed.model)
        self.assertIsNone(parsed.tag)
        self.assertNotIn(str(self.repo), command)
        self.assertEqual(
            set(parsed.fixture),
            {"tiny.weights=.cache/model weights.bin", "tiny.tokens=.cache/tokens.bin"},
        )

    def test_reference_perf_gpu_and_checkpoint_options(self):
        self.args.backend = "cuda"
        self.args.suite = "perf"
        self.args.reference_run = self.repo / "reference run"
        self.args.checkpoint_root = Path(self.tmp.name) / "large scratch"
        self.args.perf_environment = "stable A100"
        self.args.docker_container = "training container"
        command, words, parsed = self.parse(["GPU-abcdef", "3"])
        self.assertEqual(parsed.gpus, "GPU-abcdef,3")
        self.assertEqual(parsed.reference_run, Path("reference run"))
        self.assertEqual(parsed.checkpoint_root, self.args.checkpoint_root)
        self.assertEqual(parsed.perf_environment, "stable A100")
        self.assertEqual(parsed.docker_container, "training container")
        self.assertEqual(parsed.suite, "perf")

    def test_shell_metacharacters_are_literal_arguments(self):
        dangerous = "weights 'quoted'; $(touch marker) `id` $HOME.bin"
        self.assets.paths["tiny.weights"] = str(self.repo / dangerous)
        self.args.set = ['custom="a; $(touch never) "quoted""']
        self.args.record_history = True
        self.args.perf_environment = "oracle's evidence; $(not-a-command)\nsecond line"
        command, words, parsed = self.parse()
        self.assertIn("tiny.weights=" + dangerous, parsed.fixture)
        self.assertEqual(parsed.set, self.args.set)
        self.assertEqual(parsed.perf_environment, self.args.perf_environment)
        self.assertTrue(parsed.record_history)
        self.assertFalse((self.repo / "marker").exists())
        self.assertFalse((self.repo / "never").exists())

    def test_option_like_values_are_arguments_not_new_options(self):
        self.case["id"] = "--unusual/gpt2/default"
        self.args.build_dir = self.repo / "--build-folder"
        self.args.record_history = True
        self.args.perf_environment = "--machine-label"
        _, _, parsed = self.parse()
        self.assertEqual(parsed.case, [self.case["id"]])
        self.assertEqual(parsed.build_dir, Path("./--build-folder"))
        self.assertEqual(parsed.perf_environment, self.args.perf_environment)
        self.assertEqual(parsed.perf_environment, self.args.perf_environment)

    def test_only_required_fixtures_and_external_absolute_paths(self):
        outside = Path(self.tmp.name) / "shared data" / "weights.bin"
        self.assets.paths["tiny.weights"] = str(outside)
        _, _, parsed = self.parse()
        self.assertIn("tiny.weights=" + str(outside), parsed.fixture)
        self.assertFalse(
            any(item.startswith("unused.weights=") for item in parsed.fixture)
        )

    def test_missing_fixture_is_not_silently_omitted(self):
        del self.assets.paths["tiny.tokens"]
        with self.assertRaisesRegex(ValueError, "tiny.tokens"):
            self.parse()

    def test_relative_arguments_are_resolved_from_repository_root(self):
        self.args.build_dir = Path("custom build")
        self.args.checkpoint_root = Path("scratch")
        _, _, parsed = self.parse()
        self.assertEqual(parsed.build_dir, Path("custom build"))
        self.assertEqual(parsed.checkpoint_root, Path("scratch"))


if __name__ == "__main__":
    unittest.main()

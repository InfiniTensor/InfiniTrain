"""Versioned fixture integrity and data progression contracts."""

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
spec = importlib.util.spec_from_file_location(
    "fixture_generator", FIXTURES / "generate.py"
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class FixtureTests(unittest.TestCase):
    def test_published_checksums_match_generation(self):
        expected = json.loads((FIXTURES / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            actual = generator.generate(directory)
            self.assertEqual(expected, actual)
            for entry in actual["files"].values():
                data = (Path(directory) / entry["path"]).read_bytes()
                self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(entry["size_bytes"], len(data))

    def test_token_formats_and_nontrivial_resume_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            generator.generate(directory)
            decoded = {}
            for model in generator.MODELS:
                data = (Path(directory) / (model + "_tokens.bin")).read_bytes()
                magic, version, count = struct.unpack_from("<3i", data)
                self.assertEqual(version, 1)
                dtype = "H" if model == "gpt2" else "I"
                self.assertEqual(magic, 20240520 if model == "gpt2" else 20240801)
                decoded[model] = struct.unpack_from(
                    "<%d%s" % (count, dtype), data, 1024
                )
                self.assertTrue(
                    all(0 <= x < generator.CONFIG["vocab_size"] for x in decoded[model])
                )
                self.assertNotEqual(decoded[model][:32], decoded[model][32:64])
            self.assertEqual(decoded["gpt2"], decoded["llama3"])
            self.assertEqual(decoded["qwen3"], decoded["mixtral"])

    def test_architectures_allow_required_parallel_anchors(self):
        config = generator.CONFIG
        self.assertEqual(config["n_layer"] % (2 * 2), 0)  # PP2 x VPP2
        for dimension in (
            "n_head",
            "n_kv_head",
            "n_embd",
            "vocab_size",
            "intermediate_size",
        ):
            self.assertEqual(config[dimension] % 2, 0)  # TP2


if __name__ == "__main__":
    unittest.main()

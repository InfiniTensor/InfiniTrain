"""Fault injection against actual exported model artifacts (no GPU needed).

Set INFINITRAIN_ORACLE_SNAPSHOTS and INFINITRAIN_ORACLE_TOKENS to an existing
tiny_oracle candidate's snapshots root and generated token file. This integration
test is optional in the stdlib-only hosted suite; GPU validation supplies them.
"""

import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import checks

READY = (
    os.environ.get("INFINITRAIN_ORACLE_SNAPSHOTS")
    and os.environ.get("INFINITRAIN_ORACLE_TOKENS")
    and importlib.util.find_spec("torch")
    and importlib.util.find_spec("numpy")
)


@unittest.skipUnless(
    READY, "requires exported tiny model artifacts and optional torch/numpy"
)
class OperationOracleTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.candidate = self.root / "candidate"
        source = Path(os.environ["INFINITRAIN_ORACLE_SNAPSHOTS"])
        for step in (0, 1):
            shutil.copytree(
                source / ("step_" + str(step)), self.candidate / ("step_" + str(step))
            )
        spec = importlib.util.spec_from_file_location(
            "operation_reference", ROOT / "fixtures/reference.py"
        )
        self.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reference)

    def generate(self):
        self.reference.run(
            self.candidate / "step_0/rank_0/manifest.json",
            os.environ["INFINITRAIN_ORACLE_TOKENS"],
            self.root / "oracle",
            steps=1,
            candidate_snapshots=self.candidate,
        )

    def corrupt(self, kind):
        directory = self.candidate / "step_1/rank_0"
        manifest = json.loads((directory / "manifest.json").read_text())
        item = next(
            item
            for item in manifest["tensors"]
            if item["kind"] == kind
            and item["name"] == "transformer.h.0.attn.c_attn.weight"
        )
        path = directory / item["path"]
        payload = path.read_bytes()
        offset, length = item.get("offset", 0), item.get("length", len(payload))
        data = self.np.load(io.BytesIO(payload[offset : offset + length]))
        data.flat[0] += 0.1
        # Preserve the producer's NPY header alignment and bundle extents.
        # NumPy's default header padding need not match the C++ writer's.
        changed = bytearray(payload[offset : offset + length])
        changed[-data.nbytes :] = data.tobytes(order="C")
        self.assertEqual(len(changed), length)
        path.write_bytes(payload[:offset] + changed + payload[offset + length :])

    def compare(self, kind):
        return checks.compare_snapshots(
            self.root / "oracle/step_1",
            self.candidate / "step_1",
            dtype="float32",
            required_kinds=(kind,),
        )["status"]

    def test_real_forward_backward_and_optimizer_pass(self):
        self.generate()
        self.assertEqual(self.compare("gradient"), "PASS")
        self.assertEqual(self.compare("parameter"), "PASS")

    def test_changed_gradient_is_rejected(self):
        self.corrupt("gradient")
        self.generate()
        self.assertEqual(self.compare("gradient"), "FAIL")

    def test_changed_parameter_is_rejected(self):
        self.corrupt("parameter")
        self.generate()
        self.assertEqual(self.compare("gradient"), "PASS")
        self.assertEqual(self.compare("parameter"), "FAIL")


if __name__ == "__main__":
    unittest.main()

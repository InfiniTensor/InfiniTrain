"""Independent oracle data ordering at epoch boundaries, without torch."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "reference_batching", Path(__file__).resolve().parents[1] / "fixtures/reference.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceBatchingTests(unittest.TestCase):
    def test_incomplete_tail_is_dropped_and_step_can_cross_epoch(self):
        tokens = list(range(40))
        # Nine samples, two per microbatch: the ninth sample is dropped.
        # Three microbatches per step: step two crosses a four-batch epoch.
        offsets = reference.batch_token_offsets(40, 4, 2, 6, 2)
        self.assertEqual(offsets, [24, 0, 8])
        actual_inputs = [tokens[i : i + 8] for i in offsets]
        actual_labels = [tokens[i + 1 : i + 9] for i in offsets]
        self.assertEqual(
            actual_inputs,
            [
                [24, 25, 26, 27, 28, 29, 30, 31],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
            ],
        )
        self.assertEqual(
            actual_labels,
            [
                [25, 26, 27, 28, 29, 30, 31, 32],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
            ],
        )
        self.assertEqual(reference.batch_token_offsets(40, 4, 2, 6, 3), [16, 24, 0])

    def test_long_fixture_crosses_epoch_at_step_128(self):
        self.assertEqual(
            reference.batch_token_offsets(16384, 16, 2, 8, 128),
            [16256, 16288, 16320, 0],
        )
        self.assertEqual(
            reference.batch_token_offsets(16384, 16, 2, 8, 129), [32, 64, 96, 128]
        )

    def test_invalid_batch_contract_is_rejected(self):
        for arguments in ((40, 4, 2, 3, 1), (8, 4, 2, 2, 1), (40, 4, 2, 6, 0)):
            with self.assertRaises(ValueError):
                reference.batch_token_offsets(*arguments)


if __name__ == "__main__":
    unittest.main()

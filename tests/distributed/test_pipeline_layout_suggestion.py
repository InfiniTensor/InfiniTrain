#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "suggest_pipeline_layout.py"
SPEC = importlib.util.spec_from_file_location("pipeline_suggestion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PipelineLayoutSuggestionTest(unittest.TestCase):
    def test_balances_contiguous_costs(self):
        counts, stage_costs = MODULE.balanced_partition([10, 1, 1, 1, 1, 1], 2)
        self.assertEqual(counts, [1, 5])
        self.assertEqual(stage_costs, [10, 5])

    def test_reads_layer_profiler_records(self):
        records = """
0 2026-08-21 TransformerLayer.0 cuda:0 12 100 1
1 2026-08-21 TransformerLayer.1 cuda:0 15 20 1
2 2026-08-21 TransformerLayer.0 cuda:0 12 120 1
3 2026-08-21 TransformerLayer.1 cuda:0 15 40 1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.rank0"
            path.write_text(records, encoding="utf-8")
            self.assertEqual(MODULE.parse_profiler_records([str(path)]), [110, 30])

    def test_rejects_missing_profiler_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.rank0"
            path.write_text("0 now TransformerLayer.2 cuda:0 1 2 3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.parse_profiler_records([str(path)])


if __name__ == "__main__":
    unittest.main()

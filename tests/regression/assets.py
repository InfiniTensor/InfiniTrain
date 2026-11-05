"""Resolve versioned fixtures and case-local artifacts without machine paths."""

import importlib.util
import json
from pathlib import Path

from execution import file_sha256, safe_case_path


class Assets:
    def __init__(self, root, cache_dir, data_root=None, overrides=None):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.data_root = Path(data_root).expanduser().resolve() if data_root else None
        self.overrides = overrides or {}
        self.paths = {}
        self.hashes = {}
        self.manifest = json.loads(
            (self.root / "fixtures" / "manifest.json").read_text()
        )

    def prepare(self, cases):
        needed = sorted({fixture for case in cases for fixture in case["fixtures"]})
        if any(f.startswith("tiny.") for f in needed):
            tiny_dir = (
                self.cache_dir
                / "fixtures"
                / ("v" + str(self.manifest["fixture_version"]))
            )
            missing = any(
                not (tiny_dir / v["path"]).exists()
                for v in self.manifest["files"].values()
            )
            if missing:
                spec = importlib.util.spec_from_file_location(
                    "regression_generate_fixtures",
                    self.root / "fixtures" / "generate.py",
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.generate(tiny_dir)
        errors = []
        for fixture in needed:
            try:
                if fixture in self.overrides:
                    path = Path(self.overrides[fixture]).expanduser().resolve()
                    expected = (
                        self.manifest.get("files", {}).get(fixture, {}).get("sha256")
                    )
                elif fixture.startswith("tiny."):
                    item = self.manifest["files"][fixture]
                    path, expected = tiny_dir / item["path"], item["sha256"]
                elif fixture.startswith("classic."):
                    if self.data_root is None:
                        raise ValueError(
                            "set --data-root, INFINITRAIN_TEST_DATA_ROOT or fixture override"
                        )
                    manifest_path = self.root / "fixtures" / "classic.json"
                    mapping = json.loads(manifest_path.read_text())
                    item = mapping.get("files", mapping)[fixture]
                    path = self.data_root / item["path"]
                    expected = item.get("sha256")
                else:
                    raise ValueError("unknown fixture ID")
                if not path.is_file():
                    raise ValueError("file not found: " + str(path))
                digest = file_sha256(path, self.cache_dir / "hashes")
                if expected and expected != digest:
                    raise ValueError(
                        "checksum mismatch: expected {}, got {}".format(
                            expected, digest
                        )
                    )
                self.paths[fixture], self.hashes[fixture] = str(path), digest
            except (OSError, KeyError, ValueError) as error:
                errors.append("{}: {}".format(fixture, error))
        return errors

    def resolve(self, value, artifact_root, checkpoint_root=None):
        if not isinstance(value, str):
            return value
        if value.startswith("fixture:"):
            return self.paths[value[len("fixture:") :]]
        if value.startswith("artifact:"):
            relative = value[len("artifact:") :]
            root = (
                checkpoint_root
                if checkpoint_root is not None and "/checkpoints" in relative
                else artifact_root
            )
            return str(safe_case_path(root, relative))
        return value

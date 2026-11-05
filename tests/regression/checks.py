"""Fail-closed numerical contracts for training regression artifacts.

No third party packages are required. A PASS always means the requested data
was present, finite and compared; missing evidence never becomes an empty pass.
"""

from __future__ import annotations

import ast
import json
import io
import math
import re
import statistics
import struct
from dataclasses import dataclass, field
from pathlib import Path


class MetricError(ValueError):
    """An artifact is incomplete, ambiguous, corrupt or incompatible."""


DTYPES = {
    "fp32": "float32",
    "float32": "float32",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}
TOLERANCES = {"float32": (1e-5, 1e-5), "bfloat16": (1e-2, 1e-2)}
NUMBER = r"[+-]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
STEP = re.compile(r"\bstep\s+(\d+)/(\d+)\b", re.I)
FIELDS = {
    "loss": rf"train loss\s+({NUMBER})",
    "lr": rf"\blr\s+({NUMBER})",
    "step_ms": rf"({NUMBER})\s+ms\b",
    "tokens_per_second": rf"({NUMBER})\s+tok/s",
}


def result(status, reasons=(), **data):
    return {"status": status, "reasons": list(reasons), "data": data}


def normalize_dtype(dtype):
    if dtype not in DTYPES:
        raise MetricError(f"Unknown dtype: {dtype!r}")
    return DTYPES[dtype]


@dataclass
class Metrics:
    records: list[dict]
    dtype: str | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        return {"records": self.records, "dtype": self.dtype, "metadata": self.metadata}


def _finite(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise MetricError(f"{label} must be a finite number: {value!r}")


def _validate(records, *, expected_steps=None, required=("loss",)):
    if not records:
        raise MetricError("No training metrics found")
    found = []
    for record in records:
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise MetricError(f"Invalid global step: {step!r}")
        if step in found:
            raise MetricError(
                f"Duplicate step {step}; rank streams must be reduced explicitly"
            )
        found.append(step)
        for key in (
            "rank",
            "dp_rank",
            "tp_rank",
            "pp_rank",
            "consumed_samples",
            "global_tokens",
            "dp",
            "tp",
            "pp",
            "sp",
            "vpp",
            "world_size",
        ):
            minimum = (
                1
                if key in ("global_tokens", "dp", "tp", "pp", "sp", "vpp", "world_size")
                else 0
            )
            if key in record and (
                type(record[key]) is not int or record[key] < minimum
            ):
                raise MetricError(f"Step {step}: {key} must be an integer >= {minimum}")
        for key in required:
            if key not in record:
                raise MetricError(f"Step {step}: missing required metric {key}")
        if (
            "optimizer_stateful" in record
            and type(record["optimizer_stateful"]) is not bool
        ):
            raise MetricError(
                f"Step {step}: optimizer_stateful must be boolean metadata"
            )
        for key, value in record.items():
            if key not in (
                "dtype",
                "type",
                "event",
                "schema_version",
                "optimizer_stateful",
            ) and isinstance(value, (int, float)):
                _finite(value, f"step {step}/{key}")
        for key in required:
            _finite(record[key], f"step {step}/{key}")
        for key in ("step_ms", "tokens_per_second"):
            if key in record and record[key] <= 0:
                raise MetricError(f"Step {step}: {key} must be positive")
    if found != sorted(found):
        raise MetricError("Steps are out of order")
    expected = (
        list(expected_steps)
        if expected_steps is not None
        else list(range(found[0], found[-1] + 1))
    )
    if found != expected:
        raise MetricError(
            f"Incomplete/unexpected steps: expected {expected}, found {found}"
        )


def load_metrics(path, *, expected_steps=None, dtype=None, required=("loss",)):
    """Read canonical-rank JSONL or historical text logs. Raises MetricError.

    Text step denominators detect truncated logs even without expected_steps.
    For resumed text streams, callers must supply explicit expected_steps.
    """
    path = Path(path)
    records, metadata, observed_dtypes, totals = [], {}, set(), set()
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError) as error:
        raise MetricError(str(error)) from error
    structured = path.suffix == ".jsonl"
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if structured or line.lstrip().startswith("{"):
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as error:
                raise MetricError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise MetricError(f"{path}:{number}: metric must be an object")
            if "schema_version" in row and (
                type(row["schema_version"]) is not int or row["schema_version"] != 1
            ):
                raise MetricError(f"{path}:{number}: unsupported metric schema")
            if row.get("type") not in (None, "step", "metadata", "config", "complete"):
                raise MetricError(f"{path}:{number}: unsupported metric record type")
            if row.get("type") in ("metadata", "config") or row.get("event") in (
                "metadata",
                "config",
            ):
                metadata.update(row)
                if "dtype" in row:
                    observed_dtypes.add(normalize_dtype(row["dtype"]))
                continue
            if row.get("type") == "complete":
                if "complete" in metadata:
                    raise MetricError("Duplicate completion record")
                metadata["complete"] = row
                continue
            if "complete" in metadata:
                raise MetricError("Training records after completion")
            if "step" not in row:
                raise MetricError(f"{path}:{number}: missing step")
            if "dtype" in row:
                observed_dtypes.add(normalize_dtype(row["dtype"]))
            records.append(row)
        else:
            match = STEP.search(line)
            if not match:
                continue
            row = {"step": int(match[1])}
            totals.add(int(match[2]))
            for key, pattern in FIELDS.items():
                match_field = re.search(pattern, line, re.I)
                if match_field:
                    row[key] = float(match_field[1])
            memory = re.search(r"peak used:\s*(" + NUMBER + r")\s*MB", line, re.I)
            if memory:
                row["peak_memory_bytes"] = float(memory[1]) * 1048576
            records.append(row)
    if dtype is not None:
        observed_dtypes.add(normalize_dtype(dtype))
    if len(observed_dtypes) > 1:
        raise MetricError(f"Conflicting dtype metadata: {sorted(observed_dtypes)}")
    if len(totals) > 1:
        raise MetricError(f"Inconsistent declared training lengths: {sorted(totals)}")
    if expected_steps is None and totals:
        total = next(iter(totals))
        expected_steps = range(1, total + 1)
    _validate(records, expected_steps=expected_steps, required=required)
    return Metrics(records, next(iter(observed_dtypes), None), metadata)


def check_complete(metrics, expected_steps, required=("loss",)):
    try:
        _validate(metrics.records, expected_steps=expected_steps, required=required)
        return result("PASS", steps=len(metrics.records))
    except MetricError as error:
        return result("FAIL", [str(error)])


def _tolerances(dtype, atol, rtol):
    defaults = TOLERANCES[normalize_dtype(dtype)]
    values = (
        defaults[0] if atol is None else atol,
        defaults[1] if rtol is None else rtol,
    )
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise MetricError("Tolerances must be finite and nonnegative")
    return values


def compare_metrics(
    reference, candidate, *, dtype, metrics=("loss",), atol=None, rtol=None, steps=None
):
    try:
        if not metrics:
            raise MetricError("No requested comparison metrics")
        dtype = normalize_dtype(dtype)
        atol, rtol = _tolerances(dtype, atol, rtol)
        for label, run in (("reference", reference), ("candidate", candidate)):
            if run.dtype is not None and normalize_dtype(run.dtype) != dtype:
                raise MetricError(
                    f"{label} dtype {run.dtype} differs from configured {dtype}"
                )
            _validate(run.records, required=metrics)
        ref = {r["step"]: r for r in reference.records}
        cand = {r["step"]: r for r in candidate.records}
        selected = list(steps) if steps is not None else list(ref)
        if not selected:
            raise MetricError("Empty comparison step selection")
        if steps is None and ref.keys() != cand.keys():
            raise MetricError(
                f"Step sets differ: reference {list(ref)}, candidate {list(cand)}"
            )
        reasons, worst = [], 0.0
        for step in selected:
            if step not in ref or step not in cand:
                raise MetricError(f"Missing comparison step {step}")
            for metric in metrics:
                expected, actual = ref[step][metric], cand[step][metric]
                difference = abs(actual - expected)
                worst = max(worst, difference)
                exact = metric in (
                    "step",
                    "global_step",
                    "global_tokens",
                    "consumed_samples",
                    "data_position",
                )
                limit = 0 if exact else atol + rtol * abs(expected)
                if difference > limit:
                    reasons.append(
                        f"Step {step}/{metric}: {actual:.12g} vs {expected:.12g}; diff {difference:.4g} > {limit:.4g}"
                    )
        return result(
            "FAIL" if reasons else "PASS",
            reasons[:50],
            compared_steps=len(selected),
            max_absolute_error=worst,
            mismatch_count=len(reasons),
        )
    except (MetricError, KeyError, TypeError) as error:
        return result("FAIL", [str(error)])


def compare_resume(
    reference, resumed, *, dtype, resume_step, snapshot_paths=None, atol=None, rtol=None
):
    selected = [r["step"] for r in reference.records if r["step"] > resume_step]
    if [r["step"] for r in resumed.records] != selected:
        return result(
            "FAIL",
            [
                "Resumed stream must contain exactly the continuous run steps after the checkpoint"
            ],
        )
    checked = compare_metrics(
        reference,
        resumed,
        dtype=dtype,
        metrics=("loss", "lr", "consumed_samples"),
        steps=selected,
        atol=atol,
        rtol=rtol,
    )
    if snapshot_paths is None:
        return result(
            "FAIL",
            checked["reasons"]
            + ["Resume requires optimizer/parameter snapshot evidence"],
        )
    state = compare_snapshots(
        *snapshot_paths,
        dtype=dtype,
        required_kinds=("parameter", "optimizer"),
        atol=atol,
        rtol=rtol,
    )
    return result(
        "PASS" if checked["status"] == state["status"] == "PASS" else "FAIL",
        checked["reasons"] + state["reasons"],
        metrics=checked,
        snapshots=state,
    )


def _read_npy(path, data=None, *, with_dtype=False):
    """Decode one complete NPY record, preserving integer state without rounding.

    Bundled callers supply a bounded bytes/memoryview slice; old callers keep
    using a path. No numpy dependency is required for checking artifacts.
    """
    raw = Path(path).read_bytes() if data is None else data
    stream = io.BytesIO(raw)
    if stream.read(6) != b"\x93NUMPY":
        raise MetricError(f"Not a NPY record: {path}")
    version = stream.read(2)
    if len(version) != 2 or version[0] not in (1, 2, 3):
        raise MetricError(f"Invalid NPY version: {path}")
    length_bytes = 2 if version[0] == 1 else 4
    raw_length = stream.read(length_bytes)
    if len(raw_length) != length_bytes:
        raise MetricError(f"Truncated NPY header: {path}")
    length = int.from_bytes(raw_length, "little")
    if length > 1_000_000:
        raise MetricError("Unreasonable NPY header size")
    raw_header = stream.read(length)
    if len(raw_header) != length:
        raise MetricError(f"Truncated NPY header: {path}")
    header = ast.literal_eval(
        raw_header.decode("utf8" if version[0] == 3 else "latin1")
    )
    if not isinstance(header, dict) or header.get("fortran_order") is not False:
        raise MetricError("NPY header must describe a contiguous C-order tensor")
    shape = tuple(header["shape"])
    if any(type(size) is not int or size < 0 for size in shape):
        raise MetricError("NPY dimensions must be nonnegative integers")
    descriptor = header["descr"]
    suffixes = {
        "f2": ("e", "float16"),
        "f4": ("f", "float32"),
        "f8": ("d", "float64"),
        "i1": ("b", "int8"),
        "i2": ("h", "int16"),
        "i4": ("i", "int32"),
        "i8": ("q", "int64"),
        "b1": ("?", "bool"),
        "u1": ("B", "uint8"),
        "u2": ("H", "uint16"),
        "u4": ("I", "uint32"),
        "u8": ("Q", "uint64"),
    }
    if (
        not isinstance(descriptor, str)
        or len(descriptor) != 3
        or descriptor[0] not in "<>|="
        or descriptor[1:] not in suffixes
    ):
        raise MetricError(f"Unsupported NPY dtype {descriptor}")
    code, storage_dtype = suffixes[descriptor[1:]]
    if descriptor[0] == "|" and descriptor[2] != "1":
        raise MetricError(
            "Byte-order-independent NPY dtype is only valid for one-byte values"
        )
    byte_order = descriptor[0] if descriptor[0] in "<>=" else "<"
    fmt = byte_order + code
    payload = stream.read()
    if len(payload) != math.prod(shape) * struct.calcsize(fmt):
        raise MetricError(f"Truncated/extra NPY payload: {path}")
    values = [value[0] for value in struct.iter_unpack(fmt, payload)]
    if storage_dtype.startswith("float"):
        for value in values:
            _finite(value, str(path))
    return (shape, values, storage_dtype) if with_dtype else (shape, values)


def _join_shards(parts, axis, part_sizes=None):
    shapes = [shape for shape, _ in parts]
    if not shapes or axis < 0 or axis >= len(shapes[0]):
        raise MetricError("Invalid tensor shard axis")
    base = list(shapes[0])
    for shape in shapes:
        if len(shape) != len(base) or any(
            shape[i] != base[i] for i in range(len(base)) if i != axis
        ):
            raise MetricError("Incompatible shard shapes")
    count = len(parts)
    base[axis] = sum(shape[axis] for shape in shapes)
    inner, outer = math.prod(base[axis + 1 :]), math.prod(base[:axis])
    if part_sizes is None:
        part_sizes = [base[axis]]
    if sum(part_sizes) != base[axis] or any(size % count for size in part_sizes):
        raise MetricError(
            "Packed shard part_sizes must sum to global axis and divide by TP"
        )
    local_sizes = [size // count for size in part_sizes]
    if any(shape[axis] != sum(local_sizes) for shape in shapes):
        raise MetricError("Packed local shard axis size mismatch")
    merged = []
    for outer_index in range(outer):
        offset = 0
        for local_size in local_sizes:
            for shape, values in parts:
                begin = (outer_index * shape[axis] + offset) * inner
                merged.extend(values[begin : begin + local_size * inner])
            offset += local_size
    return tuple(base), merged


def _load_snapshot(
    path, expected_world_size=None, replica_tolerance=(0, 0), allowed_kinds=None
):
    path = Path(path)
    manifests = sorted(path.rglob("manifest.json")) if path.is_dir() else [path]
    if not manifests:
        raise MetricError(f"No snapshot manifests: {path}")
    grouped, steps, documents, ranks, tensor_dtypes = {}, set(), [], set(), {}
    bundle_cache = {}
    for manifest in manifests:
        manifest_parent = manifest.parent.resolve()
        resolved_paths = {}
        document = json.loads(manifest.read_text())
        if document.get("schema_version") != 1 or not isinstance(
            document.get("tensors"), list
        ):
            raise MetricError("Unsupported snapshot manifest")
        steps.add(document.get("step"))
        rank = document.get("rank", 0)
        if rank in ranks:
            raise MetricError("Duplicate rank manifest; select one snapshot step")
        ranks.add(rank)
        documents.append(document)
        local_keys = set()
        local_tensors = {
            (tensor["kind"], tensor["name"]): tensor
            for tensor in document["tensors"]
            if allowed_kinds is None or tensor["kind"] in allowed_kinds
        }
        bundle_ranges = {}

        def alias_root(key):
            seen = set()
            while local_tensors[key].get("alias_of"):
                if key in seen:
                    raise MetricError("Cyclic snapshot alias metadata")
                seen.add(key)
                key = (key[0], local_tensors[key]["alias_of"])
                if key not in local_tensors:
                    raise MetricError("Snapshot alias points to a missing tensor")
            return key

        for tensor in document["tensors"]:
            if allowed_kinds is not None and tensor["kind"] not in allowed_kinds:
                continue
            key = (tensor["kind"], tensor["name"])
            if key in local_keys:
                raise MetricError(f"Duplicate logical tensor {key} within rank {rank}")
            local_keys.add(key)
            raw_path = tensor["path"]
            if raw_path not in resolved_paths:
                tensor_path = (manifest_parent / raw_path).resolve()
                if manifest_parent not in tensor_path.parents:
                    raise MetricError("Snapshot path escapes artifact directory")
                resolved_paths[raw_path] = tensor_path
            tensor_path = resolved_paths[raw_path]
            bundled = "offset" in tensor or "length" in tensor
            data = None
            if bundled:
                offset, length = tensor.get("offset"), tensor.get("length")
                if (
                    type(offset) is not int
                    or type(length) is not int
                    or offset < 0
                    or length <= 0
                ):
                    raise MetricError(f"Invalid bundle offset/length: {key}")
                if tensor_path not in bundle_cache:
                    bundle_cache[tensor_path] = tensor_path.read_bytes()
                bundle = bundle_cache[tensor_path]
                if offset + length > len(bundle):
                    raise MetricError(f"Bundle record exceeds file bounds: {key}")
                data = memoryview(bundle)[offset : offset + length]
                bundle_ranges.setdefault(tensor_path, []).append(
                    (offset, offset + length, key)
                )
            shape, values, storage_dtype = _read_npy(tensor_path, data, with_dtype=True)
            if "dtype" in tensor and tensor["dtype"] != storage_dtype:
                raise MetricError(f"Manifest/NPY dtype mismatch: {key}")
            if key in tensor_dtypes and tensor_dtypes[key] != storage_dtype:
                raise MetricError(f"Tensor shard/replica dtypes differ: {key}")
            tensor_dtypes[key] = storage_dtype
            if list(shape) != tensor["shape"]:
                raise MetricError(f"Manifest/NPY shape mismatch: {key}")
            replica = (document.get("tp_rank", 0), document.get("pp_rank", 0))
            grouped.setdefault(key, []).append(
                (tensor.get("shard"), shape, values, replica)
            )
        for tensor_path, ranges in bundle_ranges.items():
            ordered = sorted(ranges)
            for previous, current in zip(ordered, ordered[1:]):
                if current[0] < previous[1]:
                    if current[:2] != previous[:2] or alias_root(
                        current[2]
                    ) != alias_root(previous[2]):
                        raise MetricError(
                            f"Overlapping snapshot bundle records: {tensor_path}"
                        )
    if len(steps) != 1 or not isinstance(next(iter(steps)), int):
        raise MetricError("Snapshots must have one explicit global step")
    declared_world_sizes = {
        document["world_size"] for document in documents if "world_size" in document
    }
    if expected_world_size is not None:
        declared_world_sizes.add(expected_world_size)
    if len(declared_world_sizes) > 1:
        raise MetricError("Snapshot world sizes disagree")
    expected_ranks = next(iter(declared_world_sizes), max(ranks) + 1)
    if ranks != set(range(expected_ranks)):
        raise MetricError("Missing rank snapshot manifest")
    if not grouped:
        raise MetricError("Empty tensor snapshot")
    tensors, replica_variants = {}, {}
    for key, entries in grouped.items():
        shard_counts = {
            entry[0].get("count", 1) if entry[0] else 1 for entry in entries
        }
        if len(shard_counts) != 1:
            raise MetricError(f"Inconsistent shard count: {key}")
        count = next(iter(shard_counts))
        if count < 1:
            raise MetricError("Invalid shard count")
        parts, descriptor, dp_copies, representatives = {}, None, {}, {}
        for shard, shape, values, replica in entries:
            index = shard.get("index", 0) if shard else 0
            if index < 0 or index >= count:
                raise MetricError("Invalid shard index")
            dp_key = (index, replica)
            if dp_key in dp_copies and dp_copies[dp_key] != (shape, values):
                raise MetricError(
                    f"Data-parallel replicas differ: {key}, shard {index}, TP/PP {replica}"
                )
            dp_copies[dp_key] = (shape, values)
            if index in parts and parts[index] != (shape, values):
                previous_replica = representatives[index]
                if (
                    previous_replica[1] != replica[1]
                    or previous_replica[0] == replica[0]
                ):
                    raise MetricError(
                        f"Replicated pipeline tensor differs: {key}, shard {index}"
                    )
                old_shape, old_values = parts[index]
                atol, rtol = replica_tolerance
                integral = tensor_dtypes[key] == "bool" or tensor_dtypes[
                    key
                ].startswith(("int", "uint"))
                if old_shape != shape or any(
                    abs(a - b) > (0 if integral else atol + rtol * abs(a))
                    for a, b in zip(old_values, values)
                ):
                    raise MetricError(
                        f"Tensor-parallel replicas exceed configured tolerance: {key}, shard {index}"
                    )
            if index not in parts:
                parts[index] = (shape, values)
                representatives[index] = replica
            if count > 1:
                candidate = (shard["axis"], shard.get("part_sizes"))
                if descriptor is not None and descriptor != candidate:
                    raise MetricError("Inconsistent shard layout")
                descriptor = candidate
        if set(parts) != set(range(count)):
            raise MetricError(f"Missing tensor shards: {key}")
        tensors[key] = (
            parts[0]
            if count == 1
            else _join_shards([parts[i] for i in range(count)], *descriptor)
        )
        if count == 1:
            # Preserve each TP representative for direct oracle comparison.
            # DP copies of a given TP/PP representative were checked bit-exact.
            replica_variants[key] = list(dp_copies.values())
    optimizer_contracts = {
        (document.get("optimizer_type"), document.get("optimizer_stateful"))
        for document in documents
    }
    if len(optimizer_contracts) != 1:
        raise MetricError("Snapshot ranks disagree on optimizer state contract")
    optimizer_type, optimizer_stateful = next(iter(optimizer_contracts))
    if optimizer_stateful is False and any(key[0] == "optimizer" for key in tensors):
        raise MetricError(
            "Stateless optimizer manifest contradicts stored optimizer tensors"
        )
    return {
        "step": next(iter(steps)),
        "ranks": sorted(ranks),
        "optimizer_type": optimizer_type,
        "optimizer_stateful": optimizer_stateful,
        "tensor_dtypes": tensor_dtypes,
        "replica_variants": replica_variants,
    }, tensors


def compare_snapshots(
    reference_path,
    candidate_path,
    *,
    dtype,
    atol=None,
    rtol=None,
    required_kinds=("parameter", "gradient"),
    expected_world_sizes=(None, None),
):
    try:
        if not required_kinds or not set(required_kinds).issubset(
            {"parameter", "gradient", "optimizer"}
        ):
            raise MetricError("Snapshot comparison needs known nonempty tensor kinds")
        atol, rtol = _tolerances(dtype, atol, rtol)
        ref_doc, reference = _load_snapshot(
            reference_path, expected_world_sizes[0], (atol, rtol), set(required_kinds)
        )
        cand_doc, candidate = _load_snapshot(
            candidate_path, expected_world_sizes[1], (atol, rtol), set(required_kinds)
        )
        if ref_doc.get("step") != cand_doc.get("step"):
            raise MetricError("Snapshot global steps differ")
        reference = {
            key: value for key, value in reference.items() if key[0] in required_kinds
        }
        candidate = {
            key: value for key, value in candidate.items() if key[0] in required_kinds
        }
        if reference.keys() != candidate.keys():
            raise MetricError("Logical tensor name/kind sets differ")
        kinds = {key[0] for key in reference}
        missing = set(required_kinds) - kinds
        if "optimizer" in required_kinds:
            if (ref_doc["optimizer_type"], ref_doc["optimizer_stateful"]) != (
                cand_doc["optimizer_type"],
                cand_doc["optimizer_stateful"],
            ):
                raise MetricError("Optimizer state contracts differ")
            if (
                ref_doc["optimizer_type"] == "sgd"
                and ref_doc["optimizer_stateful"] is False
            ):
                missing.discard("optimizer")
        if missing:
            raise MetricError(f"Missing snapshot kinds: {sorted(missing)}")
        reasons, count, worst = [], 0, 0.0
        for key, (shape, expected) in reference.items():
            storage_dtype = ref_doc["tensor_dtypes"][key]
            if storage_dtype != cand_doc["tensor_dtypes"][key]:
                raise MetricError(
                    f"Reference/candidate tensor storage dtypes differ: {key}"
                )
            exact = storage_dtype == "bool" or storage_dtype.startswith(("int", "uint"))
            expected_variants = ref_doc["replica_variants"].get(
                key, [(shape, expected)]
            )
            actual_variants = cand_doc["replica_variants"].get(key, [candidate[key]])
            for reference_shape, expected_values in expected_variants:
                for variant_index, (actual_shape, actual) in enumerate(actual_variants):
                    if reference_shape != actual_shape:
                        raise MetricError(f"Tensor shape differs: {key}")
                    for index, (a, b) in enumerate(zip(expected_values, actual)):
                        difference = abs(a - b)
                        worst = max(worst, difference)
                        count += 1
                        limit = 0 if exact else atol + rtol * abs(a)
                        if difference > limit and len(reasons) < 50:
                            reasons.append(
                                f"{key}[{index}] replica {variant_index}: {b!r} vs {a!r}"
                            )
        return result(
            "FAIL" if reasons else "PASS",
            reasons,
            elements=count,
            tensors=len(reference),
            max_absolute_error=worst,
        )
    except (
        MetricError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SyntaxError,
        struct.error,
    ) as error:
        return result("FAIL", [str(error)])


def performance_summary(runs, *, warmup=10, window=30, min_repeats=3):
    if warmup < 0 or window < 1 or min_repeats < 1:
        raise MetricError("Invalid performance sampling contract")
    if len(runs) < min_repeats:
        raise MetricError(
            f"Need {min_repeats} independent repetitions, received {len(runs)}"
        )
    medians, peaks, throughputs, within_jitter = [], [], [], []
    for run in runs:
        if run.metadata.get("oracle") or any(
            record.get("oracle") for record in run.records
        ):
            raise MetricError(
                "Independent numerical oracle placeholder timings are not performance evidence"
            )
        _validate(run.records, required=("step_ms",))
        selected = run.records[warmup : warmup + window]
        if len(selected) != window:
            raise MetricError(f"Need {warmup + window} steps per repetition")
        times = [record["step_ms"] for record in selected]
        median = statistics.median(times)
        medians.append(median)
        within_jitter.append((max(times) - min(times)) / median)
        if all("peak_memory_bytes" in record for record in selected):
            peaks.append(max(record["peak_memory_bytes"] for record in selected))
        if all("global_tokens" in record for record in selected):
            throughputs.append(
                sum(record["global_tokens"] for record in selected)
                / (sum(times) / 1000)
            )
    center = statistics.median(medians)
    jitter = (max(medians) - min(medians)) / center
    return {
        "step_ms": center,
        "repeat_medians": medians,
        "jitter": jitter,
        "within_run_jitter": max(within_jitter),
        "peak_memory_bytes": max(peaks) if len(peaks) == len(runs) else None,
        "tokens_per_second": (
            statistics.median(throughputs) if len(throughputs) == len(runs) else None
        ),
    }


def compare_performance(
    reference_runs,
    candidate_runs,
    *,
    warmup=10,
    window=30,
    min_repeats=3,
    threshold=0.05,
    max_jitter=0.05,
    memory_threshold=0.05,
    require_memory=True,
):
    try:
        for value in (threshold, max_jitter, memory_threshold):
            _finite(value, "performance threshold")
            if value < 0:
                raise MetricError("Performance thresholds must be nonnegative")
        options = dict(warmup=warmup, window=window, min_repeats=min_repeats)
        ref, cand = performance_summary(reference_runs, **options), performance_summary(
            candidate_runs, **options
        )
        if require_memory and (
            ref["peak_memory_bytes"] is None or cand["peak_memory_bytes"] is None
        ):
            raise MetricError("Missing peak memory evidence")
        if (
            max(
                ref["jitter"],
                cand["jitter"],
                ref["within_run_jitter"],
                cand["within_run_jitter"],
            )
            > max_jitter
        ):
            return result(
                "INCONCLUSIVE",
                ["Performance noise exceeds calibrated jitter limit"],
                reference=ref,
                candidate=cand,
            )
        slowdown = cand["step_ms"] / ref["step_ms"] - 1
        reasons = []
        if slowdown > threshold:
            reasons.append(
                f"Step time regression {slowdown:.2%} exceeds {threshold:.2%}"
            )
        if (
            ref["peak_memory_bytes"] is not None
            and cand["peak_memory_bytes"] is not None
        ):
            if ref["peak_memory_bytes"] <= 0 or cand["peak_memory_bytes"] <= 0:
                raise MetricError("Peak memory must be positive")
            growth = cand["peak_memory_bytes"] / ref["peak_memory_bytes"] - 1
            if growth > memory_threshold:
                reasons.append(
                    f"Peak memory regression {growth:.2%} exceeds {memory_threshold:.2%}"
                )
        return result(
            "FAIL" if reasons else "PASS",
            reasons,
            reference=ref,
            candidate=cand,
            slowdown=slowdown,
        )
    except (MetricError, ValueError, TypeError, KeyError) as error:
        return result("FAIL", [str(error)])


def load_distributed_metrics(prefix, *, expected_steps, dtype, world_size):
    """Validate every rank and reduce critical-path timings; never average ranks.

    Loss must already be globally DP-reduced by the training program. Select
    final pipeline stage and verify all tensor/data parallel replicas agree.
    """
    if world_size < 1:
        raise MetricError("world_size must be positive")
    streams = []
    expected_steps = list(expected_steps)
    if not expected_steps:
        raise MetricError("Empty expected step list")
    for rank in range(world_size):
        path = Path(str(prefix) + f".rank_{rank}.jsonl")
        run = load_metrics(
            path,
            expected_steps=expected_steps,
            dtype=dtype,
            required=("lr", "global_tokens", "step_ms", "consumed_samples"),
        )
        complete = run.metadata.get("complete", {})
        if complete.get("rank") != rank or complete.get("step") != expected_steps[-1]:
            raise MetricError(f"Rank {rank}: missing/mismatched completion record")
        if (
            "consumed_samples" in complete
            and complete["consumed_samples"] != run.records[-1]["consumed_samples"]
        ):
            raise MetricError(f"Rank {rank}: completion sample cursor mismatch")
        for record in run.records:
            if "world_size" in record and record["world_size"] != world_size:
                raise MetricError(
                    f"Rank {rank}: configured world size disagrees with stream"
                )
            if "dtype" not in record:
                raise MetricError(f"Rank {rank}: missing explicit dtype")
            if isinstance(record.get("rank"), bool) or record.get("rank") != rank:
                raise MetricError(f"Rank {rank}: stream rank mismatch")
            if record.get("pp_rank", 0) == record.get("pp", 1) - 1:
                _finite(record.get("loss"), f"rank {rank}/loss")
        streams.append(run)
    combined = []
    for index, step in enumerate(expected_steps):
        rows = [stream.records[index] for stream in streams]
        for key in (
            "lr",
            "global_tokens",
            "consumed_samples",
            "dtype",
            "dp",
            "tp",
            "pp",
            "sp",
        ):
            if len({row.get(key) for row in rows}) > 1:
                raise MetricError(f"Step {step}: ranks disagree on {key}")
        final = [row for row in rows if row.get("pp_rank", 0) == row.get("pp", 1) - 1]
        if not final:
            raise MetricError("No final pipeline rank loss")
        loss = final[0]["loss"]
        atol, rtol = TOLERANCES[normalize_dtype(dtype)]
        if any(abs(row["loss"] - loss) > atol + rtol * abs(loss) for row in final):
            raise MetricError(
                f"Step {step}: globally reduced losses disagree across ranks"
            )
        record = dict(final[0])
        record.pop("rank", None)
        record["step_ms"] = max(row["step_ms"] for row in rows)
        if all(
            "peak_memory_bytes" in row and row["peak_memory_bytes"] is not None
            for row in rows
        ):
            record["peak_memory_bytes"] = max(row["peak_memory_bytes"] for row in rows)
        record["tokens_per_second"] = record["global_tokens"] / (
            record["step_ms"] / 1000
        )
        combined.append(record)
    return Metrics(
        combined,
        normalize_dtype(dtype),
        {"world_size": world_size, "all_ranks_complete": True},
    )


def expected_learning_rate(step, args):
    """Independent closed-form oracle for the LR used by optimizer step (1-based)."""
    t = step - 1
    peak = float(args.get("learning_rate", 1e-4))
    minimum = float(args.get("min_lr", 0))
    style = args.get("lr_decay_style", "constant")
    warmup = int(args.get("lr_warmup_iters", 0))
    decay = int(args.get("lr_decay_iters", 0)) or int(args["num_iteration"])
    initial = float(args.get("lr_warmup_init", 0))
    if style == "none":
        return peak
    if decay <= warmup or warmup < 0:
        raise MetricError("Invalid LR schedule interval")
    if t < warmup:
        return initial + (peak - initial) * t / warmup
    if style == "constant":
        return peak
    fraction = min((t - warmup) / (decay - warmup), 1)
    if style == "linear":
        return peak + (minimum - peak) * fraction
    if style == "cosine":
        return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * fraction))
    if style == "inverse-square-root":
        return (
            minimum
            if t > decay
            else max(minimum, peak * math.sqrt(max(warmup, 1) / max(t, 1)))
        )
    raise MetricError(f"Unsupported LR style {style}")


def check_lr_schedule(metrics, args, *, atol=1e-9, rtol=1e-5):
    try:
        _validate(metrics.records, required=("lr",))
        errors = []
        for record in metrics.records:
            expected = expected_learning_rate(record["step"], args)
            if abs(record["lr"] - expected) > atol + rtol * abs(expected):
                errors.append(
                    f"Step {record['step']}: LR {record['lr']} != analytical {expected}"
                )
        return result("FAIL" if errors else "PASS", errors, steps=len(metrics.records))
    except (MetricError, ValueError, KeyError) as error:
        return result("FAIL", [str(error)])


def check_lora_frozen(initial_path, trained_path):
    try:
        _, initial = _load_snapshot(initial_path)
        _, trained = _load_snapshot(trained_path)
        initial = {
            key: value for key, value in initial.items() if key[0] == "parameter"
        }
        trained = {
            key: value for key, value in trained.items() if key[0] == "parameter"
        }
        if initial.keys() != trained.keys():
            raise MetricError("LoRA initial/final parameter sets differ")
        adapters = [
            key for key in initial if re.search(r"lora[_\.]?[ab](?:$|\.)", key[1], re.I)
        ]
        frozen = [key for key in initial if key not in adapters]
        if not adapters or not frozen:
            raise MetricError(
                "LoRA check requires both adapter and frozen base tensors"
            )
        errors = [
            f"Frozen base parameter changed: {key[1]}"
            for key in frozen
            if initial[key] != trained[key]
        ]
        changed = [key[1] for key in adapters if initial[key] != trained[key]]
        if not changed:
            errors.append("No LoRA adapter parameter changed")
        return result(
            "FAIL" if errors else "PASS",
            errors,
            frozen_tensors=len(frozen),
            changed_adapters=changed,
        )
    except (MetricError, OSError, ValueError, KeyError, TypeError) as error:
        return result("FAIL", [str(error)])


def compare_snapshot_trajectories(
    reference_root,
    candidate_root,
    *,
    steps,
    dtype,
    required_kinds=("parameter", "gradient"),
    expected_world_sizes=(None, None),
    atol=None,
    rtol=None,
    check_initial=False,
):
    """Compare complete logical tensors at every requested optimizer step.

    check_initial is for equivalent runs, not resumed runs (whose initial
    snapshot is the loaded checkpoint step). It proves globally equal weights.
    """
    steps = list(steps)
    if not steps or len(steps) != len(set(steps)):
        return result("FAIL", ["Snapshot trajectory needs a nonempty unique step list"])
    outcomes = []
    if check_initial:
        outcomes.append(
            compare_snapshots(
                Path(reference_root) / "step_0",
                Path(candidate_root) / "step_0",
                dtype=dtype,
                required_kinds=("parameter",),
                expected_world_sizes=expected_world_sizes,
                atol=0,
                rtol=0,
            )
        )
    for step in steps:
        checked = compare_snapshots(
            Path(reference_root) / f"step_{step}",
            Path(candidate_root) / f"step_{step}",
            dtype=dtype,
            required_kinds=required_kinds,
            expected_world_sizes=expected_world_sizes,
            atol=atol,
            rtol=rtol,
        )
        checked["data"]["step"] = step
        outcomes.append(checked)
    reasons = [reason for outcome in outcomes for reason in outcome["reasons"]]
    return result(
        "PASS" if all(outcome["status"] == "PASS" for outcome in outcomes) else "FAIL",
        reasons[:50],
        steps=steps,
        checks=outcomes,
    )


def check_memory_growth(
    metrics,
    *,
    warmup=10,
    window=20,
    metric="peak_memory_bytes",
    max_growth_bytes=64 * 1024 * 1024,
    max_growth_fraction=0.1,
    min_increasing_fraction=0.75,
):
    """Detect sustained allocator-memory drift after warmup, not every leak.

    Per-step peaks are reset by training. Compare robust first/last windows,
    then require continued final-window slope and mostly nondecreasing samples
    before labeling a sustained leak. A one-time allocator pool expansion is
    reported separately as INCONCLUSIVE if it exceeds the budget.
    """
    try:
        if (
            warmup < 0
            or window < 2
            or max_growth_bytes < 0
            or not 0 <= max_growth_fraction
        ):
            raise MetricError("Invalid memory growth contract")
        if not 0 <= min_increasing_fraction <= 1:
            raise MetricError("Invalid sustained-growth fraction")
        _finite(max_growth_bytes, "memory growth bytes")
        _finite(max_growth_fraction, "memory growth fraction")
        _validate(metrics.records, required=(metric,))
        values = [row[metric] for row in metrics.records[warmup:]]
        if len(values) < window * 2:
            raise MetricError(
                f"Memory drift needs warmup + two windows ({warmup + window * 2} steps)"
            )
        if min(values) <= 0:
            raise MetricError("Memory evidence must be positive")
        initial, final = statistics.median(values[:window]), statistics.median(
            values[-window:]
        )
        growth = final - initial
        allowance = max(max_growth_bytes, initial * max_growth_fraction)
        differences = [right - left for left, right in zip(values, values[1:])]
        fraction = sum(delta >= 0 for delta in differences) / len(differences)
        slope = statistics.median(
            [
                right - left
                for left, right in zip(values[-window:], values[-window + 1 :])
            ]
        )
        slope_limit = allowance / (len(values) - window)
        data = dict(
            metric=metric,
            initial_window_median=initial,
            final_window_median=final,
            growth_bytes=growth,
            allowance_bytes=allowance,
            final_slope_bytes_per_step=slope,
            slope_limit_bytes_per_step=slope_limit,
            nondecreasing_fraction=fraction,
            interpretation="allocator peak drift; does not establish absence of all leaks",
        )
        if growth > allowance:
            if slope > slope_limit and fraction >= min_increasing_fraction:
                return result(
                    "FAIL", ["Sustained allocator memory growth exceeds budget"], **data
                )
            return result(
                "INCONCLUSIVE",
                [
                    "Memory grew beyond budget without a stable leak slope; inspect allocator expansion/noise"
                ],
                **data,
            )
        return result("PASS", **data)
    except (MetricError, ValueError, TypeError, KeyError) as error:
        return result("FAIL", [str(error)])

"""Deterministic, dependency-free regression configuration and coverage planning.

No machine paths or commands are evaluated here. Fixtures and artifacts remain
symbolic until the executor resolves them. New model manifests and scenarios are
loaded by discovery; the scheduler never needs a model-name switch.
"""

import copy
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

try:
    from . import tolerances
except ImportError:
    import tolerances

ROOT = Path(__file__).resolve().parent
SCENARIO_KEYS = {
    "schema_version",
    "id",
    "models",
    "profile",
    "tags",
    "suites",
    "nproc_per_node",
    "checks",
    "templates",
    "args",
    "matrix",
    "strategy",
    "topology",
    "batch",
    "timeout_seconds",
    "reference",
    "performance",
    "high_risk",
    "repetitions",
}
MODEL_KEYS = {
    "schema_version",
    "id",
    "binary",
    "adapter",
    "backends",
    "backend_dtypes",
    "backend_dtype_reasons",
    "dtypes",
    "argument_types",
    "argument_choices",
    "defaults",
    "capabilities",
    "profiles",
}


class PlanError(ValueError):
    """A configuration cannot be safely executed."""


class UnsupportedCombination(PlanError):
    """Well-typed factors violate an explicitly supported combination constraint."""


def _is_managed_path(key):
    return key in {"save", "load"} or key.endswith(
        ("_bin", "_path", "_filepath", "_file", "_dir")
    )


def _validate_paths(case, merged):
    """Every consumed input asset is symbolic and enters the fixture fingerprint."""
    inputs = {
        "input_bin",
        "input_val_bin",
        "tokenizer_bin",
        "llmc_filepath",
        "lora_load_path",
    }
    outputs = {
        "save",
        "load",
        "lora_save_path",
        "metrics_file",
        "regression_snapshot_dir",
    }
    for key, value in merged.items():
        if not _is_managed_path(key) or value == "":
            continue
        if not isinstance(value, str):
            raise PlanError("managed path must be a string: " + key)
        if key in inputs:
            if not value.startswith("fixture:"):
                raise PlanError(
                    key + " must use fixture:ID; use --fixture ID=PATH for local files"
                )
            fixture = value[len("fixture:") :]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", fixture) or fixture in {".", ".."}:
                raise PlanError("invalid fixture ID for " + key)
        elif key in outputs:
            if case["profile"] == "classic" and value.startswith("@CKPT_ROOT_DIR@/"):
                continue  # Converted into an artifact dependency before final validation.
            if not value.startswith("artifact:"):
                raise PlanError(key + " must use a run-local artifact: path")
            relative = value[len("artifact:") :]
            if "\\" in relative or any(
                part in {"", ".", ".."} for part in relative.split("/")
            ):
                raise PlanError("unsafe artifact path for " + key)
        else:
            raise PlanError(
                "new path argument needs an explicit fixture/artifact adapter: " + key
            )
    case["fixtures"] = sorted(
        {
            value[len("fixture:") :]
            for value in merged.values()
            if isinstance(value, str) and value.startswith("fixture:")
        }
    )


def _argument_types(args, model):
    if not isinstance(args, dict):
        raise PlanError("args must be an object")
    types = model["argument_types"]
    for key, value in args.items():
        if key not in types:
            raise PlanError("%s: unknown argument %s" % (model["id"], key))
        typ = types[key]
        if typ not in {
            "bool",
            "string",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "double",
            "float",
        }:
            raise PlanError("invalid manifest argument type %s for %s" % (typ, key))
        if typ == "bool" and not isinstance(value, bool):
            raise PlanError("%s must be boolean" % key)
        if typ.startswith(("int", "uint")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise PlanError("%s must be integer" % key)
            bits = int(re.search(r"\d+", typ).group())
            low, high = (
                (0, 2**bits - 1)
                if typ.startswith("uint")
                else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            )
            if not low <= value <= high:
                raise PlanError("%s is outside %s range" % (key, typ))
        if typ in ("double", "float"):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or abs(value) > 1.7976931348623157e308
                or not math.isfinite(value)
            ):
                raise PlanError("%s must be a finite number" % key)
        if typ == "string" and not isinstance(value, str):
            raise PlanError("%s must be string" % key)
        if typ == "string" and ("\x00" in value or "\n" in value or "\r" in value):
            raise PlanError("%s contains control characters" % key)
        if (
            key in model.get("argument_choices", {})
            and value not in model["argument_choices"][key]
        ):
            raise PlanError("unsupported %s value %r" % (key, value))
    if args.get("dtype", model["defaults"].get("dtype")) not in model["dtypes"]:
        raise PlanError("unsupported dtype")
    if args.get("zero_stage", 0) not in model["capabilities"]["zero_stages"]:
        raise PlanError("ZeRO stage unsupported (ZeRO-3 is not implemented)")


def _object(value, allowed, label):
    if not isinstance(value, dict):
        raise PlanError("%s must be an object" % label)
    unknown = set(value) - allowed
    if unknown:
        raise PlanError("%s: unknown fields %s" % (label, sorted(unknown)))


def _read(path):
    try:
        value = json.loads(
            path.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                PlanError("nonfinite JSON constant " + value)
            ),
        )
    except (ValueError, OSError) as exc:
        raise PlanError("%s: %s" % (path, exc)) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PlanError("%s: schema_version must be 1" % path)
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlanError("%s must be a positive integer" % name)
    return value


def config_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def pair_keys(row):
    return {
        json.dumps([[a, row[a]], [b, row[b]]], separators=(",", ":"))
        for a, b in itertools.combinations(sorted(row), 2)
    }


def pairwise(candidates, anchors=()):
    """Stable greedy set cover over *legal* candidates (algorithm greedy-v1)."""
    rows = sorted(candidates, key=lambda x: json.dumps(x, sort_keys=True))
    universe = set().union(*(pair_keys(x) for x in rows)) if rows else set()
    uncovered = universe.copy()
    selected = []
    for anchor in anchors:
        if anchor not in rows:
            raise PlanError("high-risk anchor is not a legal matrix point: %s" % anchor)
        if anchor not in selected:
            selected.append(anchor)
            uncovered -= pair_keys(anchor)
    while uncovered:
        best = max(rows, key=lambda x: len(pair_keys(x) & uncovered))
        if not pair_keys(best) & uncovered:
            raise PlanError("internal error: unreachable pairwise obligation")
        selected.append(best)
        uncovered -= pair_keys(best)
    if not selected and rows:
        selected.append(rows[0])
    return selected, sorted(universe)


def batch_args(adapter, batch, topology):
    """Convert explicit sample/token units into model-specific CLI arguments."""
    _object(
        batch, {"micro_batch_samples", "sequence_length", "accumulation_steps"}, "batch"
    )
    micro = _positive(batch.get("micro_batch_samples", 1), "micro_batch_samples")
    seq = _positive(batch.get("sequence_length", 16), "sequence_length")
    gas = _positive(batch.get("accumulation_steps", 1), "accumulation_steps")
    dp = _positive(topology.get("dp", 1), "dp")
    samples = micro * dp * gas
    args = {"sequence_length": seq}
    if adapter == "dense":
        args.update(batch_size=micro, total_batch_size=samples * seq)
    elif adapter == "mixtral":
        args.update(micro_batch_size=micro, global_batch_size=samples)
    else:
        raise PlanError("unknown adapter %s" % adapter)
    return args


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanError("%s must be a nonnegative integer" % name)
    return value


def _finite_budget(value, name, *, fraction=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or abs(value) > 1.7976931348623157e308
        or not math.isfinite(value)
        or value < 0
    ):
        raise PlanError("%s must be finite and nonnegative" % name)
    if fraction and value > 1:
        raise PlanError("%s must lie in [0,1]" % name)


def _resolve_case_tolerances(case, dtype, backend, registry=None):
    defaults = {"equivalent": "distributed_equivalence", "resume": "distributed_equivalence",
                "oracle": "oracle", "lr_schedule": "lr_schedule"}
    aliases = {"learning_rate": "lr", "gradients": "gradient", "updated_parameters": "parameter"}
    resolved, names, differs = {}, {}, False
    try:
        registry = tolerances.load_registry(registry)
        for spec in case["checks"]:
            kind = spec["kind"]
            if kind not in defaults:
                continue
            name = spec.get("tolerance", defaults[kind])
            if not isinstance(name, str):
                raise PlanError("tolerance must name a registered policy")
            if kind in {"equivalent", "resume"}:
                metrics = spec.get("metrics", ["loss"])
            elif kind == "oracle":
                metrics = ["loss", "gradient", "parameter"]
            elif kind == "lr_schedule":
                metrics = ["lr"]
            else:
                metrics = ["loss"]
            names[kind], resolved[kind] = name, {}
            for metric in sorted({aliases.get(metric, metric) for metric in metrics}):
                value = tolerances.resolve(name, dtype, metric, backend, registry)
                historical = tolerances.resolve_historical_default(defaults[kind], dtype, metric, backend)
                resolved[kind][metric] = value
                differs = differs or value != historical
    except (ValueError, OSError) as error:
        raise PlanError("invalid tolerance policy for %s: %s" % (case["id"], error)) from error
    case["tolerance_policies"] = names
    case["resolved_tolerances"] = resolved
    return differs


def _validate_checks(case, model, merged, world, tolerance_registry=None):
    checks = case["checks"]
    if not isinstance(checks, list) or not checks:
        raise PlanError("checks must be a nonempty list")
    performance_keys = {
        "warmup_steps",
        "measure_steps",
        "repetitions",
        "threshold",
        "max_jitter",
        "memory_threshold",
    }
    fields = {
        "finite_and_complete": set(),
        "history": set(),
        "lr_schedule": {"tolerance"},
        "lora_frozen": set(),
        "oracle": {"tolerance"},
        "equivalent": {"reference_case", "metrics", "tolerance"},
        "resume": {"reference_case", "metrics", "tolerance"},
        "performance": performance_keys,
        "memory_growth": {
            "metric",
            "warmup_steps",
            "window_steps",
            "max_growth_bytes",
            "max_growth_fraction",
            "min_increasing_fraction",
        },
    }
    steps = _positive(merged["num_iteration"], "num_iteration")
    start = _positive(case.get("start_step", 1), "start_step")
    available = steps - start + 1
    if available < 1:
        raise PlanError("resume start exceeds requested final iteration")
    _positive(case["timeout_seconds"], "timeout_seconds")
    _positive(case.get("repetitions", 1), "repetitions")
    seen = set()
    for spec in checks:
        if not isinstance(spec, dict) or spec.get("kind") not in fields:
            raise PlanError("unknown check kind in " + case["id"])
        kind = spec["kind"]
        if kind in seen:
            raise PlanError("duplicate check kind " + kind)
        seen.add(kind)
        _object(spec, fields[kind] | {"kind"}, kind + " check")
        if kind in {"equivalent", "resume"}:
            ref = spec.get("reference_case")
            if (
                not isinstance(ref, str)
                or not ref
                or "\\" in ref
                or any(part in ("", ".", "..") for part in ref.split("/"))
            ):
                raise PlanError("invalid reference_case")
            metrics = spec.get("metrics", ["loss"])
            allowed = {
                "loss",
                "lr",
                "learning_rate",
                "consumed_samples",
                "gradient",
                "gradients",
                "parameter",
                "updated_parameters",
                "optimizer",
            }
            if (
                not isinstance(metrics, list)
                or not metrics
                or any(
                    not isinstance(metric, str) or metric not in allowed
                    for metric in metrics
                )
                or len(set(metrics)) != len(metrics)
            ):
                raise PlanError(
                    "reference metrics must be a nonempty unique supported list"
                )
        elif kind == "performance":
            perf = case.get("performance")
            _object(perf, performance_keys, "performance")
            for key in ["warmup_steps", "measure_steps", "repetitions"]:
                if key not in perf:
                    raise PlanError("performance requires " + key)
            warmup = _nonnegative(perf["warmup_steps"], "warmup_steps")
            window = _positive(perf["measure_steps"], "measure_steps")
            repetitions = _positive(perf["repetitions"], "performance repetitions")
            if repetitions < 3:
                raise PlanError(
                    "performance requires at least three independent repetitions"
                )
            if case.get("repetitions", repetitions) != repetitions:
                raise PlanError(
                    "top-level repetitions disagree with performance contract"
                )
            if available < warmup + window:
                raise PlanError(
                    "performance iterations do not cover warmup + measurement window"
                )
            for key, value in spec.items():
                if key != "kind":
                    _finite_budget(value, key)
                if key != "kind" and (key not in perf or value != perf[key]):
                    raise PlanError(
                        "performance check and sampling contract disagree on " + key
                    )
            for key in ["threshold", "max_jitter", "memory_threshold"]:
                if key in perf:
                    _finite_budget(perf[key], key, fraction=True)
        elif kind == "memory_growth":
            if spec.get("metric", "peak_memory_bytes") != "peak_memory_bytes":
                raise PlanError(
                    "memory_growth supports per-step peak_memory_bytes only"
                )
            warmup = _nonnegative(spec.get("warmup_steps", 10), "memory warmup_steps")
            window = _positive(spec.get("window_steps", 20), "memory window_steps")
            if window < 2 or available < warmup + 2 * window:
                raise PlanError(
                    "memory_growth needs warmup plus two windows of at least two steps"
                )
            for key, default in [
                ("max_growth_bytes", 67108864),
                ("max_growth_fraction", 0.1),
                ("min_increasing_fraction", 0.75),
            ]:
                _finite_budget(
                    spec.get(key, default),
                    key,
                    fraction=key == "min_increasing_fraction",
                )
        elif kind == "lora_frozen":
            if merged.get("lora_rank", 0) <= 0:
                raise PlanError("lora_frozen requires enabled adapters")
        elif kind == "lr_schedule":
            if "lr_decay_style" not in merged:
                raise PlanError("lr_schedule is unsupported by this model")
        elif kind == "oracle":
            shape = model["profiles"][case["profile"]]["shape"]
            expected = {
                "layers": 4,
                "hidden": 32,
                "heads": 4,
                "max_sequence_length": 64,
            }
            fixture_ids = {
                "tiny." + model["id"] + ".tokens",
                "tiny." + model["id"] + ".weights",
            }
            if (
                case["profile"] != "tiny"
                or any(shape.get(key) != value for key, value in expected.items())
                or set(case["fixtures"]) != fixture_ids
            ):
                raise PlanError(
                    "oracle requires the fixed versioned tiny fixture architecture"
                )
            if (
                merged["dtype"] != "float32"
                or world != 1
                or merged.get("zero_stage", 0) != 0
            ):
                raise PlanError("oracle requires FP32 single-rank nonsharded execution")
            if (
                merged.get("lora_rank", 0) != 0
                or merged.get("lora_load_path", "")
                or merged.get("load", "")
            ):
                raise PlanError("oracle does not support LoRA or checkpoint resume")
            if (
                merged.get("lr_decay_style", "constant") not in {"none", "constant"}
                or merged.get("lr_warmup_iters", 0) != 0
            ):
                raise PlanError("oracle requires constant learning rate without warmup")
            seq = _positive(merged["sequence_length"], "sequence_length")
            micro_key = (
                "batch_size" if model["adapter"] == "dense" else "micro_batch_size"
            )
            samples = _positive(merged[micro_key], micro_key)
            # The independent reference reproduces accumulation and epoch
            # wrapping; its fixed token fixture still needs one complete batch.
            if (16384 // seq - 1) // samples < 1:
                raise PlanError("oracle token fixture has no complete microbatch")
    if "performance" in case and not seen.intersection({"performance", "history"}):
        raise PlanError("performance sampling declared without a performance check")
    return _resolve_case_tolerances(case, merged["dtype"], merged["device"], tolerance_registry)


def validate_case(case, model, tolerance_registry=None):
    args = case["args"]
    merged = dict(model["defaults"], **args)
    _argument_types(merged, model)
    if merged.get("activation_recompute", False):
        granularity = merged.get("recompute_granularity", "full")
        if granularity == "selective":
            raise PlanError("selective activation recompute is not implemented")
        if granularity == "none":
            raise PlanError("activation_recompute requires full granularity")
        if merged.get("recompute_method", "none") in ("uniform", "block"):
            _positive(merged.get("recompute_num_layers", 0), "recompute_num_layers")
    _validate_paths(case, merged)
    if case["dtype"] not in model["dtypes"]:
        raise PlanError("unsupported dtype %s" % case["dtype"])
    if args["device"] not in model["backends"]:
        raise PlanError("unsupported backend %s" % args["device"])
    nproc = _positive(case["nproc_per_node"], "nproc_per_node")
    threads = _positive(merged.get("nthread_per_process", 1), "nthread_per_process")
    case["nthread_per_process"] = threads
    case["gpus"] = 0 if args["device"] == "cpu" else nproc * threads
    world = nproc * threads
    policy_deviation = _validate_checks(case, model, merged, world, tolerance_registry)
    backend_dtypes = model.get("backend_dtypes", {}).get(
        args["device"], model["dtypes"]
    )
    if merged["dtype"] not in backend_dtypes:
        reason = model.get("backend_dtype_reasons", {}).get(
            args["device"], "dtype is not supported by this backend"
        )
        raise UnsupportedCombination(
            "%s/%s: %s" % (args["device"], merged["dtype"], reason)
        )
    tp = _positive(merged.get("tensor_parallel", 1), "tensor_parallel")
    pp = _positive(merged.get("pipeline_parallel", 1), "pipeline_parallel")
    vpp = _positive(
        merged.get("virtual_pipeline_parallel", 1), "virtual_pipeline_parallel"
    )
    if world % (tp * pp):
        raise UnsupportedCombination("world size must be divisible by TP*PP")
    ep = _positive(merged.get("expert_parallel", 1), "expert_parallel")
    if model["adapter"] == "mixtral":
        if tp != 1 or merged.get("expert_tensor_parallel", 1) != 1:
            raise UnsupportedCombination("Mixtral requires TP=1 and expert TP=1")
        if (world // (tp * pp)) % ep:
            raise UnsupportedCombination("EP must divide the data parallel size")
    if world > 1 and not model["capabilities"]["distributed"]:
        raise UnsupportedCombination("model does not support distributed execution")
    if merged.get("zero_stage", 0) > 0 and world // (tp * pp) < 2:
        raise UnsupportedCombination(
            "DistributedOptimizer requires DP > 1 for ZeRO-1/2"
        )
    if args["device"] == "cpu" and world > 1:
        raise UnsupportedCombination("distributed training requires a GPU backend")
    if merged.get("zero_stage", 0) not in model["capabilities"]["zero_stages"]:
        raise PlanError("ZeRO stage unsupported (ZeRO-3 is not implemented)")
    shape = model["profiles"][case["profile"]]["shape"]
    if shape["heads"] % tp or shape["hidden"] % tp:
        raise UnsupportedCombination("TP must divide attention heads and hidden size")
    # Current classic GPT-2 PP8 splits 12 layers unevenly; runtime explicitly supports this.
    if pp > shape["layers"] or vpp > 1 and pp == 1:
        raise UnsupportedCombination("invalid pipeline layer/chunk layout")
    if case["profile"] != "classic" and shape["layers"] % (pp * vpp):
        raise UnsupportedCombination("tiny pipeline chunks must divide layer count")
    seq = _positive(merged.get("sequence_length", 64), "sequence_length")
    if seq > shape.get("max_sequence_length", 2**32 - 1):
        raise UnsupportedCombination(
            "sequence length exceeds fixture positional capacity"
        )
    if merged.get("sequence_parallel", False) and (tp == 1 or seq % tp):
        raise UnsupportedCombination(
            "SP requires TP>1 and sequence length divisible by TP"
        )
    if model["adapter"] == "dense":
        micro = _positive(merged["batch_size"], "batch_size")
        global_tokens = _positive(merged["total_batch_size"], "total_batch_size")
        unit = micro * seq * (world // (tp * pp))
        if global_tokens % unit:
            raise PlanError(
                "total_batch_size tokens must be divisible by microbatch*sequence*DP"
            )
        gas = global_tokens // unit
        samples = global_tokens // seq
    else:
        micro = _positive(merged["micro_batch_size"], "micro_batch_size")
        samples = _positive(merged["global_batch_size"], "global_batch_size")
        if samples % (micro * (world // (tp * pp))):
            raise PlanError("global_batch_size samples must divide into micro batches")
        gas = samples // (micro * (world // (tp * pp)))
        global_tokens = samples * seq
    _positive(merged["num_iteration"], "num_iteration")
    case["expected_steps"] = merged["num_iteration"]
    case["dtype"] = merged["dtype"]
    if case["start_step"] > case["expected_steps"]:
        raise PlanError("resume start exceeds requested final iteration")
    _positive(case["timeout_seconds"], "timeout_seconds")
    _positive(case.get("repetitions", 1), "repetitions")
    case["tags"] = [
        t
        for t in case["tags"]
        if t
        not in {
            "zero0",
            "zero1",
            "zero2",
            "tp",
            "pp",
            "sp",
            "vpp",
            "processes",
            "threads",
        }
    ]
    for tag, enabled in [
        ("zero%d" % merged.get("zero_stage", 0), True),
        ("tp", tp > 1),
        ("pp", pp > 1),
        ("sp", merged.get("sequence_parallel", False)),
        ("vpp", vpp > 1),
        ("accumulation", gas > 1),
        ("processes", nproc > 1),
        ("threads", nproc == 1),
    ]:
        if enabled and tag not in case["tags"]:
            case["tags"].append(tag)
    case["batch"] = {
        "micro_batch_samples": micro,
        "sequence_length": seq,
        "accumulation_steps": gas,
        "global_samples": samples,
        "global_tokens": global_tokens,
    }
    case["topology"] = {
        "dp": world // (tp * pp),
        "tp": tp,
        "pp": pp,
        "vpp": vpp,
        "launch": "processes" if nproc > 1 else "threads",
    }
    case["resolved_args"] = merged
    semantic = {
        "model": model["id"], "profile": case["profile"], "args": merged,
        "nproc_per_node": nproc, "fixtures": case["fixtures"], "checks": case["checks"],
    }
    # Preserve approved legacy keys only when numerical behavior is exactly the
    # immutable historical defaults. Any used JSON-policy deviation is semantic.
    if policy_deviation:
        semantic["resolved_tolerances"] = case["resolved_tolerances"]
    case["config_hash"] = config_hash(semantic)
    return case


def _make_case(model, scenario, args, suffix, backend, profile=None, nproc=None):
    profile = profile or scenario["profile"]
    if profile not in model["profiles"]:
        raise PlanError("unknown profile %s for %s" % (profile, model["id"]))
    spec = model["profiles"][profile]
    full_args = dict(spec.get("args", {})) if profile != "classic" else {}
    full_args.update(args)
    for key, fixture in spec["fixtures"].items():
        if key in full_args and full_args[key] != "fixture:" + fixture:
            raise PlanError(
                "scenario changes profile fixture "
                + key
                + "; declare a profile or use --fixture ID=PATH"
            )
    full_args.update({k: "fixture:" + v for k, v in spec["fixtures"].items()})
    full_args["device"] = backend
    for key, value in list(full_args.items()):
        if isinstance(value, str) and ("LORA_WEIGHTS_DIR" in value):
            full_args[key] = "fixture:classic.%s.lora" % model["id"]
    cid = "%s/%s/%s" % (scenario["id"], model["id"], suffix)
    if (
        not isinstance(suffix, str)
        or not suffix
        or any(part in ("", ".", "..") for part in suffix.split("/"))
        or "\\" in suffix
    ):
        raise PlanError("unsafe case suffix " + str(suffix))
    proc = scenario.get("nproc_per_node", 1) if nproc is None else nproc
    case = {
        "id": cid,
        "model": model["id"],
        "binary": model["binary"],
        "profile": profile,
        "args": full_args,
        "nproc_per_node": proc,
        "nthread_per_process": full_args.get("nthread_per_process", 1),
        "gpus": (
            proc * full_args.get("nthread_per_process", 1) if backend != "cpu" else 0
        ),
        "dtype": full_args.get("dtype", model["defaults"].get("dtype", "float32")),
        "expected_steps": full_args.get(
            "num_iteration", model["defaults"]["num_iteration"]
        ),
        "start_step": 1,
        "tags": list(scenario.get("tags", [])),
        "checks": copy.deepcopy(
            scenario.get("checks", [{"kind": "finite_and_complete"}])
        ),
        "dependencies": [],
        "timeout_seconds": scenario.get(
            "timeout_seconds", 1800 if profile == "classic" else 300
        ),
        "fixtures": sorted(
            v[len("fixture:") :]
            for v in full_args.values()
            if isinstance(v, str) and v.startswith("fixture:")
        ),
        "selection_reason": ["suite membership"],
        "scenario": scenario["id"],
    }
    for key in ("performance", "repetitions"):
        if key in scenario:
            case[key] = copy.deepcopy(scenario[key])
    return case


def _expand(scenario, model, backend, tolerance_registry=None):
    output, coverage = [], None
    if "templates" in scenario:
        for template in scenario["templates"]:
            _object(template, {"args", "variants"}, "template")
            for variant in template["variants"]:
                _object(variant, {"id", "args"}, "variant")
                case = _make_case(
                    model,
                    scenario,
                    dict(template.get("args", {}), **variant.get("args", {})),
                    variant["id"],
                    backend,
                )
                case["legacy_id"] = variant["id"]
                output.append(case)
        return output, coverage
    managed = {
        "batch_size",
        "total_batch_size",
        "micro_batch_size",
        "global_batch_size",
        "sequence_length",
        "tensor_parallel",
        "pipeline_parallel",
        "virtual_pipeline_parallel",
        "nthread_per_process",
    }
    conflicts = set(scenario.get("args", {})) & managed
    if conflicts:
        raise PlanError(
            "use batch/topology instead of derived arguments: " + str(sorted(conflicts))
        )
    matrix = scenario.get("matrix", {})
    if not isinstance(matrix, dict):
        raise PlanError("matrix must be an object")
    known_factors = set(model["argument_types"]) | {
        "dp",
        "tp",
        "pp",
        "vpp",
        "launch",
        "micro_batch_samples",
        "sequence_length",
        "accumulation_steps",
    }
    if set(matrix) - known_factors:
        raise PlanError(
            "unknown matrix factors " + str(sorted(set(matrix) - known_factors))
        )
    if "reference" in scenario:
        _object(
            scenario["reference"],
            {"args", "single_device", "resume_step", "metrics", "tolerance"},
            "reference",
        )
        if not isinstance(scenario["reference"].get("single_device", False), bool):
            raise PlanError("reference.single_device must be boolean")
        _argument_types(scenario["reference"].get("args", {}), model)
    for factor, values in matrix.items():
        if not isinstance(values, list) or not values:
            raise PlanError("matrix values must be nonempty lists")
        if any(
            isinstance(value, (dict, list))
            or value is None
            or isinstance(value, float)
            and not math.isfinite(value)
            for value in values
        ):
            raise PlanError("matrix factors must contain finite scalar values")
        serialized = [json.dumps(value, sort_keys=True) for value in values]
        if len(set(serialized)) != len(serialized):
            raise PlanError("duplicate matrix factor values for " + factor)
    _argument_types(scenario.get("args", {}), model)
    for factor, values in matrix.items():
        if factor in model["argument_types"]:
            for value in values:
                _argument_types({factor: value}, model)
    factors = sorted(matrix)
    if any(not isinstance(matrix[f], list) or not matrix[f] for f in factors):
        raise PlanError("matrix values must be nonempty lists")
    rows = [
        dict(zip(factors, values))
        for values in itertools.product(*(matrix[f] for f in factors))
    ]
    legal, excluded = [], []
    candidates = {}
    for row in rows:
        topo = dict(scenario.get("topology", {}))
        batch = dict(scenario.get("batch", {}))
        args = dict(scenario.get("args", {}))
        for key, val in row.items():
            if key in {"dp", "tp", "pp", "vpp", "launch"}:
                topo[key] = val
            elif key in {
                "micro_batch_samples",
                "sequence_length",
                "accumulation_steps",
            }:
                batch[key] = val
            else:
                args[key] = val
        _object(topo, {"dp", "tp", "pp", "vpp", "launch", "max_world_size"}, "topology")
        for key in ["dp", "tp", "pp", "vpp"]:
            _positive(topo.get(key, 1), key)
        if topo.get("launch", "threads") not in ["threads", "processes"]:
            raise PlanError("launch must be threads or processes")
        world = topo.get("dp", 1) * topo.get("tp", 1) * topo.get("pp", 1)
        suffix = (
            "/".join(
                "%s=%s" % (key, quote(str(row[key]).lower(), safe="-_."))
                for key in factors
            )
            or "default"
        )
        try:
            if world > topo.get("max_world_size", 8):
                raise UnsupportedCombination(
                    "world size exceeds scenario resource ceiling"
                )
            if model["capabilities"]["distributed"]:
                args.update(
                    tensor_parallel=topo.get("tp", 1),
                    pipeline_parallel=topo.get("pp", 1),
                    virtual_pipeline_parallel=topo.get("vpp", 1),
                    nthread_per_process=(
                        1 if topo.get("launch") == "processes" else world
                    ),
                )
            args.update(batch_args(model["adapter"], batch, topo))
            proc = world if topo.get("launch") == "processes" else 1
            case = _make_case(model, scenario, args, suffix, backend, nproc=proc)
            validate_case(case, model, tolerance_registry)
        except UnsupportedCombination as exc:
            if not matrix:
                raise
            excluded.append({"point": row, "reason": str(exc)})
            continue
        case["factors"] = row
        legal.append(row)
        candidates[json.dumps(row, sort_keys=True)] = case
    anchors = scenario.get("high_risk", [])
    if not isinstance(anchors, list):
        raise PlanError("high_risk must be a list")
    for anchor in anchors:
        if not isinstance(anchor, dict) or set(anchor) != set(matrix):
            raise PlanError(
                "high-risk anchors must specify every matrix factor exactly"
            )
        for key, value in anchor.items():
            if json.dumps(value, sort_keys=True) not in [
                json.dumps(v, sort_keys=True) for v in matrix[key]
            ]:
                raise PlanError(
                    "high-risk anchor value is outside declared matrix: " + key
                )
    if backend == "cpu":
        anchors = [a for a in anchors if a in legal]
    if scenario.get("strategy", "full") == "pairwise":
        selected, obligations = pairwise(legal, anchors)
    elif scenario.get("strategy", "full") == "full":
        selected = legal
        obligations = (
            sorted(set().union(*(pair_keys(row) for row in legal))) if legal else []
        )
    else:
        raise PlanError("strategy must be full or pairwise")
    for row in selected:
        case = candidates[json.dumps(row, sort_keys=True)]
        if row in anchors:
            case["tags"].append("high-risk")
        reference = scenario.get("reference")
        if reference:
            ref = copy.deepcopy(case)
            ref["id"] += "/reference"
            ref["tags"].append("reference")
            ref["checks"] = [{"kind": "finite_and_complete"}]
            patch = copy.deepcopy(reference.get("args", {}))
            if reference.get("single_device", False):
                ref["nproc_per_node"] = 1
                ref["nthread_per_process"] = 1
                ref["gpus"] = 0 if backend == "cpu" else 1
                if model["adapter"] == "dense":
                    patch.update(
                        nthread_per_process=1,
                        tensor_parallel=1,
                        pipeline_parallel=1,
                        virtual_pipeline_parallel=1,
                        sequence_parallel=False,
                        zero_stage=0,
                        batch_size=case["batch"]["global_samples"],
                    )
                else:
                    patch.update(micro_batch_size=case["batch"]["global_samples"])
            ref["reference_locked_args"] = copy.deepcopy(patch)
            ref["args"].update(patch)
            if "resume_step" in reference:
                split = _positive(reference["resume_step"], "resume_step")
                if split >= case["expected_steps"]:
                    raise PlanError("resume_step must precede final training step")
                ref["args"]["save"] = "artifact:%s/checkpoints" % ref["id"]
                ref["args"]["max_checkpoint_keep"] = 3
                case["args"]["load"] = (
                    "artifact:%s/checkpoints/iter_%07d" % (ref["id"], split)
                )
                case["args"]["save"] = "artifact:%s/checkpoints" % case["id"]
                case["start_step"] = split + 1
            validate_case(ref, model, tolerance_registry)
            case["dependencies"].append(ref["id"])
            case["checks"].append(
                {
                    "kind": "resume" if "resume_step" in reference else "equivalent",
                    "reference_case": ref["id"],
                    "metrics": reference.get(
                        "metrics", ["loss", "gradient", "parameter"]
                    ),
                    "tolerance": reference.get("tolerance", "distributed_equivalence"),
                }
            )
            output.append(ref)
        output.append(case)
    coverage = {
        "scenario": scenario["id"],
        "model": model["id"],
        "algorithm": (
            "greedy-v1" if scenario.get("strategy") == "pairwise" else "full-v1"
        ),
        "factors": matrix,
        "legal_candidates": len(legal),
        "selected_candidates": len(selected),
        "required_pairs": obligations,
        "covered_pairs": (
            sorted(set().union(*(pair_keys(r) for r in selected))) if selected else []
        ),
        "excluded": excluded,
        "high_risk": scenario.get("high_risk", []),
        "unavailable_high_risk": [
            a for a in scenario.get("high_risk", []) if a not in legal
        ],
    }
    return output, coverage


def _checkpoint_dependencies(cases):
    # Preserve classic save/resume topology and run parameters, replacing only paths.
    sources = {}
    for case in cases:
        raw = case["args"].get("save", "")
        if raw.startswith("@CKPT_ROOT_DIR@"):
            sources[(case["model"], raw)] = case["id"]
    for case in cases:
        args = case["args"]
        if args.get("save", "").startswith("@CKPT_ROOT_DIR@"):
            args["save"] = "artifact:%s/checkpoints" % case["id"]
        raw = args.get("load", "")
        if raw.startswith("@CKPT_ROOT_DIR@"):
            prefix, step = raw.rsplit("/", 1)
            source = sources.get((case["model"], prefix))
            if source is None:
                raise PlanError("missing checkpoint source for %s" % case["id"])
            iteration = int(step.rsplit("_", 1)[1])
            args["load"] = "artifact:%s/checkpoints/iter_%07d" % (source, iteration)
            case["dependencies"].append(source)
            case["start_step"] = int(step.rsplit("_", 1)[1]) + 1
            case["checks"].append(
                {"kind": "resume", "reference_case": source, "metrics": ["loss", "lr"]}
            )


def _history_cases(cases):
    """Keep training workloads and checkpoint producers, without diagnostic runs."""
    by_id = {case["id"]: case for case in cases}
    for case in cases:
        comparison_dependencies = {
            check["reference_case"]
            for check in case["checks"]
            if check["kind"] == "equivalent"
        }
        dependencies = set(case["dependencies"]) - comparison_dependencies
        for value in case["args"].values():
            if not isinstance(value, str) or not value.startswith("artifact:"):
                continue
            relative = value[len("artifact:"):]
            matches = [cid for cid in by_id if relative.startswith(cid + "/")]
            if matches:
                producer = max(matches, key=len)
                if producer != case["id"]:
                    dependencies.add(producer)
        case["dependencies"] = sorted(dependencies)
        case["checks"] = [{"kind": "finite_and_complete"}, {"kind": "history"}]
    needed = {
        case["id"] for case in cases
        if not (case["id"].endswith("/reference") and "reference_locked_args" in case)
    }
    def require(cid, ancestors=()):
        if cid in ancestors or cid not in by_id:
            raise PlanError("Invalid training dependency: " + cid)
        for dep in by_id[cid]["dependencies"]:
            needed.add(dep)
            require(dep, ancestors + (cid,))
    for cid in list(needed):
        require(cid)
    return [case for case in cases if case["id"] in needed]


def load_plan(
    suite="pr",
    *,
    models=None,
    tags=None,
    cases=None,
    backend="cuda",
    config_root=None,
    overrides=None,
    alignment="deep",
):
    """Resolve a suite and optional AND filters, including dependency closure.

    models/tags/cases accept a string or iterable. Case filters accept an exact
    ID or shell wildcard. Selection never silently succeeds with zero cases.
    """
    import fnmatch

    if not re.fullmatch(r"[A-Za-z0-9_-]+", suite):
        raise PlanError("invalid suite name")
    root = Path(config_root) if config_root else ROOT
    registry_path = root / "tolerances.json"
    try:
        tolerance_registry = tolerances.load_registry(registry_path if registry_path.exists() else None)
    except (ValueError, OSError) as error:
        raise PlanError("invalid tolerance registry: " + str(error)) from error
    suite_meta = _read(root / "suites" / ("%s.json" % suite))
    _object(
        suite_meta,
        {"schema_version", "id", "description", "ctest", "strict", "exclusive", "profile_mode"},
        "suite",
    )
    if type(suite_meta.get("profile_mode", False)) is not bool:
        raise PlanError("suite profile_mode must be boolean")
    manifests = {}
    for path in sorted((root / "models").glob("*.json")):
        model = _read(path)
        _object(model, MODEL_KEYS, str(path))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model["id"]):
            raise PlanError("invalid model ID")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model["binary"]):
            raise PlanError("invalid model binary")
        _argument_types(model["defaults"], model)
        for model_backend, dtypes in model.get("backend_dtypes", {}).items():
            if (
                model_backend not in model["backends"]
                or not isinstance(dtypes, list)
                or not dtypes
                or any(dtype not in model["dtypes"] for dtype in dtypes)
                or len(set(dtypes)) != len(dtypes)
            ):
                raise PlanError(
                    "invalid backend-specific dtype capability for " + model_backend
                )
        for profile in model["profiles"].values():
            _object(profile, {"fixtures", "shape", "args"}, "profile")
            _object(
                profile["shape"],
                {"layers", "hidden", "heads", "max_sequence_length"},
                "shape",
            )
            for value in profile["shape"].values():
                _positive(value, "shape dimension")
        if model["id"] in manifests:
            raise PlanError("duplicate model ID")
        manifests[model["id"]] = model
    selected, reports, all_ids = [], [], set()
    for path in sorted((root / "scenarios").glob("*.json")):
        scenario = _read(path)
        _object(scenario, SCENARIO_KEYS, str(path))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scenario["id"]):
            raise PlanError("invalid scenario ID")
        if suite not in scenario["suites"]:
            continue
        if scenario["id"] in all_ids:
            raise PlanError("duplicate scenario ID " + scenario["id"])
        all_ids.add(scenario["id"])
        for name in scenario["models"]:
            if name not in manifests:
                raise PlanError("unknown model " + name)
            expanded, report = _expand(scenario, manifests[name], backend, tolerance_registry)
            selected.extend(expanded)
            if report:
                reports.append(report)
    _checkpoint_dependencies(selected)
    if alignment == "history":
        selected = _history_cases(selected)
    elif alignment != "deep":
        raise PlanError("Unknown alignment mode: " + str(alignment))
    by_id = {}
    for case in selected:
        if case["id"] in by_id:
            raise PlanError("duplicate case ID " + case["id"])
        case["dtype"] = case["args"].get("dtype", case["dtype"])
        case["expected_steps"] = case["args"].get(
            "num_iteration", case["expected_steps"]
        )
        validate_case(case, manifests[case["model"]], tolerance_registry)
        for check in case["checks"]:
            if (
                isinstance(check, dict)
                and check.get("kind") in {"equivalent", "resume"}
                and isinstance(check.get("reference_case"), str)
                and check["reference_case"] not in case["dependencies"]
            ):
                case["dependencies"].append(check["reference_case"])
        by_id[case["id"]] = case

    def many(value):
        return [value] if isinstance(value, str) else list(value or [])

    mf, tf, cf = many(models), many(tags), many(cases)
    chosen = [
        c
        for c in selected
        if (not mf or c["model"] in mf)
        and (not tf or set(tf) <= set(c["tags"]))
        and (not cf or any(fnmatch.fnmatchcase(c["id"], f) for f in cf))
    ]
    direct = {c["id"] for c in chosen}
    needed = set(direct)

    def include(cid, stack):
        if cid in stack:
            raise PlanError("cyclic dependencies " + cid)
        if cid not in by_id:
            raise PlanError("missing dependency " + cid)
        for dep in by_id[cid]["dependencies"]:
            needed.add(dep)
            include(dep, stack + [cid])

    for cid in list(needed):
        include(cid, [])
    ordered, done = [], set()

    def visit(cid):
        if cid in done:
            return
        for dep in by_id[cid]["dependencies"]:
            visit(dep)
        case = by_id[cid]
        if cid not in direct:
            case["selection_reason"] = ["dependency closure"]
        ordered.append(case)
        done.add(cid)

    for case in selected:
        if case["id"] in needed:
            visit(case["id"])
    if not ordered:
        raise PlanError("selection contains no executable cases")
    for case in ordered:
        if overrides:
            protected = {
                "input_bin",
                "llmc_filepath",
                "save",
                "load",
                "nthread_per_process",
                "device",
                "metrics_file",
                "regression_snapshot_dir",
                "tensor_parallel",
                "pipeline_parallel",
                "virtual_pipeline_parallel",
                "sequence_parallel",
                "batch_size",
                "total_batch_size",
                "micro_batch_size",
                "global_batch_size",
                "sequence_length",
            }
            if set(overrides) & protected or any(
                _is_managed_path(key) for key in overrides
            ):
                raise PlanError(
                    "override changes managed path/topology arguments; use --fixture ID=PATH or scenario artifact declarations"
                )
            case["args"].update(overrides)
            case["args"].update(case.get("reference_locked_args", {}))
            validate_case(case, manifests[case["model"]], tolerance_registry)
            for factor in case.get("factors", {}):
                if factor in case["resolved_args"]:
                    case["factors"][factor] = case["resolved_args"][factor]
    for report in reports:
        rows = [
            c.get("factors", {})
            for c in ordered
            if c["scenario"] == report["scenario"]
            and c["model"] == report["model"]
            and "reference" not in c["tags"]
        ]
        covered = set().union(*(pair_keys(r) for r in rows)) if rows else set()
        report["covered_pairs"] = sorted(covered)
        report["uncovered_pairs"] = sorted(set(report["required_pairs"]) - covered)
    quarantines = _apply_quarantines(root, ordered, manifests, tolerance_registry)
    return {
        "schema_version": 1,
        "suite": suite,
        "alignment": alignment,
        "suite_metadata": suite_meta,
        "cases": ordered,
        "coverage": {
            "quarantines": quarantines,
            "planned_cases": len(ordered),
            "suite_cases": len(selected),
            "partial": len(ordered) != len(selected) or bool(overrides),
            "argument_overrides": dict(overrides or {}),
            "omitted_case_ids": [c["id"] for c in selected if c["id"] not in needed],
            "matrices": reports,
            "unsupported_capabilities": [
                "zero3",
                "offload",
                "recompute",
                "context_parallel",
            ],
            "required_case_ids": [c["id"] for c in selected],
        },
        "models": manifests,
    }


def _apply_quarantines(root, cases, manifests, tolerance_registry=None):
    """Annotate owned, expiring quarantine requests; never silently drop a gate."""
    from datetime import date

    path = root / "quarantines.json"
    if not path.exists():
        return []
    document = _read(path)
    _object(document, {"schema_version", "quarantines"}, "quarantines")
    entries = document.get("quarantines")
    if not isinstance(entries, list):
        raise PlanError("quarantines must be a list")
    if not entries:
        return []
    # Validate against all declared suites, including accelerator cases when the
    # current invocation is CPU-only. Stale IDs cannot hide behind suite filters.
    known = set()
    for scenario_path in sorted((root / "scenarios").glob("*.json")):
        scenario = _read(scenario_path)
        for name in scenario["models"]:
            expanded, _ = _expand(scenario, manifests[name], "cuda", tolerance_registry)
            known.update(case["id"] for case in expanded)
    active = {}
    for item in entries:
        _object(item, {"case_id", "owner", "reason", "issue", "expires"}, "quarantine")
        for key in ("case_id", "owner", "reason", "issue", "expires"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise PlanError("quarantine requires a nonempty " + key)
        if item["case_id"] not in known:
            raise PlanError("quarantine references unknown case " + item["case_id"])
        if item["case_id"] in active:
            raise PlanError("duplicate quarantine for " + item["case_id"])
        try:
            expiry = date.fromisoformat(item["expires"])
        except ValueError as exc:
            raise PlanError("quarantine expires must be YYYY-MM-DD") from exc
        if expiry < date.today():
            raise PlanError("expired quarantine for " + item["case_id"])
        active[item["case_id"]] = copy.deepcopy(item)
    selected_ids = {case["id"] for case in cases}
    for case in cases:
        if case["id"] in active:
            case["quarantine"] = copy.deepcopy(active[case["id"]])
    return [
        dict(
            item,
            selected=item["case_id"] in selected_ids,
            policy="annotate-only; required checks remain required",
        )
        for item in active.values()
    ]


def select_changed(plan, changed_paths):
    """Select changed PR surfaces plus every fixed anchor and dependency.

    Common/unknown paths and an empty diff conservatively keep the full PR suite.
    Other suites are never reduced. Input must be unfiltered so fixed anchors
    cannot disappear due to another filter before change selection.
    """
    result = copy.deepcopy(plan)
    paths = sorted(set(str(path).replace("\\", "/") for path in changed_paths))
    policy = {"paths": paths, "policy": "anchors-plus-affected-v1", "fallback": None}
    result["coverage"]["change_selection"] = policy
    if plan["suite"] != "pr":
        policy["fallback"] = "change selection applies only to pr; full suite retained"
        return result
    if plan["coverage"].get("partial"):
        raise PlanError(
            "--changed requires an unfiltered PR plan so fixed anchors are preserved"
        )
    if not paths:
        policy["fallback"] = "empty diff; full PR suite retained"
        return result
    models, scenarios = set(), set()
    declared_scenarios = {case["scenario"] for case in plan["cases"]}
    for path in paths:
        parts = Path(path).parts
        if path.startswith("/") or ".." in parts:
            policy["fallback"] = "unrecognized path; full PR suite retained: " + path
            return result
        if len(parts) >= 3 and parts[0] == "example" and parts[1] in plan["models"]:
            models.add(parts[1])
        elif (
            len(parts) == 4
            and parts[:3] == ("tests", "regression", "models")
            and Path(parts[3]).stem in plan["models"]
        ):
            models.add(Path(parts[3]).stem)
        elif (
            len(parts) == 4
            and parts[:3] == ("tests", "regression", "scenarios")
            and Path(parts[3]).stem in declared_scenarios
        ):
            scenarios.add(Path(parts[3]).stem)
        else:
            policy["fallback"] = (
                "common or unknown path; full PR suite retained: " + path
            )
            return result
    by_id = {case["id"]: case for case in result["cases"]}
    chosen = set()
    for case in result["cases"]:
        reasons = []
        if "anchor" in case["tags"]:
            reasons.append("fixed anchor")
        if case["model"] in models:
            reasons.append("changed model surface")
        if case["scenario"] in scenarios:
            reasons.append("changed scenario")
        if reasons:
            case["selection_reason"] = reasons
            chosen.add(case["id"])

    def require(cid, ancestors=()):
        if cid in ancestors:
            raise PlanError("cyclic dependency in changed plan: " + cid)
        if cid not in by_id:
            raise PlanError("missing dependency in changed plan: " + cid)
        for dep in by_id[cid]["dependencies"]:
            if dep not in chosen:
                by_id[dep]["selection_reason"] = ["dependency closure"]
                chosen.add(dep)
            require(dep, ancestors + (cid,))

    for cid in list(chosen):
        require(cid)
    if not chosen:
        policy["fallback"] = "no matching cases or anchors; full PR suite retained"
        return result
    result["cases"] = [case for case in result["cases"] if case["id"] in chosen]
    coverage = result["coverage"]
    required = coverage["required_case_ids"]
    coverage.update(
        planned_cases=len(result["cases"]),
        partial=len(chosen) != len(required),
        omitted_case_ids=[cid for cid in required if cid not in chosen],
    )
    policy.update(
        affected_models=sorted(models),
        affected_scenarios=sorted(scenarios),
        anchor_case_ids=[
            case["id"] for case in result["cases"] if "anchor" in case["tags"]
        ],
    )
    for report in coverage["matrices"]:
        rows = [
            case.get("factors", {})
            for case in result["cases"]
            if case["scenario"] == report["scenario"]
            and case["model"] == report["model"]
            and "reference" not in case["tags"]
        ]
        covered = set().union(*(pair_keys(row) for row in rows)) if rows else set()
        report["covered_pairs"] = sorted(covered)
        report["uncovered_pairs"] = sorted(set(report["required_pairs"]) - covered)
    for item in coverage.get("quarantines", []):
        item["selected"] = item["case_id"] in chosen
    return result

"""Named numerical tolerance policies, independent of execution/scheduling.

Edit tolerances.json to add a policy. Each policy has complete dtype defaults,
optional metric overrides, then optional backend+metric overrides. There is no
inheritance. Declared backends without an override use the generic metric/dtype
policy; unregistered backend names fail. ``privateuse1`` aliases ``privateuseone``.

Integer metrics and integer tensor states remain exact: consumed_samples always
resolves to zero; checks.compare_snapshots independently treats integral state
exactly regardless of the floating optimizer tolerance returned here.
"""

from collections.abc import Mapping
import json
import math
from pathlib import Path
import re


DEFAULT_PATH = Path(__file__).with_name("tolerances.json")
METRICS = frozenset(("loss", "lr", "consumed_samples", "parameter", "gradient", "optimizer"))
DTYPES = frozenset(("float32", "bfloat16"))
_DTYPE_ALIASES = {"fp32": "float32", "bf16": "bfloat16"}
_BACKEND_ALIASES = {"privateuse1": "privateuseone"}
# This code-defined snapshot is deliberately independent from editable JSON.
# The planner can detect policy changes without invalidating historical case
# hashes when the initially shipped numerical behavior is unchanged.
_HISTORICAL_REGISTRY = {
    "schema_version": 1,
    "backends": ["cpu", "cuda", "privateuseone"],
    "policies": {
        name: {"defaults": {"float32": {"atol": 1e-5, "rtol": 1e-5},
                            "bfloat16": {"atol": 1e-2, "rtol": 1e-2}}}
        for name in ("distributed_equivalence", "oracle")
    },
}
_HISTORICAL_REGISTRY["policies"]["lr_schedule"] = {
    "defaults": {dtype: {"atol": 1e-9, "rtol": 1e-5} for dtype in DTYPES}
}


def _backend(name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Invalid tolerance backend name: " + repr(name))
    return _BACKEND_ALIASES.get(name, name)


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(label + " must be an object")
    return value


def _tolerance(value, label):
    value = _mapping(value, label)
    if set(value) != {"atol", "rtol"}:
        raise ValueError(label + " requires exactly atol and rtol")
    for name, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise ValueError(label + "/" + name + " must be finite and nonnegative")


def _dtype_map(value, label, *, complete=False, metric=None):
    value = _mapping(value, label)
    if not value or set(value) - DTYPES or complete and set(value) != DTYPES:
        raise ValueError(label + " has missing/unknown dtype entries")
    for dtype, tolerance in value.items():
        _tolerance(tolerance, label + "/" + dtype)
        if metric == "consumed_samples" and any(tolerance.values()):
            raise ValueError("consumed_samples is an exact integer metric; its tolerance must be zero")


def _metric_map(value, label):
    value = _mapping(value, label)
    if set(value) - METRICS:
        raise ValueError(label + " has unknown metrics: " + repr(sorted(set(value) - METRICS)))
    for metric, dtypes in value.items():
        _dtype_map(dtypes, label + "/" + metric, metric=metric)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate tolerance JSON key: " + key)
        result[key] = value
    return result


def load_registry(registry=None):
    """Read and validate a JSON path or mapping. Call once per planning operation.

    Pass the returned mapping to resolve repeatedly to avoid repeated filesystem
    reads. A fresh plan reads current bytes so edited policies affect its hash.
    """
    if registry is None:
        registry = DEFAULT_PATH
    if isinstance(registry, (str, Path)):
        with Path(registry).open() as stream:
            registry = json.load(stream, object_pairs_hook=_unique_object)
    registry = _mapping(registry, "Tolerance registry")
    if set(registry) != {"schema_version", "backends", "policies"}:
        raise ValueError("Tolerance registry requires schema_version, backends and policies only")
    if type(registry["schema_version"]) is not int or registry["schema_version"] != 1:
        raise ValueError("Unsupported tolerance registry schema")
    backends = registry["backends"]
    if not isinstance(backends, list) or not backends:
        raise ValueError("Tolerance registry needs a nonempty backend list")
    normalized_backends = [_backend(name) for name in backends]
    if len(set(normalized_backends)) != len(normalized_backends):
        raise ValueError("Duplicate tolerance backend or alias")
    policies = _mapping(registry["policies"], "Tolerance policies")
    if not policies:
        raise ValueError("Tolerance registry has no policies")
    for name, policy in policies.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name):
            raise ValueError("Invalid tolerance policy name: " + repr(name))
        policy = _mapping(policy, "Policy " + name)
        if "defaults" not in policy or set(policy) - {"defaults", "metrics", "backends"}:
            raise ValueError("Policy " + name + " requires defaults and allows only metrics/backends overrides")
        _dtype_map(policy["defaults"], "Policy " + name + "/defaults", complete=True)
        _metric_map(policy.get("metrics", {}), "Policy " + name + "/metrics")
        overrides = _mapping(policy.get("backends", {}), "Policy " + name + "/backends")
        seen = set()
        for backend, metrics in overrides.items():
            backend = _backend(backend)
            if backend not in normalized_backends:
                raise ValueError("Policy " + name + " overrides unregistered backend " + backend)
            if backend in seen:
                raise ValueError("Duplicate backend alias override in policy " + name)
            seen.add(backend)
            _metric_map(metrics, "Policy " + name + "/backends/" + backend)
    return registry


def resolve(policy_name, dtype, metric, backend, registry=None):
    """Return {atol,rtol}; unknown names and malformed policies raise ValueError."""
    registry = load_registry(registry)
    if not all(isinstance(value, str) for value in (policy_name, dtype, metric, backend)):
        raise ValueError("Tolerance policy, dtype, metric and backend must be names")
    dtype = _DTYPE_ALIASES.get(dtype, dtype)
    if dtype not in DTYPES:
        raise ValueError("Unknown tolerance dtype: " + repr(dtype))
    if metric not in METRICS:
        raise ValueError("Unknown tolerance metric: " + repr(metric))
    backend = _backend(backend)
    if backend not in {_backend(name) for name in registry["backends"]}:
        raise ValueError("Unknown tolerance backend: " + backend)
    if policy_name not in registry["policies"]:
        raise ValueError("Unknown tolerance policy: " + repr(policy_name))
    policy = registry["policies"][policy_name]
    selected = dict(policy["defaults"][dtype])
    selected.update(policy.get("metrics", {}).get(metric, {}).get(dtype, {}))
    for name, overrides in policy.get("backends", {}).items():
        if _backend(name) == backend:
            selected.update(overrides.get(metric, {}).get(dtype, {}))
    if metric == "consumed_samples":
        return {"atol": 0, "rtol": 0}
    return selected


def resolve_historical_default(policy_name, dtype, metric, backend):
    """Return the original shipped behavior, unaffected by tolerances.json edits."""
    return resolve(policy_name, dtype, metric, backend, _HISTORICAL_REGISTRY)

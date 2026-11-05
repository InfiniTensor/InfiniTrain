# Configuration contract (version 1)

The planner uses Python's standard library and reads all `models/*.json`,
`scenarios/*.json`, and the requested `suites/<name>.json`. Unknown fields and
unknown CLI arguments fail before resource allocation. JSON is deliberate: no
YAML dependency, implicit coercion, executable expressions, or inheritance graph.

## Add a test

Add one scenario JSON, for example:

```json
{
  "schema_version": 1,
  "id": "zero2_accumulation",
  "models": ["gpt2", "llama3"],
  "profile": "tiny",
  "tags": ["optimizer", "zero2", "accumulation"],
  "suites": ["pr", "nightly", "release"],
  "matrix": {"dtype": ["float32", "bfloat16"]},
  "topology": {"dp": 2, "launch": "processes"},
  "batch": {
    "micro_batch_samples": 2,
    "sequence_length": 16,
    "accumulation_steps": 4
  },
  "args": {"zero_stage": 2, "num_iteration": 4},
  "reference": {
    "single_device": true,
    "metrics": ["loss", "gradient", "parameter"]
  },
  "timeout_seconds": 300
}
```

`args` are actual model flags, without `--`. Scalar matrix factors override the
same scalar in args/topology/batch. Merge order is model profile args, scenario
args, matrix point, derived topology/batch, explicit runner overrides. Do not
specify batch/topology-derived flags directly in `args`: use their semantic
fields. `batch` is always samples + sequence length; the model adapter translates
it into dense token-count flags or Mixtral sample-count flags. `world=DP*TP*PP`;
SP and VPP do not multiply world size. Launch is `threads` or `processes`.

In `--alignment deep` mode, `reference.single_device` generates an explicit dependency using the same
fixture, global batch, learning rate and iteration count, but one device and
one microbatch per update. `reference.args` can instead patch a particular flag.
`reference.resume_step` generates continuous and resumed runs and compares the
post-checkpoint suffix, with non-overfit data for cursor restoration tests.
All reference cases are listed in the resolved plan and cost/resource estimates.
Default `history` mode keeps checkpoint producers but omits additional numerical
comparison runs.

`strategy` defaults to `full`. For `pairwise`, the deterministic `greedy-v1`
algorithm covers all pairs present in the legal candidate set. `high_risk` lists
complete matrix points that are mandatory regardless of greedy selection.
`topology.max_world_size` bounds candidate resource usage. Invalid combinations
are retained with reasons in coverage output. CPU excludes accelerator-only
points explicitly and lists unavailable high-risk anchors. Pairwise coverage is
not a claim about arbitrary three/four-factor interactions.

The classic migration uses `templates` with shared args and explicit `variants`
solely to preserve every original command and semantic ID. New tests should use
matrix/scenario syntax rather than duplicate classic templates.

Suite metadata may declare `profile_mode: true` for a dedicated instrumented
build. This is distinct from a scenario's asset `profile` (`tiny` / `classic`).
The `perf` suite uses this mode for `kernel_basic` and `kernel_ddp`: two dtypes,
one execution of ten steps, identical per-device shapes, and DP=1/2 only.
Mixtral currently has no DDP workload. Other suites use normal builds; release
does not implicitly include profiling jobs. Existing optional `performance`
sampling contracts below are not used by these kernel profiling scenarios.

## Add a model

Add a manifest with `id`, `binary`, `adapter` (`dense` or `mixtral`), supported
backends/dtypes, `argument_types` (gflags types), explicit effective defaults,
capabilities, and profiles. Each profile declares fixture IDs and shape
(`layers`, `hidden`, `heads`) used for topology constraints. Register fixture
checksums/acquisition metadata separately. A model using an existing adapter
requires no scheduler changes; a genuinely different CLI needs a new adapter.

## Checks and identities

Checks are structured objects with `kind`. Common kinds are
`finite_and_complete`, `oracle`, `equivalent`, `resume`, `lr_schedule`,
`lora_frozen`, and `performance`. Unknown checks must fail execution, never be
ignored. In the default `history` alignment mode, the planner replaces diagnostic
checks with complete-log validation and previous-log loss/throughput comparison.
Generated single-device comparison runs are removed; checkpoint producers remain
dependencies. No tensor snapshots or PyTorch oracle are required in this mode.
Explicit `--alignment deep` mode adds the scenario's numerical checks and
same-run reference dependencies while retaining previous-log comparison.
Both modes use historical run directories for loss and throughput comparison.
Performance specifies warmup/measurement windows and independent repetitions;
previous-log throughput comparison uses the mean logged token rate after the
first step, as described in the usage guide.

Case IDs consist of scenario/model/matrix-point. A separate SHA256 hash covers
resolved effective arguments, profile, fixtures, topology and checks; changing a
learning rate does not silently reuse an incompatible historical run. Paths remain
`fixture:<id>` or `artifact:<case-id>/<relative-path>` until execution. Committed
configuration never embeds machine dataset/build/checkpoint paths.

`load_plan(suite='pr', models=None, tags=None, cases=None, backend='cuda',
config_root=None, overrides=None)` returns suite metadata, ordered cases,
model manifests, and coverage. Filters are ANDed; multiple tags require all tags,
case filters accept shell wildcards. Dependencies are automatically included.
Empty selection is an error. Partial selection lists omitted required IDs and
uncovered pairs. Raw planned coverage is never presented as executed coverage.

## Change selection and quarantine ownership

`select_changed(plan, changed_paths)` reduces only an unfiltered PR plan. Model
source/manifests select that model; scenario paths select that scenario. Every
fixed `anchor` and transitive reference remains selected. Shared framework,
unknown/deleted paths, and an empty diff conservatively keep the full PR suite.
Nightly/perf/release are never reduced. Combining change selection with an
already partial plan is rejected because otherwise fixed anchors could vanish.
Coverage retains the complete required ID list, omitted IDs and uncovered pairs.

An optional `quarantines.json` records temporary ownership without deleting a
required test:

```json
{
  "schema_version": 1,
  "quarantines": [{
    "case_id": "tiny_smoke/gpt2/default",
    "owner": "training-team",
    "reason": "Intermittent issue under investigation",
    "issue": "#123",
    "expires": "2099-01-01"
  }]
}
```

Use a short real expiry, not the illustrative date above. Empty owner/reason/
issue, duplicate or unknown case IDs, invalid dates and expired entries are
configuration errors. Entries remain visible even when not selected. The
planner annotates cases and does **not** suppress execution; strict gates still
require their checks to pass. Any executor implementing an explicit quarantine
skip must report SKIP and must not count that result as satisfying a strict gate.

Release `long_stability` cases additionally check sustained per-step allocator
peak drift after warmup. This uses `peak_memory_bytes`; it is a useful regression
signal, not a proof that all retained-allocation leaks are absent.


Malformed configuration is distinct from an unsupported combination: unknown
arguments/factors, invalid flag types or enums, NaN/Inf, integer overflow and
duplicate matrix values fail the whole plan. Legitimate constraints (CPU cannot
run distributed training, TP/SP/PP/VPP divisibility, tiny fixture sequence
capacity) produce explicit matrix exclusions. A single explicit illegal scenario
still fails instead of disappearing.

CLI scalar overrides apply after selection and to its dependency closure. They
preserve reference transforms (for example ZeRO-0 and the single-device batch
adapter), and cannot set managed topology/batch/path flags. Overrides mark the
run partial even when IDs stay the same, so a strict canonical gate cannot be
silently shortened or changed. Use scenario JSON to define another canonical
configuration. Sequence length is checked against tiny fixture capacity.


Check contracts are validated during planning, before GPU allocation. Checks
must be nonempty and use a supported kind/field set. Explicit equivalence/resume
references join the dependency graph and must resolve without cycles. Unknown
metric/tolerance policies are rejected instead of ignored. Performance contracts
require at least three independent repetitions, consistent duplicate sampling
fields, finite budgets and enough steps for warmup plus the measured window.
Memory drift needs warmup plus two windows. The oracle accepts only its fixed
FP32 tiny fixture, one rank, no adapters/sharded optimizer/resume,
and a constant learning rate. Microbatch accumulation and token-stream epoch
wrapping are supported; each microbatch must fit the fixed token fixture.
Unsupported requests fail rather than modifying
the candidate into a different test. Nonpositive timeouts fail planning.


The current DistributedOptimizer requires **DP > 1** for ZeRO-1/2. A model-parallel
world with TP/PP but DP=1 does not satisfy this requirement. Such combinations
are excluded explicitly before launch. Four-device TP2×PP2 anchors use ZeRO-0;
separate DP2 accumulation scenarios cover Adam ZeRO-1/2 in Llama3 and Qwen3,
while the GPT-2 pairwise matrix covers the supported distributed SGD paths.


Manifests declare `backend_dtypes` separately from the model's overall dtype
list. Current CPU training is FP32-only because CPU Linear uses float Eigen
views and FP32 output allocation and CPU Matmul reads float pointers. CUDA
retains FP32/BF16. CPU BF16 points are reported as unsupported combinations,
never coerced into FP32 cases. An explicit unsupported request fails planning.

All input-file arguments, including LoRA weights and validation/tokenizer data,
must use `fixture:ID`. Their effective IDs are recomputed during validation,
including reference patches, and actual bytes are hashed by asset preparation.
Use `--fixture ID=PATH` for a local path; `--set` rejects every managed path flag.
Scenario input/model-weight arguments cannot silently override profile fixture
assignments. Checkpoint/output paths are run-local `artifact:` declarations.
